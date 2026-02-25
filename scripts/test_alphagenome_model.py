#!/usr/bin/env python3
"""
Test AlphaGenome model in multi-GPU context.

Since basic NCCL tests pass but training fails, the issue must be
in the AlphaGenome model itself or how we're using it.

NOTE: This test uses DIRECT checkpoint loading (not AlphaGenomeModel wrapper)
because the wrapper enforces single-device contexts.
See AlphaGenome docs: https://www.alphagenomedocs.com/index.html
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import optax
import haiku as hk
import jmp
import orbax.checkpoint as ocp
from alphagenome_research.model import model as model_lib, dna_model
from alphagenome_research.model.metadata import metadata as metadata_lib

print("=" * 70)
print("ALPHAGENOME MODEL MULTI-GPU DIAGNOSTIC")
print("=" * 70)
print("  Reference: https://www.alphagenomedocs.com/index.html")
print("  Pattern: alphagenome_research/evals/track_prediction.py")

# Setup
mesh = Mesh(jax.devices(), ('batch',))
batch_sharding = NamedSharding(mesh, P('batch'))
replicated = NamedSharding(mesh, P())

# Load AlphaGenome checkpoint (direct orbax loading)
print("\n[Step 1] Loading AlphaGenome checkpoint (direct orbax)...")
checkpoint_dir = '/home/ray/AG_Finetune/checkpoints/pre_trained/alphagenome-all-folds'
try:
   # Pattern from alphagenome_research/evals/track_prediction.py:74
    params, state = ocp.StandardCheckpointer().restore(checkpoint_dir)
    print(f"✓ Checkpoint loaded: {len(jax.tree.leaves(params))} param leaves, {len(jax.tree.leaves(state))} state leaves")
except Exception as e:
    print(f"✗ Failed to load checkpoint: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Load metadata for mouse
print("\n[Step 2] Loading metadata for mouse...")
try:
    metadata = {
        dna_model.Organism.MUS_MUSCULUS: metadata_lib.load(dna_model.Organism.MUS_MUSCULUS)
    }
    print(f"✓ Metadata loaded for {list(metadata.keys())}")
except Exception as e:
    print(f"✗ Failed to load metadata: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Create Haiku transform (pattern from track_prediction.py:79-83)
print("\n[Step 3] Creating Haiku transform...")
try:
    @hk.transform_with_state
    def forward(dna_sequence, organism_index):
        # Use float32 precision policy (safe, no bf16 NaN issues)
        policy = jmp.get_policy('params=float32,compute=float32,output=float32')
        with hk.mixed_precision.push_policy(model_lib.AlphaGenome, policy):
            return model_lib.AlphaGenome(metadata)(dna_sequence, organism_index)
    
    print("✓ Haiku transform created")
except Exception as e:
    print(f"✗ Failed to create transform: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 1: Forward pass on single device (no sharding)
print("\n[Test 1] Forward pass on single device (no NCCL)...")
try:
    test_seq = jnp.ones((1, 131072, 4), dtype=jnp.float32)
    test_org = jnp.array([1], dtype=jnp.int32)  # Mouse = 1
    
    # Pattern from track_prediction.py:89-91
    (predictions, _), _ = forward.apply(params, state, None, test_seq, test_org)
    predictions = dna_model.extract_predictions(predictions)
    # predictions[OutputType.ATAC] is the raw 1bp array, not an object
    atac_1bp = predictions[dna_model.OutputType.ATAC]
    print(f"✓ Forward pass works: ATAC shape {atac_1bp.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Forward pass with sharded input
print("\n[Test 2] Forward pass with sharded batch (triggers NCCL)...")
try:
    test_seq = jnp.ones((8, 131072, 4), dtype=jnp.float32)
    test_org = jnp.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.int32)
    
    # Shard inputs
    test_seq = jax.device_put(test_seq, batch_sharding)
    test_org = jax.device_put(test_org, NamedSharding(mesh, P('batch')))
    
    # Put params and state on all devices (replicated)
    params_rep = jax.device_put(params, replicated)
    state_rep = jax.device_put(state, replicated)
    
    (predictions, _), _ = forward.apply(params_rep, state_rep, None, test_seq, test_org)
    predictions = dna_model.extract_predictions(predictions)
    atac_1bp = predictions[dna_model.OutputType.ATAC]  # Raw array
    print(f"✓ Sharded forward pass works: ATAC shape {atac_1bp.shape}")
except Exception as e:
    print(f"✗ Sharded forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n  → This is where NCCL corruption happens!")
    exit(1)

# Test 3: Jitted forward pass with sharded input
print("\n[Test 3] Jitted forward pass with sharding...")
try:
    @jax.jit
    def jitted_forward(p, s, seq, org):
        (preds, _), new_s = forward.apply(p, s, None, seq, org)
        return dna_model.extract_predictions(preds)
    
    test_seq = jnp.ones((8, 131072, 4), dtype=jnp.float32)
    test_org = jnp.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.int32)
    
    test_seq = jax.device_put(test_seq, batch_sharding)
    test_org = jax.device_put(test_org, NamedSharding(mesh, P('batch')))
    params_rep = jax.device_put(params, replicated)
    state_rep = jax.device_put(state, replicated)
    
    predictions = jitted_forward(params_rep, state_rep, test_seq, test_org)
    atac_1bp = predictions[dna_model.OutputType.ATAC]  # Raw array
    print(f"✓ Jitted forward pass works: ATAC shape {atac_1bp.shape}")
except Exception as e:
    print(f"✗ Jitted forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n  → NCCL fails during JIT compilation of forward.apply!")
    exit(1)

# Test 4: Gradient computation
print("\n[Test 4] Gradient computation (like training)...")
try:
    def loss_fn(p, s, seq, org, target):
        (preds, _), _ = forward.apply(p, s, None, seq, org)
        preds = dna_model.extract_predictions(preds)
        atac_1bp = preds[dna_model.OutputType.ATAC]  # Raw array: (batch, seq_len, 256 tracks)
        # For testing, just use first track or sum across tracks
        return jnp.sum((atac_1bp[..., 0] - target) ** 2)  # Use first track only
    
    @jax.jit
    def grad_fn(p, s, seq, org, target):
        return jax.grad(loss_fn, argnums=0)(p, s, seq, org, target)
    
    test_seq = jnp.ones((8, 131072, 4), dtype=jnp.float32)
    test_org = jnp.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.int32)
    test_target = jnp.ones((8, 131072), dtype=jnp.float32)  # Single track target
    
    test_seq = jax.device_put(test_seq, batch_sharding)
    test_org = jax.device_put(test_org, NamedSharding(mesh, P('batch')))
    test_target = jax.device_put(test_target, NamedSharding(mesh, P('batch', None)))
    params_rep = jax.device_put(params, replicated)
    state_rep = jax.device_put(state, replicated)
    
    print("  Computing gradients (first compilation, may take 2-5 minutes)...")
    grads = grad_fn(params_rep, state_rep, test_seq, test_org, test_target)
    print(f"✓ Gradient computation works")
except Exception as e:
    print(f"✗ Gradient computation failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n  → NCCL fails during gradient AllReduce!")
    exit(1)

# Test 5: Full training step with optimizer
print("\n[Test 5] Full training step with optimizer.update...")
try:
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(params)
    opt_state = jax.device_put(opt_state, replicated)
    
    def loss_fn(p, s, seq, org, target):
        (preds, _), _ = forward.apply(p, s, None, seq, org)
        preds_extracted = dna_model.extract_predictions(preds)
        atac_1bp = preds_extracted[dna_model.OutputType.ATAC]  # Raw array: (batch, seq_len, 256)
        # Use first track for testing
        return jnp.sum((atac_1bp[..., 0] - target) ** 2), preds_extracted
    
    @jax.jit
    def train_step(p, s, opt_st, seq, org, target):
        (loss, preds), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, s, seq, org, target)
        updates, new_opt_st = optimizer.update(grads, opt_st, p)
        new_p = optax.apply_updates(p, updates)
        return loss, preds, new_p, new_opt_st
    
    test_seq = jnp.ones((8, 131072, 4), dtype=jnp.float32)
    test_org = jnp.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.int32)
    test_target = jnp.ones((8, 131072), dtype=jnp.float32)
    
    test_seq = jax.device_put(test_seq, batch_sharding)
    test_org = jax.device_put(test_org, NamedSharding(mesh, P('batch')))
    test_target = jax.device_put(test_target, NamedSharding(mesh, P('batch', None)))
    params_rep = jax.device_put(params, replicated)
    state_rep = jax.device_put(state, replicated)
    
    print("  Running full training step (compile + optimizer.update)...")
    loss, preds, new_params, new_opt_state = train_step(
        params_rep, state_rep, opt_state, test_seq, test_org, test_target
    )
    print(f"✓ Full training step works: loss={jax.device_get(loss)}")
except Exception as e:
    print(f"✗ Full training step failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n  → This matches your training error!")
    print(f"  → Error type: {type(e).__name__}")
    print(f"  → Check if params have correct structure or if there's a nested jit issue")
    exit(1)

print("\n" + "=" * 70)
print("✓ ALL ALPHAGENOME TESTS PASSED")
print("=" * 70)
print("""
If this passes, the issue must be in:
  1. The actual training data (TensorFlow dataset)
  2. The collate_fn or data preprocessing
  3. How we're modifying the model (head replacement / LoRA)
  4. The loss function (Borzoi loss implementation)

Next: Test with actual training data and modified model.
""")
