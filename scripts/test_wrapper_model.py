#!/usr/bin/env python3
"""
Test our HeadReplaceAlphaGenomeModel wrapper in multi-GPU context.

Since basic NCCL and AlphaGenome model work, test our wrapper code.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import sys
sys.path.insert(0, '/home/ray/AG_Finetune')

from models.model import HeadReplaceAlphaGenomeModel
from utils.borzoi_loss import cut_and_tag_loss_baskerville

print("=" * 70)
print("HEADREPLACE MODEL MULTI-GPU DIAGNOSTIC")
print("=" * 70)

# Setup
mesh = Mesh(jax.devices(), ('batch',))
batch_sharding = NamedSharding(mesh, P('batch'))
organism_sharding = NamedSharding(mesh, P('batch'))
signal_sharding = NamedSharding(mesh, P('batch', None))
param_sharding = NamedSharding(mesh, P())

print("\n[Step 1] Initializing HeadReplaceAlphaGenomeModel...")
try:
    model = HeadReplaceAlphaGenomeModel(
        checkpoint_dir='/home/ray/AG_Finetune/checkpoints/pre_trained/alphagenome-all-folds',
        learning_rate=5e-4,
        warmup_steps=1000,
        total_steps=50000,
        mesh=mesh,
        batch_sharding=batch_sharding,
        organism_sharding=organism_sharding,
        signal_sharding=signal_sharding,
        param_sharding=param_sharding,
    )
    print(f"✓ Model initialized")
    print(f"  Trainable params: {model.num_trainable_params:,}")
except Exception as e:
    print(f"✗ Model initialization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 1: Forward pass on single batch
print("\n[Test 1] Forward pass (single device, no NCCL)...")
try:
    test_seq = jnp.ones((1, 131072, 4), dtype=jnp.float32)
    test_org = jnp.array([1], dtype=jnp.int32)
    
    predictions, _ = model.apply_fn(model.params, model.state, None, test_seq, test_org)
    atac_1bp = predictions.atac.scaled_predictions_1bp
    print(f"✓ Forward pass works: ATAC shape {atac_1bp.shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Forward pass with sharded batch
print("\n[Test 2] Forward pass with sharded batch...")
try:
    test_seq = jnp.ones((8, 131072, 4), dtype=jnp.float32)
    test_org = jnp.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.int32)
    
    # Shard inputs
    test_seq = jax.device_put(test_seq, batch_sharding)
    test_org = jax.device_put(test_org, organism_sharding)
    
    predictions, _ = model.apply_fn(model.params, model.state, None, test_seq, test_org)
    atac_1bp = predictions.atac.scaled_predictions_1bp
    print(f"✓ Sharded forward pass works: ATAC shape {atac_1bp.shape}")
except Exception as e:
    print(f"✗ Sharded forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Training step WITHOUT actual data (dummy targets)
print("\n[Test 3] Training step with dummy data...")
try:
    test_seq = jnp.ones((8, 131072, 4), dtype=jnp.float32)
    test_org = jnp.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.int32)
    test_target = jnp.ones((8, 131072), dtype=jnp.float32) * 100.0  # Dummy signal
    
    # Shard inputs
    test_seq = jax.device_put(test_seq, batch_sharding)
    test_org = jax.device_put(test_org, organism_sharding)
    test_target = jax.device_put(test_target, signal_sharding)
    
    loss, predictions, grad_norm = model.train_step(
        test_seq, test_org, test_target, cut_and_tag_loss_baskerville
    )
    
    print(f"✓ Training step works!")
    print(f"  Loss: {loss}")
    print(f"  Grad norm: {grad_norm}")
except Exception as e:
    print(f"✗ Training step failed: {e}")
    print(f"\n  → Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    
    if "corrupted comm object" in str(e):
        print("\n  → FOUND IT! The wrapper's train_step triggers NCCL corruption")
        print("  → Check models/model.py HeadReplaceAlphaGenomeModel._train_step_jit")
    exit(1)

# Test 4: Multiple training steps (like actual training loop)
print("\n[Test 4] Multiple training steps...")
try:
    for i in range(3):
        test_seq = jnp.ones((8, 131072, 4), dtype=jnp.float32) * (i + 1)
        test_org = jnp.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.int32)
        test_target = jnp.ones((8, 131072), dtype=jnp.float32) * (100.0 + i * 10)
        
        test_seq = jax.device_put(test_seq, batch_sharding)
        test_org = jax.device_put(test_org, organism_sharding)
        test_target = jax.device_put(test_target, signal_sharding)
        
        loss, predictions, grad_norm = model.train_step(
            test_seq, test_org, test_target, cut_and_tag_loss_baskerville
        )
        print(f"  Step {i+1}: loss={loss:.4f}, grad_norm={grad_norm:.4f}")
    
    print("✓ Multiple training steps work!")
except Exception as e:
    print(f"✗ Multiple training steps failed at step {i+1}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("✓ ALL WRAPPER TESTS PASSED")
print("=" * 70)
print("""
Model wrapper works! The issue must be in:
  1. The actual TensorFlow dataset (data loading/collation)
  2. Specific data values causing numerical issues
  3. Something in the full training script setup

Next: Load actual training data in isolation and test.
""")
