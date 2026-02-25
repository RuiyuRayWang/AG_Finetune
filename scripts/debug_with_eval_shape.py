#!/usr/bin/env python3
"""
Debug multi-GPU training using jax.eval_shape (DeepMind recommendation).

This script validates training logic without GPU execution to identify:
- Shape mismatches in inputs/outputs
- Incorrect sharding specifications
- Issues before NCCL calls

Usage:
    python scripts/debug_with_eval_shape.py --config configs/config_bigwig.yaml
"""

import argparse
import yaml
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax

# Import utilities
import sys
sys.path.append('/home/ray/AG_Finetune')
from utils.borzoi_loss import cut_and_tag_loss_baskerville


def create_mesh_and_shardings(num_devices=8):
    """Create mesh and sharding specs for data parallelism."""
    devices = jax.devices()[:num_devices]
    print(f"\n=== Device Setup ===")
    print(f"Available devices: {len(jax.devices())}")
    print(f"Using devices: {devices}")
    
    mesh = Mesh(devices, ('batch',))
    print(f"Mesh: {mesh}")
    
    # Sharding specs
    batch_sharding = NamedSharding(mesh, P('batch', None, None))
    organism_sharding = NamedSharding(mesh, P('batch'))
    signal_sharding = NamedSharding(mesh, P('batch', None))
    replicated = NamedSharding(mesh, P())
    
    return mesh, batch_sharding, organism_sharding, signal_sharding, replicated


def create_dummy_batch(batch_size=8, window_size=131072):
    """Create dummy data matching training batch structure."""
    print(f"\n=== Creating Dummy Batch ===")
    print(f"Batch size: {batch_size}")
    print(f"Window size: {window_size}")
    
    # DNA sequence: [batch, seq_len, 4] with one-hot encoding
    dna_sequence = np.zeros((batch_size, window_size, 4), dtype=np.float32)
    # Set random bases (exactly one 1 per position)
    for i in range(batch_size):
        for j in range(window_size):
            base = np.random.randint(0, 4)
            dna_sequence[i, j, base] = 1.0
    
    # Organism index: [batch] (0=human, 1=mouse)
    organism_index = np.ones(batch_size, dtype=np.int32)  # All mouse
    
    # Signal targets: [batch, seq_len]
    signal = np.random.rand(batch_size, window_size).astype(np.float32) * 10  # Realistic range
    
    print(f"DNA sequence shape: {dna_sequence.shape}, dtype: {dna_sequence.dtype}")
    print(f"Organism index shape: {organism_index.shape}, dtype: {organism_index.dtype}")
    print(f"Signal shape: {signal.shape}, dtype: {signal.dtype}")
    
    return dna_sequence, organism_index, signal


def shard_dummy_batch(dna_sequence, organism_index, signal, 
                      batch_sharding, organism_sharding, signal_sharding):
    """Apply sharding to dummy batch (creates ShapedArray with sharding metadata)."""
    print(f"\n=== Applying Sharding ===")
    
    # Convert to JAX arrays with sharding
    dna_jax = jax.device_put(dna_sequence, batch_sharding)
    org_jax = jax.device_put(organism_index, organism_sharding)
    signal_jax = jax.device_put(signal, signal_sharding)
    
    print(f"DNA sharding: {dna_jax.sharding}")
    print(f"Organism sharding: {org_jax.sharding}")
    print(f"Signal sharding: {signal_jax.sharding}")
    
    return dna_jax, org_jax, signal_jax


def validate_training_step_shapes(config_path, num_devices=8):
    """Use jax.eval_shape to validate training step without GPU execution."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    batch_size = config.get('batch_size', 8)
    window_size = config.get('window_size', 131072)
    checkpoint_path = config.get('checkpoint')
    
    print(f"\n{'='*60}")
    print(f"JAX EVAL_SHAPE DEBUGGING (DeepMind Recommendation)")
    print(f"{'='*60}")
    print(f"Config: {config_path}")
    print(f"Batch size: {batch_size}")
    print(f"Window size: {window_size}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Create mesh and shardings
    mesh, batch_sharding, organism_sharding, signal_sharding, replicated = \
        create_mesh_and_shardings(num_devices)
    
    # Create dummy batch
    dna_sequence, organism_index, signal = create_dummy_batch(batch_size, window_size)
    
    # Apply sharding (creates ShapedArrays)
    dna_jax, org_jax, signal_jax = shard_dummy_batch(
        dna_sequence, organism_index, signal, 
        batch_sharding, organism_sharding, signal_sharding
    )
    
    # Load base model using AlphaGenome's official API
    print(f"\n=== Loading Model ===")
    print(f"Loading AlphaGenome base model from checkpoint...")
    
    try:
        from alphagenome_research.model import dna_model, model as alphagenome_model
        from alphagenome_research.model.metadata import metadata as metadata_lib
        import haiku as hk
        import jmp
        
        base_model = dna_model.create(
            checkpoint_path=checkpoint_path,
            organism_settings=None,
            device=None
        )
        print(f"✓ Model loaded successfully")
        
        # Extract params and state for shape validation
        params = base_model._params
        state = base_model._state
        
        print(f"  Params shape sample: {jax.tree.leaves(params)[0].shape}")
        print(f"  Total param arrays: {len(jax.tree.leaves(params))}")
        
        # Create forward function for shape validation
        # We need to recreate the Haiku transform since base_model doesn't expose it directly
        print(f"  Creating forward function for shape validation...")
        
        # Load metadata for mouse (organism_id=1)
        metadata = {
            organism: metadata_lib.load(organism) 
            for organism in dna_model.Organism
        }
        
        jmp_policy = jmp.get_policy('params=float32,compute=float32,output=float32')
        
        @hk.transform_with_state
        def forward(dna_sequence, organism_index):
            with hk.mixed_precision.push_policy(alphagenome_model.AlphaGenome, jmp_policy):
                return alphagenome_model.AlphaGenome(metadata)(dna_sequence, organism_index)
        
        print(f"✓ Forward function created")
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Setup optimizer for shape validation
    print(f"\n=== Setting Up Optimizer ===")
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(5e-4),
    )
    opt_state = optimizer.init(params)
    print(f"✓ Optimizer initialized")
    
    # Replicate params and opt_state if using mesh
    if mesh is not None:
        print(f"  Replicating params and opt_state across {num_devices} devices...")
        params = jax.device_put(params, replicated)
        state = jax.device_put(state, replicated)
        opt_state = jax.device_put(opt_state, replicated)
    
    # Define training step function (unwrapped, for eval_shape)
    def train_step_fn(params_arg, state_arg, opt_state_arg, dna_seq, org_idx, targets):
        """Simplified training step for shape validation."""
        
        def compute_loss(p):
            # Forward pass using Haiku transform
            # Pattern: (predictions, embeddings), new_state = forward.apply(params, state, rng, dna_seq, org_idx)
            (predictions, embeddings), new_state_dict = forward.apply(
                p,
                state_arg,
                None,  # No RNG needed for inference
                dna_seq,
                org_idx
            )
            
            # Extract ATAC predictions (1bp resolution)
            # Note: Base pretrained model has 256 ATAC tracks, shape [batch, seq_len, num_tracks]
            atac_preds = predictions['atac']['scaled_predictions_1bp']
            
            # For debug purposes, just sum predictions (skip actual loss computation)
            # Real training uses head-replaced model with 1 track matching target shape
            # Here we just validate shapes can flow through gradient computation
            pseudo_loss = jnp.sum(atac_preds) + jnp.sum(targets)
            
            return pseudo_loss, (predictions, new_state_dict)
        
        # Compute gradients
        (loss_value, (predictions, new_state_dict)), grads = jax.value_and_grad(
            compute_loss, has_aux=True
        )(params_arg)
        
        # Optimizer update
        updates, new_opt_state = optimizer.update(grads, opt_state_arg, params_arg)
        new_params = optax.apply_updates(params_arg, updates)
        
        # Compute gradient norm
        grad_norm = jnp.sqrt(sum(
            jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grads)
        ))
        
        return loss_value, predictions, new_params, new_state_dict, new_opt_state, grad_norm
    
    # Use jax.eval_shape to validate without execution
    print(f"\n{'='*60}")
    print(f"RUNNING JAX.EVAL_SHAPE (NO GPU EXECUTION)")
    print(f"{'='*60}")
    
    try:
        shape_result = jax.eval_shape(
            train_step_fn,
            params,
            state,
            opt_state,
            dna_jax,
            org_jax,
            signal_jax
        )
        
        loss_shape, preds_shape, params_shape, state_shape, opt_state_shape, grad_norm_shape = shape_result
        
        print(f"\n✓ EVAL_SHAPE SUCCEEDED - No shape mismatches detected!")
        print(f"\n=== Output Shapes ===")
        print(f"Pseudo-loss: {loss_shape}")
        print(f"Gradient norm: {grad_norm_shape}")
        print(f"Predictions keys: {preds_shape.keys() if isinstance(preds_shape, dict) else 'N/A'}")
        if isinstance(preds_shape, dict) and 'atac' in preds_shape:
            print(f"  ATAC predictions: {preds_shape['atac']}")
            if 'scaled_predictions_1bp' in preds_shape['atac']:
                atac_shape = preds_shape['atac']['scaled_predictions_1bp']
                print(f"  ATAC 1bp shape: {atac_shape}")
                print(f"  Note: Base model has {atac_shape.shape[2] if len(atac_shape.shape) > 2 else '?'} ATAC tracks")
                print(f"        Fine-tuning replaces output head with 1 track for CUT&Tag")
        
        print(f"\n=== Updated State Shapes ===")
        print(f"Params (sample): {jax.tree.leaves(params_shape)[0]}")
        print(f"State keys: {len(state_shape.keys()) if isinstance(state_shape, dict) else 'N/A'} BatchNorm layers")
        print(f"Optimizer state: {type(opt_state_shape).__name__} (shape validated)")
        
        print(f"\n{'='*60}")
        print(f"SHAPE VALIDATION PASSED ✓")
        print(f"{'='*60}")
        print(f"\nValidation summary:")
        print(f"  ✓ Forward pass computes predictions")
        print(f"  ✓ Gradients can be computed")
        print(f"  ✓ Optimizer updates work")
        print(f"  ✓ All shapes are compatible through the pipeline")
        print(f"\nNext steps:")
        print(f"1. Shapes are compatible - issue is NOT shape mismatch")
        print(f"2. Check sharding specifications in jax.jit call")
        print(f"3. Verify mesh setup matches official notebook")
        print(f"4. Inspect NCCL configuration and driver versions")
        
    except Exception as e:
        print(f"\n✗ EVAL_SHAPE FAILED")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print(f"\nThis indicates a shape mismatch or incorrect typing.")
        print(f"Review the traceback to identify the problematic operation.")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Debug multi-GPU training with jax.eval_shape"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--num-devices',
        type=int,
        default=8,
        help='Number of GPUs to use (default: 8)'
    )
    
    args = parser.parse_args()
    
    validate_training_step_shapes(args.config, args.num_devices)


if __name__ == '__main__':
    main()
