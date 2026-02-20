"""
Debug script to test multi-GPU gradient computation.

This reproduces the EXACT training setup to find where NaN appears:
- Batch size 8 across 8 GPUs (1 sample per GPU)
- Device sharding with NamedSharding
- Gradient computation with JAX automatic parallelization
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from torch.utils.data import DataLoader
from alphagenome_research.model import dna_model, model as alphagenome_model
from dataset.dataset import CutAndTagDataset
from utils.borzoi_loss_v2 import cut_and_tag_loss_baskerville
from typing import Dict, List, Tuple
import haiku as hk
import jmp


def collate_to_jax(batch: List[Tuple[np.ndarray, int, Dict]]) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    """Custom collate function."""
    sequences = [item[0] for item in batch]
    organism_ids = [item[1] for item in batch]
    signals = [item[2]['signal'] for item in batch]
    
    return (
        jnp.array(sequences, dtype=jnp.float32),
        jnp.array(organism_ids, dtype=jnp.int32),
        {'signal': jnp.array(signals, dtype=jnp.float32)}
    )


def check_for_nan(data, name):
    """Check if data contains NaN/Inf."""
    if isinstance(data, dict):
        for key, val in data.items():
            check_for_nan(val, f"{name}.{key}")
    elif isinstance(data, (np.ndarray, jnp.ndarray)):
        has_nan = bool(jnp.any(jnp.isnan(data)))
        has_inf = bool(jnp.any(jnp.isinf(data)))
        if has_nan or has_inf:
            print(f"    ❌ {name}: NaN={has_nan}, Inf={has_inf}")
            print(f"       Shape: {data.shape}, dtype: {data.dtype}")
            if has_nan:
                print(f"       NaN count: {jnp.sum(jnp.isnan(data))}")
            if has_inf:
                print(f"       Inf count: {jnp.sum(jnp.isinf(data))}")
            return True
        else:
            print(f"    ✓ {name}: Clean")
            return False
    return False


def test_multigpu_gradients():
    """Test gradient computation with multi-GPU sharding."""
    
    print("="*60)
    print("MULTI-GPU GRADIENT DEBUG")
    print("="*60)
    
    num_devices = jax.device_count()
    print(f"\nDevices: {num_devices}")
    print(f"Devices list: {jax.devices()}")
    
    if num_devices < 2:
        print("⚠ Only 1 GPU detected. Multi-GPU test requires 2+ GPUs.")
        return
    
    # Setup device mesh (same as training)
    print(f"\nSetting up multi-GPU mesh with {num_devices} devices...")
    mesh = Mesh(jax.devices(), ('batch',))
    batch_sharding = NamedSharding(mesh, P('batch', None, None))
    organism_sharding = NamedSharding(mesh, P('batch'))
    signal_sharding = NamedSharding(mesh, P('batch', None))
    # Replicate parameters across all devices (empty P() means replicate)
    param_sharding = NamedSharding(mesh, P())
    print(f"✓ Mesh created: {mesh}")
    
    # Load base model
    print("\nLoading AlphaGenome model...")
    base_model = dna_model.create(
        checkpoint_path='/home/ray/AG_Finetune/checkpoints/pre_trained/alphagenome-all-folds',
        organism_settings=None,
        device=None
    )
    print(f"✓ Model loaded on {base_model._device_context._device}")
    
    # Replicate parameters across all devices (CRITICAL for multi-GPU)
    print("Replicating model parameters across all devices...")
    base_params_replicated = jax.device_put(base_model._params, param_sharding)
    print(f"✓ Parameters replicated across {num_devices} devices")
    
    # Create forward function
    print("Setting up forward function...")
    jmp_policy = jmp.get_policy('params=float32,compute=bfloat16,output=bfloat16')
    
    @hk.transform_with_state
    def _forward(dna_sequence, organism_index):
        with hk.mixed_precision.push_policy(alphagenome_model.AlphaGenome, jmp_policy):
            return alphagenome_model.AlphaGenome(
                base_model._metadata,
                num_splice_sites=512,
                splice_site_threshold=0.1,
            )(dna_sequence, organism_index)
    
    def apply_fn(params, state, dna_sequence, organism_index):
        (predictions, _), _ = _forward.apply(
            params, state, None, dna_sequence, organism_index
        )
        return predictions
    
    print("✓ Forward function ready")
    
    # Create dataset with batch size 8 (matching training)
    print("\nCreating dataset...")
    dataset = CutAndTagDataset(
        fasta_path='/home/ray/AG_Finetune/data/resources/mm10.fa',
        signal_bedgraph='/home/ray/AG_Finetune/data/CUTTag/processed_bedgraph/D1MSN_H3K27ac_averaged_normalized.bedgraph',
        genome_windows='/home/ray/AG_Finetune/exports/peak_safe_windows/D1MSN_H3K27ac_merged_peak_safe_windows.bed',
        window_size=131072,
        organism_id=1,
        augment=True,
        max_shift=1024,
    )
    
    batch_size = 8
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_to_jax,
        drop_last=True,
    )
    
    print(f"✓ Dataset created: {len(dataset)} samples")
    print(f"✓ DataLoader created with batch size {batch_size}")
    
    # Define gradient computation function
    def compute_loss_and_grads(params, sequences, organism_ids, targets):
        """Compute loss and gradients (simulating training)."""
        def loss_fn(p):
            predictions = apply_fn(p, base_model._state, sequences, organism_ids)
            loss = cut_and_tag_loss_baskerville(
                predictions,
                targets['signal'],
                scale=1.0,
                clip_soft=64.0,
                total_weight=1.0,
                epsilon=1e-6,
            )
            return loss, predictions
        
        (loss, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return loss, predictions, grads
    
    # JIT compile for performance
    compute_loss_and_grads_jit = jax.jit(compute_loss_and_grads)
    
    # Test batches
    print(f"\n{'='*60}")
    print("Testing batches 1-20 with MULTI-GPU SHARDING")
    print(f"{'='*60}")
    
    for batch_idx, (sequences, organism_ids, targets) in enumerate(data_loader):
        if batch_idx >= 20:
            break
        
        print(f"\n{'='*20} Batch {batch_idx + 1} {'='*20}")
        
        # Skip low signal batches
        target_sum = float(jnp.sum(targets['signal']))
        if target_sum < 1e-6:
            print("  Skipped (low signal)")
            continue
        
        # Apply device sharding (CRITICAL: same as training)
        sequences = jax.device_put(sequences, batch_sharding)
        organism_ids = jax.device_put(organism_ids, organism_sharding)
        targets = {
            'signal': jax.device_put(targets['signal'], signal_sharding)
        }
        
        print(f"  Sharding applied:")
        print(f"    Sequences: {sequences.sharding}")
        print(f"    Shape: {sequences.shape}")
        
        # Compute loss and gradients
        print(f"  Computing loss and gradients...")
        try:
            loss, predictions, grads = compute_loss_and_grads_jit(
                base_params_replicated,  # Use replicated params, not base_model._params
                sequences,
                organism_ids,
                targets
            )
            
            # Check loss
            if jnp.isnan(loss) or jnp.isinf(loss):
                print(f"    ❌ Loss is NaN/Inf: {float(loss)}")
            else:
                print(f"    ✓ Loss: {float(loss):.6f}")
            
            # Check gradients
            has_nan_grad = False
            has_inf_grad = False
            
            def check_grad(g):
                nonlocal has_nan_grad, has_inf_grad
                if jnp.any(jnp.isnan(g)):
                    has_nan_grad = True
                if jnp.any(jnp.isinf(g)):
                    has_inf_grad = True
                return g
            
            jax.tree.map(check_grad, grads)
            
            if has_nan_grad or has_inf_grad:
                print(f"    ❌ Gradients contain NaN={has_nan_grad}, Inf={has_inf_grad}")
                
                # Find which gradient has NaN
                def print_nan_location(path, g):
                    if jnp.any(jnp.isnan(g)) or jnp.any(jnp.isinf(g)):
                        print(f"       {path}: shape={g.shape}, NaN={jnp.any(jnp.isnan(g))}, Inf={jnp.any(jnp.isinf(g))}")
                
                jax.tree_util.tree_map_with_path(
                    lambda path, g: print_nan_location(path, g),
                    grads
                )
            else:
                # Compute gradient norm
                grad_norm = jnp.sqrt(sum(
                    jnp.sum(g**2) for g in jax.tree.leaves(grads)
                ))
                print(f"    ✓ Gradients clean, norm={float(grad_norm):.6e}")
                
        except Exception as e:
            print(f"    ❌ Gradient computation failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    test_multigpu_gradients()
