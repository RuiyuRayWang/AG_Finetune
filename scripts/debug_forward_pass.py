"""
Debug script to test the forward pass and loss computation.

This simulates what happens during actual training to find where NaN appears.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from torch.utils.data import DataLoader
from alphagenome_research.model import dna_model, model as alphagenome_model
from dataset.dataset import CutAndTagDataset
from utils.borzoi_loss_v2 import cut_and_tag_loss_baskerville, baskerville_target_transform
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


def test_forward_and_loss():
    """Test forward pass and loss computation on actual batches."""
    
    print("="*60)
    print("FORWARD PASS & LOSS DEBUG")
    print("="*60)
    
    # Load base model
    print("\nLoading AlphaGenome model...")
    base_model = dna_model.create(
        checkpoint_path='/home/ray/AG_Finetune/checkpoints/pre_trained/alphagenome-all-folds',
        organism_settings=None,
        device=None
    )
    print(f"✓ Model loaded on {base_model._device_context._device}")
    
    # Create a forward function for raw batch prediction
    # (mirroring what lora_model.py does)
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
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = CutAndTagDataset(
        fasta_path='/home/ray/AG_Finetune/data/resources/mm10.fa',
        signal_bedgraph='/home/ray/AG_Finetune/data/CUTTag/processed_bedgraph/D1MSN_H3K27ac_averaged_normalized.bedgraph',
        genome_windows='/home/ray/AG_Finetune/exports/peak_safe_windows/D1MSN_H3K27ac_merged_peak_safe_windows.bed',
        window_size=131072,
        organism_id=1,
        augment=True,  # Enable augmentation
        max_shift=1024,
    )
    
    # Create data loader  
    # Note: Batch size 1 works perfectly, but batch size 8 with 8 GPUs causes NaN
    # This suggests the issue is in gradient aggregation or batch-level numerics
    batch_size = 1
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_to_jax,
        drop_last=True,
    )
    
    print(f"✓ Dataset created: {len(dataset)} samples")
    print(f"✓ DataLoader created: {len(data_loader)} batches")
    
    # Disable multi-GPU for debugging - we just want to find the NaN source
    # Using multi-GPU would require replicating model parameters across devices
    print(f"\n⚠ Disabling multi-GPU for debugging (using single GPU)")
    mesh = None
    batch_sharding = None
    organism_sharding = None
    signal_sharding = None
    
    # Test batches, especially around batch 13 where NaN appeared
    print(f"\n{'='*60}")
    print("Testing batches 1-30 (where NaN typically appears)")
    print(f"{'='*60}")
    
    for batch_idx, (sequences, organism_ids, targets) in enumerate(data_loader):
        if batch_idx >= 30:
            break
        
        print(f"\n{'='*20} Batch {batch_idx + 1} {'='*20}")
        
        # Skip low signal batches (like in training)
        target_sum = float(jnp.sum(targets['signal']))
        if target_sum < 1e-6:
            print("  Skipped (low signal)")
            continue
        
        # Apply device sharding if multi-GPU
        if batch_sharding is not None:
            sequences = jax.device_put(sequences, batch_sharding)
            organism_ids = jax.device_put(organism_ids, organism_sharding)
            targets = {
                'signal': jax.device_put(targets['signal'], signal_sharding)
            }
        
        # Check inputs
        print("  1. Input Check:")
        check_for_nan(sequences, "sequences")
        check_for_nan(targets['signal'], "targets")
        
        # Forward pass
        print("  2. Forward Pass:")
        try:
            # Use the apply function we created
            predictions = apply_fn(
                base_model._params,
                base_model._state,
                sequences,
                organism_ids
            )
            print("    ✓ Forward pass completed")
            
            # Check predictions
            if isinstance(predictions, dict) and 'atac' in predictions:
                atac_preds = predictions['atac'].get('scaled_predictions_1bp', None)
                if atac_preds is not None:
                    check_for_nan(atac_preds, "predictions.atac.scaled_predictions_1bp")
        except Exception as e:
            print(f"    ❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Test target transformation
        print("  3. Target Transformation:")
        try:
            targets_transformed = baskerville_target_transform(
                targets['signal'],
                scale=1.0,
                clip_soft=64.0,
                clip_hard=384.0,
                use_sqrt=True,
            )
            print("    ✓ Target transform completed")
            check_for_nan(targets_transformed, "targets_transformed")
        except Exception as e:
            print(f"    ❌ Target transform failed: {e}")
            continue
        
        # Test loss computation
        print("  4. Loss Computation:")
        try:
            loss = cut_and_tag_loss_baskerville(
                predictions,
                targets['signal'],
                scale=1.0,
                clip_soft=64.0,
                total_weight=1.0,
                epsilon=1e-6,
            )
            print(f"    ✓ Loss computed: {float(loss):.6f}")
            
            if jnp.isnan(loss) or jnp.isinf(loss):
                print(f"    ❌ Loss is NaN/Inf!")
                
                # Debug: check intermediate values
                print("\n    Debug Info:")
                print(f"      Target stats: min={float(jnp.min(targets['signal'])):.6f}, "
                      f"max={float(jnp.max(targets['signal'])):.6f}, "
                      f"mean={float(jnp.mean(targets['signal'])):.6f}")
                
                if atac_preds is not None:
                    print(f"      Pred stats: min={float(jnp.min(atac_preds)):.6f}, "
                          f"max={float(jnp.max(atac_preds)):.6f}, "
                          f"mean={float(jnp.mean(atac_preds)):.6f}")
                
        except Exception as e:
            print(f"    ❌ Loss computation failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Test gradient computation (this is where NaN might occur!)
        print("  5. Gradient Computation:")
        try:
            def loss_fn(params_to_test):
                # Simplified: just test if gradients flow through loss computation
                preds_copy = predictions
                return cut_and_tag_loss_baskerville(
                    preds_copy,
                    targets['signal'],
                    scale=1.0,
                    clip_soft=64.0,
                    total_weight=1.0,
                    epsilon=1e-6,
                )
            
            # Compute gradient w.r.t. predictions (proxy for parameter gradients)
            grad_fn = jax.grad(lambda pred_val: cut_and_tag_loss_baskerville(
                {'atac': {'scaled_predictions_1bp': pred_val}},
                targets['signal'],
                scale=1.0,
                clip_soft=64.0,
                total_weight=1.0,
                epsilon=1e-6,
            ))
            
            if atac_preds is not None:
                grads = grad_fn(atac_preds)
                has_nan = bool(jnp.any(jnp.isnan(grads)))
                has_inf = bool(jnp.any(jnp.isinf(grads)))
                
                if has_nan or has_inf:
                    print(f"    ❌ Gradients contain NaN={has_nan}, Inf={has_inf}")
                    print(f"       Grad stats: min={float(jnp.min(grads)):.6f}, "
                          f"max={float(jnp.max(grads)):.6f}, "
                          f"mean={float(jnp.mean(grads)):.6f}")
                    print(f"       NaN count: {jnp.sum(jnp.isnan(grads))}")
                    print(f"       Inf count: {jnp.sum(jnp.isinf(grads))}")
                else:
                    grad_norm = float(jnp.sqrt(jnp.sum(grads**2)))
                    print(f"    ✓ Gradients computed: norm={grad_norm:.6f}")
                    
        except Exception as e:
            print(f"    ❌ Gradient computation failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    test_forward_and_loss()
