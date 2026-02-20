"""
Comprehensive investigation of NaN gradient batches.

This script will:
1. Identify batches that produce NaN gradients
2. Compare them with clean batches
3. Analyze predictions, loss, and gradient values at each step
4. Identify the root cause of NaN
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from torch.utils.data import DataLoader
from alphagenome_research.model import dna_model
from models.lora_model import LoRAAlphaGenomeModel
from dataset.dataset import CutAndTagDataset
from utils.lora_utils import get_attention_qkv_layers, apply_lora_to_params
from utils.borzoi_loss_v2 import cut_and_tag_loss_baskerville
from typing import Dict, List, Tuple
import pickle


def collate_to_jax(batch: List[Tuple[np.ndarray, int, Dict]]) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    sequences = [item[0] for item in batch]
    organism_ids = [item[1] for item in batch]
    signals = [item[2]['signal'] for item in batch]
    return (
        jnp.array(sequences, dtype=jnp.float32),
        jnp.array(organism_ids, dtype=jnp.int32),
        {'signal': jnp.array(signals, dtype=jnp.float32)}
    )


def analyze_predictions(predictions, name="predictions"):
    """Analyze prediction statistics."""
    if isinstance(predictions, dict):
        print(f"\n{name} (dict):")
        for key, val in predictions.items():
            analyze_predictions(val, f"{name}[{key}]")
    else:
        has_nan = jnp.any(jnp.isnan(predictions))
        has_inf = jnp.any(jnp.isinf(predictions))
        print(f"  {name}:")
        print(f"    Shape: {predictions.shape}, dtype: {predictions.dtype}")
        print(f"    Has NaN: {has_nan}, Has Inf: {has_inf}")
        print(f"    Min: {float(jnp.min(predictions)):.6e}")
        print(f"    Max: {float(jnp.max(predictions)):.6e}")
        print(f"    Mean: {float(jnp.mean(predictions)):.6e}")
        print(f"    Std: {float(jnp.std(predictions)):.6e}")
        
        # Check for extreme values
        extreme_low = jnp.sum(predictions < -100)
        extreme_high = jnp.sum(predictions > 100)
        if extreme_low > 0 or extreme_high > 0:
            print(f"    ⚠️  Extreme values: {extreme_low} < -100, {extreme_high} > 100")


def test_forward_pass(lora_model, sequences, organism_ids, targets):
    """Test forward pass and check for NaN in predictions."""
    print("\n" + "="*80)
    print("TESTING FORWARD PASS")
    print("="*80)
    
    # Merge LoRA parameters
    full_lora_params = {
        'lora_A': lora_model.lora_params['lora_A'],
        'lora_B': lora_model.lora_params['lora_B'],
        'scaling': lora_model.lora_params['scaling'],
        'rank': lora_model.lora_params['rank'],
        'alpha': lora_model.lora_params['alpha'],
        'target_layers': lora_model.lora_params['target_layers']
    }
    merged_params = apply_lora_to_params(lora_model.base_params, full_lora_params)
    
    # Forward pass
    predictions = lora_model.apply_fn(
        merged_params,
        lora_model.base_state,
        sequences,
        organism_ids
    )
    
    analyze_predictions(predictions, "Raw predictions")
    
    return predictions


def test_loss_computation(predictions, targets, loss_fn):
    """Test loss computation step by step."""
    print("\n" + "="*80)
    print("TESTING LOSS COMPUTATION")
    print("="*80)
    
    print("\nInput to loss function:")
    print(f"  Targets shape: {targets['signal'].shape}")
    print(f"  Targets dtype: {targets['signal'].dtype}")
    print(f"  Targets has NaN: {jnp.any(jnp.isnan(targets['signal']))}")
    print(f"  Targets min: {float(jnp.min(targets['signal'])):.6e}")
    print(f"  Targets max: {float(jnp.max(targets['signal'])):.6e}")
    print(f"  Targets mean: {float(jnp.mean(targets['signal'])):.6e}")
    
    # Compute loss
    loss = loss_fn(predictions, targets['signal'])
    
    print(f"\nLoss value:")
    print(f"  Loss: {float(loss):.6f}")
    print(f"  Loss is NaN: {jnp.isnan(loss)}")
    print(f"  Loss is Inf: {jnp.isinf(loss)}")
    
    return loss


def test_gradient_computation(lora_model, sequences, organism_ids, targets, loss_fn):
    """Test gradient computation and identify where NaN appears."""
    print("\n" + "="*80)
    print("TESTING GRADIENT COMPUTATION")
    print("="*80)
    
    # Define loss function for gradient computation
    def compute_loss_for_grad(lora_trainable):
        full_lora_params = {
            'lora_A': lora_trainable['lora_A'],
            'lora_B': lora_trainable['lora_B'],
            'scaling': lora_model.lora_params['scaling'],
            'rank': lora_model.lora_params['rank'],
            'alpha': lora_model.lora_params['alpha'],
            'target_layers': lora_model.lora_params['target_layers']
        }
        merged_params = apply_lora_to_params(lora_model.base_params, full_lora_params)
        predictions = lora_model.apply_fn(
            merged_params,
            lora_model.base_state,
            sequences,
            organism_ids
        )
        loss = loss_fn(predictions, targets['signal'])
        return loss
    
    lora_trainable = {
        'lora_A': lora_model.lora_params['lora_A'],
        'lora_B': lora_model.lora_params['lora_B']
    }
    
    # Compute gradients
    loss_value, grads = jax.value_and_grad(compute_loss_for_grad)(lora_trainable)
    
    print(f"\nLoss from gradient computation: {float(loss_value):.6f}")
    
    # Analyze gradients
    print("\nGradient statistics:")
    for key, grad in grads.items():
        print(f"\n  {key}:")
        print(f"    Shape: {grad.shape}")
        
        # Flatten for analysis
        grad_flat = grad.reshape(-1)
        has_nan = jnp.any(jnp.isnan(grad_flat))
        has_inf = jnp.any(jnp.isinf(grad_flat))
        
        print(f"    Has NaN: {has_nan}")
        print(f"    Has Inf: {has_inf}")
        print(f"    NaN count: {int(jnp.sum(jnp.isnan(grad_flat)))}")
        print(f"    Inf count: {int(jnp.sum(jnp.isinf(grad_flat)))}")
        
        # Stats on finite values
        finite_mask = jnp.isfinite(grad_flat)
        finite_grads = grad_flat[finite_mask]
        
        if jnp.sum(finite_mask) > 0:
            print(f"    Finite values: {int(jnp.sum(finite_mask))}/{len(grad_flat)}")
            print(f"    Min (finite): {float(jnp.min(finite_grads)):.6e}")
            print(f"    Max (finite): {float(jnp.max(finite_grads)):.6e}")
            print(f"    Mean (finite): {float(jnp.mean(finite_grads)):.6e}")
            print(f"    Std (finite): {float(jnp.std(finite_grads)):.6e}")
        else:
            print(f"    ⚠️  ALL GRADIENTS ARE NaN/Inf!")
    
    return grads


def compare_batches(lora_model, clean_batch, nan_batch, loss_fn):
    """Compare a clean batch vs a NaN-producing batch."""
    print("\n" + "="*80)
    print("COMPARING CLEAN VS NaN BATCH")
    print("="*80)
    
    for batch_name, batch_data in [("CLEAN", clean_batch), ("NaN", nan_batch)]:
        sequences, organism_ids, targets = batch_data
        
        print(f"\n{'='*80}")
        print(f"BATCH: {batch_name}")
        print(f"{'='*80}")
        
        # Test forward pass
        predictions = test_forward_pass(lora_model, sequences, organism_ids, targets)
        
        # Test loss
        loss = test_loss_computation(predictions, targets, loss_fn)
        
        # Test gradients
        grads = test_gradient_computation(lora_model, sequences, organism_ids, targets, loss_fn)


def main():
    print("="*80)
    print("NaN GRADIENT INVESTIGATION")
    print("="*80)
    
    # Setup
    batch_size = 8
    num_devices = jax.device_count()
    
    print(f"\nSetup: {num_devices} GPUs, batch size {batch_size}")
    
    # Load model
    print("\nLoading AlphaGenome base model...")
    base_model = dna_model.create(
        checkpoint_path='/home/ray/AG_Finetune/checkpoints/pre_trained/alphagenome-all-folds',
        organism_settings=None,
        device=None
    )
    
    # Setup dataset
    print("\nSetting up dataset...")
    dataset = CutAndTagDataset(
        fasta_path='/home/ray/AG_Finetune/data/resources/mm10.fa',
        signal_bedgraph='/home/ray/AG_Finetune/data/CUTTag/processed_bedgraph/D1MSN_H3K27ac_averaged_normalized.bedgraph',
        genome_windows=None,
        window_size=131072,
        organism_id=1,
        augment=True,
        max_shift=1024,
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Match training setup - use random batches
        num_workers=0,
        collate_fn=collate_to_jax,
        drop_last=True,
    )
    
    # Initialize LoRA model
    print("\nInitializing LoRA model...")
    target_layers = get_attention_qkv_layers()
    
    if num_devices > 1:
        mesh = Mesh(jax.devices(), ('batch',))
        batch_sharding = NamedSharding(mesh, P('batch', None, None))
        organism_sharding = NamedSharding(mesh, P('batch'))
        signal_sharding = NamedSharding(mesh, P('batch', None))
    else:
        mesh = None
        batch_sharding = None
        organism_sharding = None
        signal_sharding = None
    
    lora_model = LoRAAlphaGenomeModel(
        base_model=base_model,
        lora_rank=8,
        lora_alpha=16.0,
        target_layers=target_layers,
        learning_rate=1e-5,
        mesh=mesh,
        warmup_steps=1300,
        total_steps=26010,
    )
    
    loss_fn = cut_and_tag_loss_baskerville
    
    # Scan through batches to find clean and NaN-producing batches
    print("\n" + "="*80)
    print("SCANNING BATCHES FOR NaN GRADIENTS")
    print("="*80)
    
    clean_batch = None
    nan_batch = None
    clean_batch_idx = None
    nan_batch_idx = None
    
    print("\nTesting up to 100 batches to find examples...")
    nan_count = 0
    clean_count = 0
    
    for batch_idx, (sequences, organism_ids, targets) in enumerate(data_loader):
        if batch_idx >= 100:
            break
        
        # Shard if multi-GPU
        if batch_sharding is not None:
            sequences = jax.device_put(sequences, batch_sharding)
            organism_ids = jax.device_put(organism_ids, organism_sharding)
            targets = {'signal': jax.device_put(targets['signal'], signal_sharding)}
        
        # Quick test: compute gradients
        loss_value, predictions, grad_norm = lora_model.train_step(
            sequences, organism_ids, targets, loss_fn
        )
        
        grad_val = float(grad_norm)
        
        # Collect examples
        if grad_val > 1e-10:
            clean_count += 1
            if clean_batch is None:
                clean_batch = (sequences, organism_ids, targets)
                clean_batch_idx = batch_idx + 1
                print(f"  Batch {batch_idx+1}: Clean (grad_norm={grad_val:.6e}) ✓")
        else:
            nan_count += 1
            print(f"  Batch {batch_idx+1}: NaN/Zero grad (grad_norm={grad_val:.6e}, loss={float(loss_value):.6f}) ⚠️")
            if nan_batch is None:
                nan_batch = (sequences, organism_ids, targets)
                nan_batch_idx = batch_idx + 1
        
        if clean_batch is not None and nan_batch is not None:
            # Found both, but continue to count total
            pass
    
    print(f"\nSummary: {clean_count} clean, {nan_count} NaN batches out of {batch_idx+1} tested")
    
    if clean_batch is None:
        print("\n❌ Could not find a clean batch!")
        return
    
    if nan_batch is None:
        print("\n✅ No NaN batches found!")
        return
    
    print(f"\nUsing Batch {clean_batch_idx} (clean) and Batch {nan_batch_idx} (NaN) for detailed analysis...")
    
    # Deep analysis
    compare_batches(lora_model, clean_batch, nan_batch, loss_fn)
    
    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)
    print("\nKey findings are displayed above. Compare the two batches:")
    print("\nLook for differences in:")
    print("  1. Prediction statistics (Min/Max/Mean)")
    print("  2. Extreme prediction values (abs > 100)")
    print("  3. NaN/Inf in predictions")
    print("  4. Target statistics")
    print("  5. Loss values")
    print("  6. Gradient patterns")
    print("\nThe difference between clean and NaN batches will reveal the root cause.")


if __name__ == '__main__':
    main()
