"""
Trace when LoRA parameters become unstable during sequential training.

This will:
1. Train sequentially (fixed seed for reproducibility)
2. Monitor parameter statistics after each batch
3. Identify the exact batch where instability begins
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
from utils.lora_utils import get_attention_qkv_layers
from utils.borzoi_loss_v2 import cut_and_tag_loss_baskerville
from typing import Dict, List, Tuple
import torch


def collate_to_jax(batch: List[Tuple[np.ndarray, int, Dict]]) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    sequences = [item[0] for item in batch]
    organism_ids = [item[1] for item in batch]
    signals = [item[2]['signal'] for item in batch]
    return (
        jnp.array(sequences, dtype=jnp.float32),
        jnp.array(organism_ids, dtype=jnp.int32),
        {'signal': jnp.array(signals, dtype=jnp.float32)}
    )


def analyze_parameters(lora_params, name="LoRA"):
    """Analyze parameter statistics."""
    print(f"\n{name} Parameters:")
    for key in ['lora_A', 'lora_B']:
        if key in lora_params:
            params = lora_params[key]
            # Flatten all layers
            all_values = []
            for layer_key, layer_params in params.items():
                flat = layer_params.reshape(-1)
                all_values.append(flat)
            
            if all_values:
                all_values = jnp.concatenate(all_values)
                
                has_nan = jnp.any(jnp.isnan(all_values))
                has_inf = jnp.any(jnp.isinf(all_values))
                
                print(f"  {key}:")
                print(f"    Has NaN: {has_nan}, Has Inf: {has_inf}")
                
                if has_nan or has_inf:
                    print(f"    ⚠️⚠️⚠️ PARAMETERS CORRUPTED ⚠️⚠️⚠️")
                    nan_count = int(jnp.sum(jnp.isnan(all_values)))
                    inf_count = int(jnp.sum(jnp.isinf(all_values)))
                    print(f"    NaN count: {nan_count}/{len(all_values)}")
                    print(f"    Inf count: {inf_count}/{len(all_values)}")
                else:
                    min_val = float(jnp.min(all_values))
                    max_val = float(jnp.max(all_values))
                    mean_val = float(jnp.mean(all_values))
                    std_val = float(jnp.std(all_values))
                    
                    print(f"    Min: {min_val:.6e}, Max: {max_val:.6e}")
                    print(f"    Mean: {mean_val:.6e}, Std: {std_val:.6e}")
                    
                    # Check for extreme values
                    extreme_count = int(jnp.sum(jnp.abs(all_values) > 10.0))
                    if extreme_count > 0:
                        print(f"    ⚠️  {extreme_count} values with abs > 10.0")


def main():
    print("="*80)
    print("TRACING PARAMETER DRIFT DURING TRAINING")
    print("="*80)
    
    # Setup
    batch_size = 8
    num_devices = jax.device_count()
    
    # IMPORTANT: Fix random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print(f"\nSetup: {num_devices} GPUs, batch size {batch_size}")
    print("Fixed seeds: numpy=42, torch=42")
    
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
        shuffle=True,
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
    
    # Initial parameter check
    print("\n" + "="*80)
    print("INITIAL PARAMETERS (Before Training)")
    print("="*80)
    analyze_parameters(lora_model.lora_params, "Initial")
    
    # Train for first 30 batches and monitor
    print("\n" + "="*80)
    print("TRAINING AND MONITORING (First 30 batches)")
    print("="*80)
    
    first_nan_batch = None
    
    for batch_idx, (sequences, organism_ids, targets) in enumerate(data_loader):
        if batch_idx >= 30:
            break
        
        # Shard if multi-GPU
        if batch_sharding is not None:
            sequences = jax.device_put(sequences, batch_sharding)
            organism_ids = jax.device_put(organism_ids, organism_sharding)
            targets = {'signal': jax.device_put(targets['signal'], signal_sharding)}
        
        # Training step
        loss_value, predictions, grad_norm = lora_model.train_step(
            sequences, organism_ids, targets, loss_fn
        )
        
        loss_val = float(loss_value)
        grad_val = float(grad_norm)
        
        # Print batch info
        status = "✓" if grad_val > 1e-10 else "⚠️ NaN/Zero"
        print(f"\nBatch {batch_idx+1}: Loss={loss_val:.6f}, Grad={grad_val:.6e} {status}")
        
        # Check parameters after update
        analyze_parameters(lora_model.lora_params, f"  After batch {batch_idx+1}")
        
        # Detect first NaN
        if grad_val < 1e-10 and first_nan_batch is None:
            first_nan_batch = batch_idx + 1
            print(f"\n{'='*80}")
            print(f"⚠️  FIRST NaN GRADIENT DETECTED AT BATCH {first_nan_batch}")
            print(f"{'='*80}")
            
            # Detailed analysis
            print("\nThis batch caused the instability!")
            print(f"  Previous batch {batch_idx}: Had valid gradients")
            print(f"  Current batch {batch_idx+1}: NaN gradients appeared")
            print("\nParameter analysis above shows if parameters became corrupt.")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if first_nan_batch:
        print(f"\n⚠️  NaN gradients first appeared at batch {first_nan_batch}")
        print(f"\nThis suggests:")
        print(f"  1. Batch {first_nan_batch-1} caused a large parameter update")
        print(f"  2. The update destabilized the model")
        print(f"  3. Subsequent batches produce NaN gradients")
        print(f"\nRecommended fixes:")
        print(f"  1. Reduce learning rate (1e-5 → 5e-6 or 1e-6)")
        print(f"  2. Re-enable gradient clipping with smaller threshold (10.0 → 1.0)")
        print(f"  3. Increase warmup steps (1300 → 2600)")
    else:
        print("\n✅ No NaN gradients in first 30 batches")
        print("Parameters remain stable throughout training")


if __name__ == '__main__':
    main()
