"""
Debug script to find where NaN appears in the training pipeline.

Tests:
1. Dataset -> no NaN (we confirmed this)
2. DataLoader batching -> NaN?
3. Collate to JAX -> NaN?
4. Device sharding -> NaN?
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from torch.utils.data import DataLoader
from dataset.dataset import CutAndTagDataset
from typing import Dict, List, Tuple


def collate_to_jax(batch: List[Tuple[np.ndarray, int, Dict]]) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    """Custom collate function to convert batch to JAX arrays."""
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
            check_for_nan(val, f"{name}[{key}]")
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            check_for_nan(item, f"{name}[{i}]")
    elif isinstance(data, (np.ndarray, jnp.ndarray)):
        has_nan = np.any(np.isnan(data))
        has_inf = np.any(np.isinf(data))
        if has_nan or has_inf:
            print(f"  ❌ {name}: NaN={has_nan}, Inf={has_inf}")
            print(f"     Shape: {data.shape}, dtype: {data.dtype}")
            print(f"     NaN count: {np.sum(np.isnan(data))}")
            print(f"     Inf count: {np.sum(np.isinf(data))}")
            print(f"     Min: {np.nanmin(data)}, Max: {np.nanmax(data)}, Mean: {np.nanmean(data)}")
            return True
        else:
            print(f"  ✓ {name}: Clean (shape={data.shape}, dtype={data.dtype})")
            return False
    return False


def test_pipeline():
    """Test each stage of the training pipeline."""
    
    print("="*60)
    print("TRAINING PIPELINE DEBUG")
    print("="*60)
    
    # Setup
    batch_size = 8
    num_devices = jax.device_count()
    
    print(f"\nSetup:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num devices: {num_devices}")
    
    # Create dataset WITH augmentation (as in training)
    print(f"\n{'='*60}")
    print("STAGE 1: Create Dataset (NO BED FILTER - matches training)")
    print(f"{'='*60}")
    
    dataset = CutAndTagDataset(
        fasta_path='/home/ray/AG_Finetune/data/resources/mm10.fa',
        signal_bedgraph='/home/ray/AG_Finetune/data/CUTTag/processed_bedgraph/D1MSN_H3K27ac_averaged_normalized.bedgraph',
        genome_windows=None,  # NO BED FILE - matches actual training setup!
        window_size=131072,
        organism_id=1,
        augment=True,  # Enable augmentation like in training
        max_shift=1024,
    )
    print(f"✓ Dataset created with {len(dataset)} samples")
    
    # Test individual samples with augmentation
    print(f"\n{'='*60}")
    print("STAGE 2: Test Individual Samples (with augmentation)")
    print(f"{'='*60}")
    
    print("\nTesting 100 random samples...")
    np.random.seed(42)
    test_indices = np.random.choice(len(dataset), size=min(100, len(dataset)), replace=False)
    
    nan_count = 0
    for idx in test_indices:
        seq, org_id, targets = dataset[idx]
        signal = targets['signal']
        
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            nan_count += 1
            print(f"  Sample {idx}: Contains NaN/Inf!")
            print(f"    Signal: min={np.nanmin(signal)}, max={np.nanmax(signal)}, mean={np.nanmean(signal)}")
    
    if nan_count == 0:
        print(f"  ✓ All {len(test_indices)} samples clean")
    else:
        print(f"  ❌ {nan_count}/{len(test_indices)} samples have NaN/Inf")
    
    # Create DataLoader
    print(f"\n{'='*60}")
    print("STAGE 3: Create DataLoader")
    print(f"{'='*60}")
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=None,  # Use default collation first
        drop_last=True,
    )
    print(f"✓ DataLoader created")
    
    # Test batching (default collation)
    print(f"\n{'='*60}")
    print("STAGE 4: Test Default Batching (PyTorch)")
    print(f"{'='*60}")
    
    print("\nChecking first 20 batches with default collation...")
    for batch_idx, batch in enumerate(data_loader):
        if batch_idx >= 20:
            break
        
        print(f"\nBatch {batch_idx}:")
        sequences, organism_ids, targets = batch
        signal = targets['signal']
        
        # Check numpy arrays
        has_nan = check_for_nan(signal.numpy(), f"  signal (numpy)")
        
        if has_nan:
            print(f"  ❌ Found NaN in batch {batch_idx} AFTER default collation")
            break
    
    # Test JAX collation
    print(f"\n{'='*60}")
    print("STAGE 5: Test JAX Collation")
    print(f"{'='*60}")
    
    data_loader_jax = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_to_jax,
        drop_last=True,
    )
    
    print("\nChecking first 20 batches with JAX collation...")
    for batch_idx, (sequences, organism_ids, targets) in enumerate(data_loader_jax):
        if batch_idx >= 20:
            break
        
        print(f"\nBatch {batch_idx}:")
        check_for_nan(sequences, "  sequences")
        check_for_nan(targets['signal'], "  signal")
    
    # Test device sharding
    if num_devices > 1:
        print(f"\n{'='*60}")
        print("STAGE 6: Test Device Sharding")
        print(f"{'='*60}")
        
        mesh = Mesh(jax.devices(), ('batch',))
        batch_sharding = NamedSharding(mesh, P('batch', None, None))
        organism_sharding = NamedSharding(mesh, P('batch'))
        signal_sharding = NamedSharding(mesh, P('batch', None))
        
        print(f"\nChecking first 20 batches with device sharding...")
        
        data_loader_jax2 = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_to_jax,
            drop_last=True,
        )
        
        for batch_idx, (sequences, organism_ids, targets) in enumerate(data_loader_jax2):
            if batch_idx >= 20:
                break
            
            print(f"\nBatch {batch_idx}:")
            
            # Before sharding
            print("  Before device_put:")
            check_for_nan(targets['signal'], "    signal")
            
            # After sharding
            sequences_sharded = jax.device_put(sequences, batch_sharding)
            organism_ids_sharded = jax.device_put(organism_ids, organism_sharding)
            targets_sharded = {
                'signal': jax.device_put(targets['signal'], signal_sharding)
            }
            
            print("  After device_put:")
            check_for_nan(targets_sharded['signal'], "    signal")
    
    print(f"\n{'='*60}")
    print("PIPELINE TEST COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    test_pipeline()
