"""
Debug script to check for signal amplification in CUT&Tag data loading.

This script tests whether high signal values in training are due to:
1. Actual bedgraph values
2. Data loading bugs (overlapping intervals being summed)
3. Batch collation issues
4. JAX sharding issues
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader
from dataset.dataset import CutAndTagDataset

def collate_to_jax(batch):
    """Same as in train.py"""
    sequences = [item[0] for item in batch]
    organism_ids = [item[1] for item in batch]
    signals = [item[2]['signal'] for item in batch]
    
    return (
        jnp.array(sequences, dtype=jnp.float32),
        jnp.array(organism_ids, dtype=jnp.int32),
        {'signal': jnp.array(signals, dtype=jnp.float32)}
    )

def main():
    # Same paths as in training
    fasta_path = '/home/ray/AG_Finetune/data/resources/mm10.fa'
    signal_path = '/home/ray/AG_Finetune/data/CUTTag/processed_bedgraph/D1MSN_H3K27ac_averaged_normalized.bedgraph'
    
    print("Creating dataset...")
    dataset = CutAndTagDataset(
        fasta_path=fasta_path,
        signal_bedgraph=signal_path,
        genome_windows=None,  # Random sampling like in the failed run
        window_size=131072,
        organism_id=1,
        augment=True,
        max_shift=1024,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader with batch_size=8, like in training
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_to_jax,
        num_workers=0,
    )
    
    print("\n" + "="*60)
    print("Checking batches for high signal values...")
    print("="*60)
    
    high_value_batches = []
    
    for batch_idx, (sequences, organism_ids, targets) in enumerate(dataloader):
        # Get max value in this batch
        max_val = float(jnp.max(targets['signal']))
        mean_val = float(jnp.mean(targets['signal']))
        
        # Check if max > 10,000 (anything suspicious)
        if max_val > 10000.0:
            high_value_batches.append((batch_idx, max_val, mean_val))
            print(f"Batch {batch_idx}: MAX={max_val:.2f}, MEAN={mean_val:.6f}")
            
            # For the first high-value batch, print details
            if len(high_value_batches) == 1:
                print("\n  Detailed analysis of first high-value batch:")
                signal_arr = np.array(targets['signal'])
                for sample_idx in range(signal_arr.shape[0]):
                    sample_max = signal_arr[sample_idx].max()
                    sample_mean = signal_arr[sample_idx].mean()
                    sample_nonzero = (signal_arr[sample_idx] > 0).sum()
                    if sample_max > 10000.0:
                        print(f"    Sample {sample_idx}: max={sample_max:.2f}, mean={sample_mean:.6f}, nonzero_bins={sample_nonzero}")
                        # Find positions with highest values
                        high_positions = np.where(signal_arr[sample_idx] > 5000.0)[0]
                        if len(high_positions) > 0:
                            print(f"      Positions with value > 5000: {len(high_positions)} positions")
                            print(f"      Values at these positions: {signal_arr[sample_idx][high_positions[:10]]}")
        
        # Stop after checking 300 batches (cover the range where corruption happened)
        if batch_idx >= 300:
            break
    
    print("\n" + "="*60)
    print(f"Summary: Found {len(high_value_batches)} batches with max > 10,000")
    print("="*60)
    
    if high_value_batches:
        print("\nHigh value batches:")
        for batch_idx, max_val, mean_val in high_value_batches[:20]:  # Show first 20
            print(f"  Batch {batch_idx}: max={max_val:.2f}, mean={mean_val:.6f}")
        
        if len(high_value_batches) > 20:
            print(f"  ... and {len(high_value_batches) - 20} more")
    
    # Now check directly from bedgraph to see true max
    print("\n" + "="*60)
    print("Checking bedgraph file directly...")
    print("="*60)
    
    import pandas as pd
    df = pd.read_csv(
        signal_path,
        sep='\t',
        header=None,
        names=['chrom', 'start', 'end', 'score'],
        comment='#'
    )
    
    print(f"Bedgraph max value: {df['score'].max():.2f}")
    print(f"Bedgraph mean value: {df['score'].mean():.6f}")
    print(f"Bedgraph 99.9th percentile: {df['score'].quantile(0.999):.2f}")
    
    # Check for duplicate intervals (same chrom, start, end)
    duplicates = df.groupby(['chrom', 'start', 'end']).size()
    dup_count = (duplicates > 1).sum()
    if dup_count > 0:
        print(f"\n⚠️  WARNING: Found {dup_count} duplicate intervals in bedgraph!")
        print("  This could cause issues if multiple values for same position are being summed.")
    else:
        print("\n✓ No duplicate intervals found in bedgraph")

if __name__ == '__main__':
    main()
