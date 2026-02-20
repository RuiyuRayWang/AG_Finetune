"""
Debug script to inspect actual training data being loaded.

This helps identify if extreme values are still present in the training data.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataset.dataset import CutAndTagDataset
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', required=True)
    parser.add_argument('--signal', required=True)
    parser.add_argument('--windows-bed', required=True)
    parser.add_argument('--window-size', type=int, default=131072)
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of random samples to check')
    
    args = parser.parse_args()
    
    print("="*80)
    print("DEBUGGING TRAINING DATA")
    print("="*80)
    print(f"Loading dataset...")
    
    dataset = CutAndTagDataset(
        fasta_path=args.fasta,
        signal_bedgraph=args.signal,
        genome_windows=args.windows_bed,
        window_size=args.window_size,
        organism_id=1,
        augment=False,  # No augmentation for debugging
        max_shift=0,
    )
    
    print(f"\nDataset has {len(dataset)} windows")
    print(f"\nSampling {args.num_samples} random windows...")
    
    # Sample random indices
    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
    
    all_signal_values = []
    extreme_count = 0
    
    for i, idx in enumerate(indices):
        seq, org_id, targets = dataset[idx]
        signal = targets['signal']
        
        max_val = np.max(signal)
        mean_val = np.mean(signal[signal > 0]) if np.any(signal > 0) else 0
        
        all_signal_values.extend(signal[signal > 0].tolist())
        
        if max_val > 80:
            extreme_count += 1
            chrom, start, end = dataset.regions[idx]
            print(f"  Sample {i+1}: EXTREME VALUE FOUND!")
            print(f"    Region: {chrom}:{start}-{end}")
            print(f"    Max: {max_val:.2f}, Mean: {mean_val:.2f}")
        
        if (i + 1) % 20 == 0:
            print(f"  Checked {i+1}/{len(indices)} samples...")
    
    print(f"\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    all_values = np.array(all_signal_values)
    
    print(f"Total non-zero values: {len(all_values):,}")
    print(f"Min:    {np.min(all_values):.6f}")
    print(f"Max:    {np.max(all_values):.6f}")
    print(f"Mean:   {np.mean(all_values):.6f}")
    print(f"Median: {np.median(all_values):.6f}")
    print(f"Std:    {np.std(all_values):.6f}")
    print()
    
    print("Percentiles:")
    for p in [90, 95, 99, 99.5, 99.9, 99.99, 100]:
        val = np.percentile(all_values, p)
        print(f"  {p:>6.2f}%: {val:>10.2f}")
    print()
    
    print(f"Samples with values > 80: {extreme_count}/{len(indices)}")
    if extreme_count > 0:
        print("❌ PROBLEM: Safe windows still contain extreme values!")
        print("   Action: Re-run identify_hypersensitive_regions.py with stricter parameters")
    else:
        print("✅ Good: No extreme values found in sampled windows")
    
    print("="*80)


if __name__ == '__main__':
    main()
