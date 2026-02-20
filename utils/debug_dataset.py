"""
Debug script to inspect CutAndTagDataset for NaN/Inf values.

Usage:
    python scripts/debug_dataset.py
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import CutAndTagDataset


def check_dataset():
    """Load dataset and check for NaN/Inf values."""
    
    print("="*60)
    print("DATASET DEBUG")
    print("="*60)
    
    # Create dataset with same parameters as training
    dataset = CutAndTagDataset(
        fasta_path='/home/ray/AG_Finetune/data/resources/mm10.fa',
        signal_bedgraph='/home/ray/AG_Finetune/data/CUTTag/processed_bedgraph/D1MSN_H3K27ac_averaged_normalized.bedgraph',
        genome_windows='/home/ray/AG_Finetune/exports/peak_safe_windows/D1MSN_H3K27ac_merged_peak_safe_windows.bed',
        window_size=131072,
        organism_id=1,
        augment=False,  # Disable augmentation for reproducible testing
        max_shift=1024,
    )
    
    print(f"\nDataset loaded: {len(dataset)} samples")
    print("="*60)
    
    # Check first 100 samples
    print("\nChecking first 100 samples for NaN/Inf...")
    
    nan_samples = []
    inf_samples = []
    zero_samples = []
    
    for i in range(min(100, len(dataset))):
        seq, org_id, targets = dataset[i]
        signal = targets['signal']
        
        # Check for NaN
        if np.any(np.isnan(signal)):
            nan_samples.append(i)
            print(f"  Sample {i}: Contains NaN")
            print(f"    NaN count: {np.sum(np.isnan(signal))}")
            print(f"    Signal stats: min={np.nanmin(signal):.6f}, max={np.nanmax(signal):.6f}, mean={np.nanmean(signal):.6f}")
        
        # Check for Inf
        if np.any(np.isinf(signal)):
            inf_samples.append(i)
            print(f"  Sample {i}: Contains Inf")
            print(f"    Inf count: {np.sum(np.isinf(signal))}")
        
        # Check for all zeros
        if np.all(signal == 0.0):
            zero_samples.append(i)
    
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"{'='*60}")
    print(f"Total samples checked: {min(100, len(dataset))}")
    print(f"Samples with NaN: {len(nan_samples)}")
    print(f"Samples with Inf: {len(inf_samples)}")
    print(f"Samples with all zeros: {len(zero_samples)}")
    
    if nan_samples:
        print(f"\nNaN sample indices: {nan_samples[:10]}..." if len(nan_samples) > 10 else f"\nNaN sample indices: {nan_samples}")
    
    # Inspect the raw BedGraph data
    print(f"\n{'='*60}")
    print("INSPECTING RAW BEDGRAPH DATA:")
    print(f"{'='*60}")
    
    import pandas as pd
    df = pd.read_csv(
        '/home/ray/AG_Finetune/data/CUTTag/processed_bedgraph/D1MSN_H3K27ac_averaged_normalized.bedgraph',
        sep='\t',
        header=None,
        names=['chrom', 'start', 'end', 'score'],
        comment='#',
        nrows=1000  # Check first 1000 rows
    )
    
    print(f"Total rows loaded: {len(df)}")
    print(f"\nColumn types:")
    print(df.dtypes)
    
    print(f"\nNaN counts per column:")
    print(df.isna().sum())
    
    print(f"\nInf counts per column:")
    for col in ['start', 'end', 'score']:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            inf_count = np.isinf(df[col]).sum()
            print(f"  {col}: {inf_count}")
    
    print(f"\nFirst 10 rows:")
    print(df.head(10))
    
    print(f"\nScore statistics:")
    print(f"  Min: {df['score'].min():.6f}")
    print(f"  Max: {df['score'].max():.6f}")
    print(f"  Mean: {df['score'].mean():.6f}")
    print(f"  Median: {df['score'].median():.6f}")
    print(f"  Std: {df['score'].std():.6f}")
    
    # Check for problematic rows
    print(f"\n{'='*60}")
    print("PROBLEMATIC ROWS:")
    print(f"{'='*60}")
    
    # Reload full file to find all problematic rows
    df_full = pd.read_csv(
        '/home/ray/AG_Finetune/data/CUTTag/processed_bedgraph/D1MSN_H3K27ac_averaged_normalized.bedgraph',
        sep='\t',
        header=None,
        names=['chrom', 'start', 'end', 'score'],
        comment='#'
    )
    
    print(f"Total rows in file: {len(df_full)}")
    
    # Find rows with NaN
    nan_rows = df_full[df_full.isna().any(axis=1)]
    print(f"Rows with NaN: {len(nan_rows)}")
    if len(nan_rows) > 0:
        print("\nFirst 5 NaN rows:")
        print(nan_rows.head())
    
    # Find rows with Inf
    inf_mask = np.isinf(df_full['score'])
    inf_rows = df_full[inf_mask]
    print(f"\nRows with Inf in score: {len(inf_rows)}")
    if len(inf_rows) > 0:
        print("\nFirst 5 Inf rows:")
        print(inf_rows.head())
    
    # Check data types
    print(f"\n{'='*60}")
    print("DATA TYPE CHECK:")
    print(f"{'='*60}")
    for idx in range(min(10, len(df_full))):
        row = df_full.iloc[idx]
        print(f"Row {idx}: chrom={row['chrom']!r}, start={row['start']!r} ({type(row['start']).__name__}), "
              f"end={row['end']!r} ({type(row['end']).__name__}), score={row['score']!r} ({type(row['score']).__name__})")


if __name__ == '__main__':
    check_dataset()
