"""
Create peak-based safe training windows by intersecting SEACR peaks with safe windows.

This combines:
1. SEACR peaks (high-signal regions, reduces sparsity)
2. Safe windows (excludes Tn5 hypersensitive sites, prevents gradient explosion)

Usage:
    python scripts/create_peak_safe_windows.py \
        --metadata data/metadata.tsv \
        --output-dir exports/peak_safe_windows
"""

import argparse
import os
import subprocess
from pathlib import Path
import pandas as pd


def run_bedtools_intersect(peaks_bed, safe_windows_bed, output_bed):
    """
    Intersect SEACR peaks with safe windows using bedtools.
    Outputs standard 3-column BED format (chr, start, end).
    
    Args:
        peaks_bed: SEACR peak file (6 columns)
        safe_windows_bed: Safe genomic windows file
        output_bed: Output intersection file (3 columns)
    """
    # First intersect to get overlapping peaks
    cmd = [
        'bedtools', 'intersect',
        '-a', peaks_bed,      # SEACR peaks
        '-b', safe_windows_bed,  # Safe windows
        '-wa',                # Write original peak entry from -a
    ]
    
    # Run intersect and extract only first 3 columns (chr, start, end)
    with open(output_bed, 'w') as out_f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate()
        
        if proc.returncode != 0:
            print(f"  ⚠️  Error running bedtools: {stderr}")
            return 0
        
        # Parse output and write only chr, start, end
        for line in stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                # Write only first 3 columns (standard BED format)
                out_f.write(f"{parts[0]}\t{parts[1]}\t{parts[2]}\n")
    
    # Count output peaks
    with open(output_bed, 'r') as f:
        num_peaks = sum(1 for line in f if line.strip() and not line.startswith('#'))
    
    return num_peaks


def process_sample(row, safe_windows_dir, output_dir):
    """
    Process one sample: intersect its peaks with corresponding safe windows.
    
    Args:
        row: DataFrame row with sample metadata
        safe_windows_dir: Directory containing safe window BED files
        output_dir: Output directory for peak-safe windows
    
    Returns:
        Dict with statistics
    """
    sample_id = row['sample_id']
    cell_type = row['cell_type']
    epitope = row['epitope']
    peaks_bed = row['SEACR_peaks']
    
    # Construct safe windows filename
    safe_windows_filename = f"{cell_type}_{epitope}_safe_windows.bed"
    safe_windows_bed = os.path.join(safe_windows_dir, safe_windows_filename)
    
    # Check if files exist
    if not os.path.exists(peaks_bed):
        print(f"  ⚠️  Peak file not found: {peaks_bed}")
        return None
    
    if not os.path.exists(safe_windows_bed):
        print(f"  ⚠️  Safe windows not found: {safe_windows_bed}")
        return None
    
    # Output filename
    output_bed = os.path.join(output_dir, f"{sample_id}_peak_safe_windows.bed")
    
    # Count input peaks
    with open(peaks_bed, 'r') as f:
        num_input_peaks = sum(1 for line in f if line.strip() and not line.startswith('#'))
    
    print(f"\n{sample_id} ({cell_type} {epitope})")
    print(f"  Input peaks: {num_input_peaks:,}")
    
    # Run intersection
    num_output_peaks = run_bedtools_intersect(peaks_bed, safe_windows_bed, output_bed)
    
    num_excluded = num_input_peaks - num_output_peaks
    pct_excluded = 100.0 * num_excluded / num_input_peaks if num_input_peaks > 0 else 0.0
    
    print(f"  Safe peaks:  {num_output_peaks:,}")
    print(f"  Excluded:    {num_excluded:,} ({pct_excluded:.2f}%)")
    
    return {
        'sample_id': sample_id,
        'cell_type': cell_type,
        'epitope': epitope,
        'input_peaks': num_input_peaks,
        'safe_peaks': num_output_peaks,
        'excluded_peaks': num_excluded,
        'pct_excluded': pct_excluded,
        'output_file': output_bed
    }


def merge_peaks_by_condition(stats_df, output_dir):
    """
    Merge peak-safe windows by cell_type + epitope combination.
    
    Args:
        stats_df: DataFrame with per-sample statistics
        output_dir: Output directory
    
    Returns:
        Dict of condition -> merged BED path
    """
    print("\n" + "="*80)
    print("Merging peaks by condition (cell_type + epitope)")
    print("="*80)
    
    merged_files = {}
    
    # Group by cell_type and epitope
    for (cell_type, epitope), group in stats_df.groupby(['cell_type', 'epitope']):
        condition = f"{cell_type}_{epitope}"
        print(f"\n{condition}:")
        print(f"  Replicates: {len(group)}")
        
        # Collect all peak files for this condition
        peak_files = group['output_file'].tolist()
        
        # Merge output filename
        merged_bed = os.path.join(output_dir, f"{condition}_merged_peak_safe_windows.bed")
        
        # Concatenate and sort all peaks
        cmd = f"cat {' '.join(peak_files)} | sort -k1,1 -k2,2n > {merged_bed}"
        subprocess.run(cmd, shell=True, check=True)
        
        # Count merged peaks
        with open(merged_bed, 'r') as f:
            num_peaks = sum(1 for line in f if line.strip())
        
        print(f"  Merged peaks: {num_peaks:,}")
        print(f"  Output: {merged_bed}")
        
        merged_files[condition] = merged_bed
    
    return merged_files


def main():
    parser = argparse.ArgumentParser(
        description='Create peak-based safe training windows',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--metadata', required=True,
                       help='Metadata TSV with sample info and SEACR peaks')
    parser.add_argument('--safe-windows-dir', 
                       default='/home/ray/AG_Finetune/exports',
                       help='Directory containing safe window BED files')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for peak-safe windows')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("Creating Peak-Based Safe Training Windows")
    print("="*80)
    print(f"Metadata:         {args.metadata}")
    print(f"Safe windows dir: {args.safe_windows_dir}")
    print(f"Output dir:       {args.output_dir}")
    print()
    
    # Load metadata
    metadata_df = pd.read_csv(args.metadata, sep='\t')
    print(f"✓ Loaded {len(metadata_df)} samples from metadata")
    
    # Process each sample
    print("\n" + "="*80)
    print("Processing Individual Samples")
    print("="*80)
    
    stats = []
    for idx, row in metadata_df.iterrows():
        result = process_sample(row, args.safe_windows_dir, args.output_dir)
        if result is not None:
            stats.append(result)
    
    # Create summary DataFrame
    stats_df = pd.DataFrame(stats)
    
    # Summary statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)
    print(f"\nTotal samples processed: {len(stats_df)}")
    print(f"Total input peaks:       {stats_df['input_peaks'].sum():,}")
    print(f"Total safe peaks:        {stats_df['safe_peaks'].sum():,}")
    print(f"Total excluded:          {stats_df['excluded_peaks'].sum():,} ({100.0 * stats_df['excluded_peaks'].sum() / stats_df['input_peaks'].sum():.2f}%)")
    
    print("\n\nBy Cell Type and Epitope:")
    print("-"*80)
    summary = stats_df.groupby(['cell_type', 'epitope']).agg({
        'input_peaks': 'sum',
        'safe_peaks': 'sum',
        'excluded_peaks': 'sum'
    })
    summary['pct_excluded'] = 100.0 * summary['excluded_peaks'] / summary['input_peaks']
    print(summary.to_string())
    
    # Save summary
    summary_file = os.path.join(args.output_dir, 'peak_safe_windows_summary.tsv')
    stats_df.to_csv(summary_file, sep='\t', index=False)
    print(f"\n✓ Saved summary to: {summary_file}")
    
    # Merge peaks by condition
    merged_files = merge_peaks_by_condition(stats_df, args.output_dir)
    
    # Create union of all peaks for multi-head training
    print("\n" + "="*80)
    print("Creating Union of All Peaks (for multi-head training)")
    print("="*80)
    
    all_merged_files = list(merged_files.values())
    union_bed = os.path.join(args.output_dir, 'all_conditions_union_peak_safe_windows.bed')
    
    # Concatenate all condition files and merge overlapping regions with bedtools
    print(f"\nMerging {len(all_merged_files)} condition files...")
    
    # Concatenate
    temp_cat = os.path.join(args.output_dir, 'temp_all_concatenated.bed')
    cmd = f"cat {' '.join(all_merged_files)} | sort -k1,1 -k2,2n > {temp_cat}"
    subprocess.run(cmd, shell=True, check=True)
    
    # Merge overlapping regions
    cmd = f"bedtools merge -i {temp_cat} > {union_bed}"
    subprocess.run(cmd, shell=True, check=True)
    
    # Clean up temp file
    os.remove(temp_cat)
    
    # Count union peaks
    with open(union_bed, 'r') as f:
        num_union_peaks = sum(1 for line in f if line.strip())
    
    print(f"✓ Union peaks: {num_union_peaks:,}")
    print(f"✓ Saved to: {union_bed}")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nGenerated {len(stats_df)} per-sample peak-safe window files")
    print(f"Generated {len(merged_files)} merged condition files")
    print(f"Generated 1 union file for multi-head training: {num_union_peaks:,} peaks")
    print()
    print("Files for single-condition training:")
    for condition, filepath in merged_files.items():
        print(f"  {condition}: {filepath}")
    print()
    print(f"File for multi-head training:")
    print(f"  All conditions: {union_bed}")
    print()


if __name__ == '__main__':
    main()
