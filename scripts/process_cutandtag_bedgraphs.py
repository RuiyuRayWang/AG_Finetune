"""
Process CUT&Tag bedGraph files following AlphaGenome methodology.

This script:
1. Groups bedGraph files by cell type and target epitope
2. Averages replicates within each group 
3. Normalizes to 100 million insertions per track

Usage:
    python process_cutandtag_bigwigs.py --config samples.tsv --output-dir processed_bedgraphs --chrom-sizes mm10.chrom.sizes
    
Note: Input should be bedGraph files. Output will be bedGraph files that can be 
converted to BigWig using UCSC's bedGraphToBigWig tool if needed.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_bedgraph(bedgraph_path):
    """Load bedGraph file and return intervals by chromosome."""
    intervals_by_chrom = defaultdict(list)
    total_signal = 0.0
    
    with open(bedgraph_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            value = float(parts[3])
            
            intervals_by_chrom[chrom].append((start, end, value))
            total_signal += value * (end - start)
    
    return intervals_by_chrom, total_signal


def average_bedgraphs(bedgraph_files, chrom_sizes, output_path, normalize_to=100_000_000):
    """Average multiple bedGraph files and normalize."""
    print(f"  Processing {len(bedgraph_files)} replicates...")
    
    # Load all bedGraph files
    print(f"  Loading bedGraph files...")
    all_intervals = []
    totals = []
    
    for bg_file in bedgraph_files:
        intervals, total = load_bedgraph(bg_file)
        all_intervals.append(intervals)
        totals.append(total)
    
    print(f"  Replicate totals: {[f'{t:,.0f}' for t in totals]}")
    
    # Get all chromosomes present across all files
    all_chroms = set()
    for intervals in all_intervals:
        all_chroms.update(intervals.keys())
    
    # Filter to only chromosomes in chrom_sizes and sort
    valid_chroms = sorted([c for c in all_chroms if c in chrom_sizes])
    print(f"  Averaging across {len(valid_chroms)} chromosomes...")
    
    # Open output file
    with open(output_path, 'w') as out_f:
        averaged_total = 0.0
        
        for chrom in valid_chroms:
            chrom_len = chrom_sizes[chrom]
            
            # Create coverage array for this chromosome
            coverage = np.zeros(chrom_len, dtype=np.float64)
            
            # Add intervals from each replicate
            for intervals in all_intervals:
                if chrom in intervals:
                    for start, end, value in intervals[chrom]:
                        # Ensure we don't go beyond chromosome length
                        end = min(end, chrom_len)
                        if start < chrom_len:
                            coverage[start:end] += value
            
            # Average across replicates
            coverage /= len(bedgraph_files)
            
            # Calculate total signal for this chromosome
            averaged_total += np.sum(coverage)
            
            # Convert coverage array to intervals (where value changes)
            if np.any(coverage > 0):
                # Find positions where value changes
                diff_indices = np.where(np.diff(coverage) != 0)[0] + 1
                boundaries = np.concatenate([[0], diff_indices, [len(coverage)]])
                
                for i in range(len(boundaries) - 1):
                    start_pos = int(boundaries[i])
                    end_pos = int(boundaries[i + 1])
                    val = float(coverage[start_pos])
                    
                    # Only write non-zero intervals
                    if val > 0:
                        out_f.write(f"{chrom}\t{start_pos}\t{end_pos}\t{val:.6f}\n")
    
    print(f"  Pre-normalization total: {averaged_total:,.0f}")
    
    # Normalize
    print(f"  Normalizing to {normalize_to:,} insertions...")
    normalize_bedgraph(output_path, averaged_total, normalize_to)
    
    print(f"  ✓ Saved to {output_path}")


def normalize_bedgraph(bedgraph_path, current_total, target_total):
    """Normalize bedGraph file to target total signal."""
    if current_total == 0:
        print(f"  Warning: Total signal is 0, skipping normalization")
        return
    
    scale_factor = target_total / current_total
    print(f"  Scale factor: {scale_factor:.6f}")
    
    # Create temporary output
    temp_path = bedgraph_path + ".tmp"
    
    with open(bedgraph_path, 'r') as in_f, open(temp_path, 'w') as out_f:
        for line in in_f:
            if line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                out_f.write(line)
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            
            chrom = parts[0]
            start = parts[1]
            end = parts[2]
            value = float(parts[3]) * scale_factor
            
            out_f.write(f"{chrom}\t{start}\t{end}\t{value:.6f}\n")
    
    # Replace original
    os.replace(temp_path, bedgraph_path)


def load_chrom_sizes(chrom_sizes_file):
    """Load chromosome sizes from file."""
    chrom_sizes = {}
    with open(chrom_sizes_file) as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                chrom_sizes[parts[0]] = int(parts[1])
    return chrom_sizes


def main():
    parser = argparse.ArgumentParser(
        description='Process CUT&Tag bedGraph files following AlphaGenome methods'
    )
    parser.add_argument('--config', required=True,
                       help='TSV file with columns: sample_id, cell_type, epitope, bedgraph_path')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for processed bedGraphs')
    parser.add_argument('--chrom-sizes', required=True,
                       help='Chromosome sizes file (mm10.chrom.sizes)')
    parser.add_argument('--normalize-to', type=float, default=100_000_000,
                       help='Normalize to this many total insertions (default: 100M)')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading sample configuration...")
    df = pd.read_csv(args.config, sep='\t')
    
    # Accept either 'bigwig_path' or 'bedgraph_path' column for backwards compatibility
    if 'bedgraph_path' in df.columns:
        path_col = 'bedgraph_path'
    elif 'bigwig_path' in df.columns:
        path_col = 'bigwig_path'
    else:
        print(f"Error: Config file must have column 'bedgraph_path' or 'bigwig_path'")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)
    
    required_cols = ['sample_id', 'cell_type', 'epitope', path_col]
    
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Config file must have columns: {required_cols}")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} samples")
    
    # Load chromosome sizes
    print(f"\nLoading chromosome sizes from {args.chrom_sizes}...")
    chrom_sizes = load_chrom_sizes(args.chrom_sizes)
    print(f"Loaded {len(chrom_sizes)} chromosomes")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Group by cell type and epitope
    print("\nGrouping replicates...")
    groups = defaultdict(list)
    for _, row in df.iterrows():
        key = (row['cell_type'], row['epitope'])
        groups[key].append(row[path_col])
    
    print(f"Found {len(groups)} unique cell_type × epitope combinations:")
    for (cell_type, epitope), files in groups.items():
        print(f"  {cell_type} / {epitope}: {len(files)} replicates")
    
    # Process each group
    print("\n" + "="*60)
    print("PROCESSING GROUPS")
    print("="*60)
    
    for (cell_type, epitope), bedgraph_files in groups.items():
        print(f"\n{cell_type} / {epitope}")
        print("-" * 60)
        
        # Output filename
        output_name = f"{cell_type}_{epitope}_averaged_normalized.bedgraph"
        output_path = os.path.join(args.output_dir, output_name)
        
        # Average and normalize
        average_bedgraphs(bedgraph_files, chrom_sizes, output_path, args.normalize_to)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"\nProcessed bedGraphs saved to: {args.output_dir}")
    print("\nTo convert to BigWig format, use:")
    print("  bedGraphToBigWig input.bedgraph chrom.sizes output.bw")


if __name__ == '__main__':
    main()