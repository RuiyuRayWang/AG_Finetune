"""
Find and report genomic coordinates of extremely large values in bedGraph files.

This script identifies outlier intervals that may cause gradient explosion during training.
"""

import argparse
import sys
from collections import defaultdict


def analyze_bedgraph(bedgraph_path, threshold=100.0):
    """
    Find intervals with values above threshold and report statistics.
    
    Args:
        bedgraph_path: Path to bedGraph file
        threshold: Value threshold to report (default: 100.0)
    """
    large_intervals = []
    all_values = []
    
    print(f"Reading {bedgraph_path}...")
    print(f"Finding intervals with value > {threshold}")
    print("="*80)
    
    with open(bedgraph_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            value = float(parts[3])
            
            all_values.append(value)
            
            if value > threshold:
                large_intervals.append({
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'value': value,
                    'length': end - start
                })
    
    print(f"\nTotal intervals: {len(all_values):,}")
    print(f"Intervals > {threshold}: {len(large_intervals):,}")
    print()
    
    # Statistics on all values
    import numpy as np
    values_array = np.array(all_values)
    nonzero_values = values_array[values_array > 0]
    
    print("Overall Statistics:")
    print("-" * 80)
    print(f"  Min:        {np.min(values_array):.2f}")
    print(f"  Max:        {np.max(values_array):.2f}")
    print(f"  Mean:       {np.mean(values_array):.2f}")
    print(f"  Median:     {np.median(values_array):.2f}")
    print(f"  Std:        {np.std(values_array):.2f}")
    print()
    print(f"  Non-zero values: {len(nonzero_values):,}")
    print(f"  Non-zero mean:   {np.mean(nonzero_values):.2f}")
    print(f"  Non-zero median: {np.median(nonzero_values):.2f}")
    print()
    
    # Percentiles
    percentiles = [90, 95, 99, 99.5, 99.9, 99.99]
    print("Percentiles:")
    print("-" * 80)
    for p in percentiles:
        val = np.percentile(values_array, p)
        print(f"  {p:>6.2f}%:  {val:>10.2f}")
    print()
    
    if large_intervals:
        # Sort by value descending
        large_intervals.sort(key=lambda x: x['value'], reverse=True)
        
        print(f"\nTop {min(50, len(large_intervals))} intervals with value > {threshold}:")
        print("-" * 80)
        print(f"{'Chrom':<10} {'Start':>12} {'End':>12} {'Length':>10} {'Value':>12}")
        print("-" * 80)
        
        for interval in large_intervals[:50]:
            print(f"{interval['chrom']:<10} {interval['start']:>12,} {interval['end']:>12,} "
                  f"{interval['length']:>10,} {interval['value']:>12.2f}")
        
        # Group by chromosome
        print(f"\n\nIntervals grouped by chromosome:")
        print("-" * 80)
        chrom_counts = defaultdict(int)
        chrom_max = defaultdict(float)
        for interval in large_intervals:
            chrom_counts[interval['chrom']] += 1
            chrom_max[interval['chrom']] = max(chrom_max[interval['chrom']], interval['value'])
        
        for chrom in sorted(chrom_counts.keys()):
            print(f"  {chrom:<10}  {chrom_counts[chrom]:>6,} intervals  (max value: {chrom_max[chrom]:>10.2f})")
        
        # Save to file
        output_file = bedgraph_path.replace('.bedgraph', f'_large_values_gt{int(threshold)}.bed')
        print(f"\n\nSaving all {len(large_intervals):,} intervals to: {output_file}")
        with open(output_file, 'w') as out:
            out.write(f"# Intervals with value > {threshold} from {bedgraph_path}\n")
            out.write(f"# chrom\tstart\tend\tvalue\tlength\n")
            for interval in large_intervals:
                out.write(f"{interval['chrom']}\t{interval['start']}\t{interval['end']}\t"
                         f"{interval['value']:.6f}\t{interval['length']}\n")
        print(f"✓ Saved to {output_file}")
        
        # UCSC browser links for top 10
        print("\n\nUCSC Genome Browser links (top 10):")
        print("-" * 80)
        genome = 'mm10'  # Change if using different genome
        for i, interval in enumerate(large_intervals[:10], 1):
            chrom = interval['chrom']
            start = max(0, interval['start'] - 5000)  # Add 5kb padding
            end = interval['end'] + 5000
            value = interval['value']
            url = f"https://genome.ucsc.edu/cgi-bin/hgTracks?db={genome}&position={chrom}%3A{start}-{end}"
            print(f"{i:2d}. {chrom}:{interval['start']:,}-{interval['end']:,}  (value={value:.1f})")
            print(f"    {url}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description='Find genomic coordinates with extremely large values in bedGraph'
    )
    parser.add_argument('bedgraph', help='Input bedGraph file')
    parser.add_argument('--threshold', type=float, default=100.0,
                       help='Value threshold to report (default: 100.0)')
    
    args = parser.parse_args()
    
    analyze_bedgraph(args.bedgraph, args.threshold)


if __name__ == '__main__':
    main()
