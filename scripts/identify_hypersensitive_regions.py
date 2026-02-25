"""
Identify Tn5-hypersensitive regions from CUT&Tag data for exclusion during training.

Strategy (distance-based clustering):
1. Find intervals with extremely high values (seed threshold, e.g., > 80)
2. For each seed, expand to include nearby intervals within clustering window
3. Merge overlapping clusters to get comprehensive hypersensitive regions
4. Output regions for exclusion during training

This prevents gradient explosion during LoRA fine-tuning by excluding regions
with extreme Tn5 accessibility. Uses physical proximity rather than gene names
to avoid false positives.

Usage:
    # Step 1: Find large values and annotate with genes
    python scripts/find_large_bedgraph_values.py \\
        data/CUTTag/processed_bedgraph/D2MSN_H3K4me1_averaged_normalized.bedgraph \\
        --threshold 20
    
    # Step 2: Identify hypersensitive clusters
    python scripts/identify_hypersensitive_regions.py \\
        exports/D2MSN_H3K4me1_averaged_normalized_large_values_gt20.bed \\
        --seed-threshold 80 \\
        --cluster-window 50000 \\
        --output exports/D2MSN_H3K4me1_hypersensitive_regions.bed
    
    # Step 3: Create safe genome windows (inverse)
    python scripts/identify_hypersensitive_regions.py \\
        exports/D2MSN_H3K4me1_averaged_normalized_large_values_gt20.bed \\
        --seed-threshold 80 \\
        --cluster-window 50000 \\
        --output exports/D2MSN_H3K4me1_hypersensitive_regions.bed \\
        --output-safe exports/D2MSN_H3K4me1_safe_windows.bed \\
        --chrom-sizes data/resources/mm10.chrom.sizes
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path


def load_annotated_intervals(bed_path):
    """
    Load intervals from annotated BED file.
    
    Expected format (from find_large_bedgraph_values.py):
        chrom  start  end  value  length  gene  gene_distance
    
    Returns:
        List of dicts with interval information
    """
    intervals = []
    
    with open(bed_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue
            
            interval = {
                'chrom': parts[0],
                'start': int(parts[1]),
                'end': int(parts[2]),
                'value': float(parts[3]),
                'length': int(parts[4]),
                'gene': parts[5]
            }
            
            if len(parts) >= 7:
                try:
                    interval['gene_distance'] = int(parts[6])
                except:
                    interval['gene_distance'] = parts[6]  # Could be "overlap"
            
            intervals.append(interval)
    
    return intervals


def identify_hypersensitive_clusters(intervals, seed_threshold=80.0, cluster_window=50000):
    """
    Identify hypersensitive clusters using distance-based approach.
    
    Algorithm:
    1. Find seed intervals with value >= seed_threshold
    2. For each seed, expand to include all intervals within cluster_window distance
    3. Merge overlapping clusters
    
    Args:
        intervals: List of interval dicts
        seed_threshold: Value threshold to identify seed intervals (default: 80.0)
        cluster_window: Distance in bp to expand around seeds (default: 50,000 bp)
    
    Returns:
        List of (chrom, start, end, num_intervals, max_value) tuples
    """
    # Group intervals by chromosome
    by_chrom = defaultdict(list)
    for interval in intervals:
        by_chrom[interval['chrom']].append(interval)
    
    # Sort intervals by position within each chromosome
    for chrom in by_chrom:
        by_chrom[chrom].sort(key=lambda x: x['start'])
    
    # Find seed intervals (high values)
    seeds = []
    for interval in intervals:
        if interval['value'] >= seed_threshold:
            seeds.append(interval)
    
    print(f"  Found {len(seeds)} seed intervals with value >= {seed_threshold}")
    
    # Build clusters around seeds
    clusters = []
    
    for seed in seeds:
        chrom = seed['chrom']
        seed_start = seed['start']
        seed_end = seed['end']
        
        # Define cluster window around seed
        cluster_start = seed_start - cluster_window
        cluster_end = seed_end + cluster_window
        
        # Collect all intervals within this window on same chromosome
        cluster_intervals = []
        for interval in by_chrom[chrom]:
            # Check if interval overlaps with cluster window
            if not (interval['end'] < cluster_start or interval['start'] > cluster_end):
                cluster_intervals.append(interval)
        
        if cluster_intervals:
            # Get actual bounds of cluster (min start, max end)
            cluster_actual_start = min(i['start'] for i in cluster_intervals)
            cluster_actual_end = max(i['end'] for i in cluster_intervals)
            max_value = max(i['value'] for i in cluster_intervals)
            
            clusters.append({
                'chrom': chrom,
                'start': cluster_actual_start,
                'end': cluster_actual_end,
                'num_intervals': len(cluster_intervals),
                'max_value': max_value,
                'seed_value': seed['value']
            })
    
    print(f"  Generated {len(clusters)} initial clusters")
    
    # Merge overlapping clusters
    merged_clusters = merge_clusters(clusters)
    
    print(f"  Merged into {len(merged_clusters)} non-overlapping clusters")
    
    return merged_clusters


def merge_clusters(clusters):
    """
    Merge overlapping or adjacent clusters.
    
    Args:
        clusters: List of cluster dicts
    
    Returns:
        List of merged (chrom, start, end, num_intervals, max_value) tuples
    """
    if not clusters:
        return []
    
    # Group by chromosome
    by_chrom = defaultdict(list)
    for cluster in clusters:
        by_chrom[cluster['chrom']].append(cluster)
    
    merged = []
    
    for chrom in sorted(by_chrom.keys()):
        chrom_clusters = sorted(by_chrom[chrom], key=lambda x: x['start'])
        
        if not chrom_clusters:
            continue
        
        current = chrom_clusters[0].copy()
        
        for cluster in chrom_clusters[1:]:
            # Merge if overlapping or adjacent (within 1kb)
            if cluster['start'] <= current['end'] + 1000:
                # Expand current cluster
                current['end'] = max(current['end'], cluster['end'])
                current['num_intervals'] += cluster['num_intervals']
                current['max_value'] = max(current['max_value'], cluster['max_value'])
            else:
                # Save current and start new
                merged.append((
                    current['chrom'],
                    current['start'],
                    current['end'],
                    current['num_intervals'],
                    current['max_value']
                ))
                current = cluster.copy()
        
        # Add last cluster
        merged.append((
            current['chrom'],
            current['start'],
            current['end'],
            current['num_intervals'],
            current['max_value']
        ))
    
    return sorted(merged)


def load_chrom_sizes(chrom_sizes_file):
    """Load chromosome sizes from file."""
    chrom_sizes = {}
    with open(chrom_sizes_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                chrom_sizes[parts[0]] = int(parts[1])
    return chrom_sizes


def invert_regions(hypersensitive_regions, chrom_sizes, min_window_size=262144):
    """
    Create safe windows by inverting hypersensitive regions.
    
    Args:
        hypersensitive_regions: List of (chrom, start, end, ...) tuples to exclude
        chrom_sizes: Dict of chrom -> size
        min_window_size: Minimum size for safe windows
    
    Returns:
        List of (chrom, start, end) tuples for safe regions
    """
    # Group by chromosome - extract just chrom, start, end
    by_chrom = defaultdict(list)
    for region in hypersensitive_regions:
        chrom = region[0]
        start = region[1]
        end = region[2]
        by_chrom[chrom].append((start, end))
    
    safe_windows = []
    
    for chrom in sorted(chrom_sizes.keys()):
        chrom_size = chrom_sizes[chrom]
        excluded = sorted(by_chrom.get(chrom, []))
        
        current_pos = 0
        
        for start, end in excluded:
            # Add safe region before this excluded region
            if start - current_pos >= min_window_size:
                safe_windows.append((chrom, current_pos, start))
            current_pos = max(current_pos, end)
        
        # Add remaining region at end of chromosome
        if chrom_size - current_pos >= min_window_size:
            safe_windows.append((chrom, current_pos, chrom_size))
    
    return safe_windows


def main():
    parser = argparse.ArgumentParser(
        description='Identify Tn5-hypersensitive regions for training exclusion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input_bed', help='Annotated BED file from find_large_bedgraph_values.py')
    parser.add_argument('--seed-threshold', type=float, default=80.0,
                       help='Seed threshold to identify hypersensitive sites (default: 80.0)')
    parser.add_argument('--cluster-window', type=int, default=50000,
                       help='Distance in bp to expand around seeds for clustering (default: 50,000)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output BED file for hypersensitive regions')
    parser.add_argument('--output-safe', type=str, default=None,
                       help='Optional: Output BED file for safe windows (inverted regions)')
    parser.add_argument('--chrom-sizes', type=str, default=None,
                       help='Chromosome sizes file (required for --output-safe)')
    parser.add_argument('--min-window-size', type=int, default=262144,
                       help='Minimum safe window size in bp (default: 262144 = 2^18)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.output_safe and not args.chrom_sizes:
        parser.error("--chrom-sizes is required when using --output-safe")
    
    print("="*80)
    print("Identifying Tn5-hypersensitive regions (distance-based clustering)")
    print("="*80)
    print(f"Input:            {args.input_bed}")
    print(f"Seed threshold:   {args.seed_threshold} (identifies hypersensitive peaks)")
    print(f"Cluster window:   {args.cluster_window:,} bp (expansion around seeds)")
    print()
    
    # Load intervals
    print("Loading annotated intervals...")
    intervals = load_annotated_intervals(args.input_bed)
    print(f"✓ Loaded {len(intervals):,} intervals with values above baseline")
    print()
    
    # Identify hypersensitive clusters using distance-based approach
    print(f"Clustering hypersensitive regions...")
    clusters = identify_hypersensitive_clusters(
        intervals, 
        args.seed_threshold,
        args.cluster_window
    )
    print()
    
    # Display cluster statistics
    print(f"Cluster Statistics:")
    print("-" * 80)
    total_bp = sum(end - start for chrom, start, end, num_intervals, max_value in clusters)
    print(f"  Total clusters:   {len(clusters):,}")
    print(f"  Total bp:         {total_bp:,} ({total_bp/1e6:.2f} Mb)")
    print()
    
    # Show top clusters by size and value
    print("Top 10 clusters by max value:")
    print("-" * 80)
    print(f"{'Chrom':<10} {'Start':>12} {'End':>12} {'Size':>12} {'Intervals':>10} {'Max Value':>12}")
    print("-" * 80)
    sorted_by_value = sorted(clusters, key=lambda x: x[4], reverse=True)
    for chrom, start, end, num_intervals, max_value in sorted_by_value[:10]:
        size = end - start
        print(f"{chrom:<10} {start:>12,} {end:>12,} {size:>12,} {num_intervals:>10,} {max_value:>12.1f}")
    print()
    
    # Group by chromosome
    chrom_counts = defaultdict(int)
    chrom_max = defaultdict(float)
    for chrom, start, end, num_intervals, max_value in clusters:
        chrom_counts[chrom] += 1
        chrom_max[chrom] = max(chrom_max[chrom], max_value)
    
    print("Clusters by chromosome:")
    print("-" * 80)
    for chrom in sorted(chrom_counts.keys()):
        print(f"  {chrom:<10}  {chrom_counts[chrom]:>6,} clusters  (max value: {chrom_max[chrom]:>10.1f})")
    print()
    
    # Save hypersensitive regions
    print(f"Saving hypersensitive regions to: {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        f.write(f"# Tn5-hypersensitive regions (distance-based clustering)\n")
        f.write(f"# Source: {args.input_bed}\n")
        f.write(f"# Seed threshold: {args.seed_threshold}\n")
        f.write(f"# Cluster window: {args.cluster_window:,} bp\n")
        f.write(f"# Total clusters: {len(clusters)}\n")
        f.write(f"# Total bp: {total_bp:,} ({total_bp/1e6:.2f} Mb)\n")
        f.write(f"# chrom\tstart\tend\tnum_intervals\tmax_value\n")
        for chrom, start, end, num_intervals, max_value in clusters:
            f.write(f"{chrom}\t{start}\t{end}\t{num_intervals}\t{max_value:.2f}\n")
    print(f"✓ Saved {len(clusters):,} clusters")
    print()
    
    # Generate safe windows if requested
    if args.output_safe:
        print(f"Generating safe windows (inverted regions)...")
        chrom_sizes = load_chrom_sizes(args.chrom_sizes)
        print(f"✓ Loaded {len(chrom_sizes)} chromosome sizes")
        
        safe_windows = invert_regions(
            clusters, 
            chrom_sizes, 
            args.min_window_size
        )
        
        total_safe_bp = sum(end - start for chrom, start, end in safe_windows)
        print(f"✓ Generated {len(safe_windows):,} safe windows")
        print(f"  Total: {total_safe_bp:,} bp ({total_safe_bp/1e6:.2f} Mb)")
        print()
        
        print(f"Saving safe windows to: {args.output_safe}")
        with open(args.output_safe, 'w') as f:
            f.write(f"# Safe genomic windows (excluding hypersensitive regions)\n")
            f.write(f"# Min window size: {args.min_window_size:,} bp\n")
            f.write(f"# Total windows: {len(safe_windows)}\n")
            f.write(f"# Total bp: {total_safe_bp:,} ({total_safe_bp/1e6:.2f} Mb)\n")
            f.write(f"# chrom\tstart\tend\n")
            for chrom, start, end in safe_windows:
                f.write(f"{chrom}\t{start}\t{end}\n")
        print(f"✓ Saved {len(safe_windows):,} safe windows")
        print()
    
    print("="*80)
    print("Done! Use the output files for training:")
    print(f"  Exclude: {args.output}")
    if args.output_safe:
        print(f"  Or use safe windows: {args.output_safe}")
        print()
        print("Example training command:")
        print("  python scripts/train.py \\")
        print("    --genome-windows", args.output_safe, "\\")
        print("    ...")
    print("="*80)


if __name__ == '__main__':
    main()
