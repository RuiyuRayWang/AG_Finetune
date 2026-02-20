"""
Find and report genomic coordinates of extremely large values in bedGraph files.

This script identifies outlier intervals that may cause gradient explosion during training.
Annotates intervals with closest gene names from gencode GTF.
"""

import argparse
import sys
import os
import gzip
from collections import defaultdict


def load_gencode_genes(gtf_path):
    """
    Load gene annotations from gencode GTF file.
    
    Returns:
        dict mapping chromosome -> list of (start, end, gene_name, gene_type) tuples
    """
    print(f"Loading gene annotations from {gtf_path}...")
    genes_by_chrom = defaultdict(list)
    
    open_func = gzip.open if gtf_path.endswith('.gz') else open
    
    with open_func(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            
            feature_type = parts[2]
            if feature_type != 'gene':  # Only process gene features
                continue
            
            chrom = parts[0]
            start = int(parts[3])
            end = int(parts[4])
            attributes = parts[8]
            
            # Parse attributes to get gene_name
            gene_name = None
            gene_type = None
            for attr in attributes.split(';'):
                attr = attr.strip()
                if attr.startswith('gene_name'):
                    gene_name = attr.split('"')[1]
                elif attr.startswith('gene_type'):
                    gene_type = attr.split('"')[1]
            
            if gene_name:
                genes_by_chrom[chrom].append((start, end, gene_name, gene_type))
    
    # Sort genes by start position for efficient searching
    for chrom in genes_by_chrom:
        genes_by_chrom[chrom].sort(key=lambda x: x[0])
    
    total_genes = sum(len(genes) for genes in genes_by_chrom.values())
    print(f"✓ Loaded {total_genes:,} genes across {len(genes_by_chrom)} chromosomes")
    
    return genes_by_chrom


def find_closest_gene(chrom, start, end, genes_by_chrom):
    """
    Find the closest protein-coding gene to an interval.
    
    Excludes pseudogenes (e.g., Gm* genes) and only considers protein_coding genes.
    
    Returns:
        (gene_name, distance) tuple. Distance is 0 if overlapping, 
        otherwise distance to nearest gene boundary.
    """
    if chrom not in genes_by_chrom:
        return (None, None)
    
    genes = genes_by_chrom[chrom]
    interval_center = (start + end) // 2
    
    closest_gene = None
    min_distance = float('inf')
    
    for gene_start, gene_end, gene_name, gene_type in genes:
        # Skip non-protein-coding genes
        if gene_type != 'protein_coding':
            continue
        
        # Skip pseudogenes (Gm* genes)
        if gene_name.startswith('Gm'):
            continue
        
        # Check if interval overlaps gene
        if not (end < gene_start or start > gene_end):
            # Overlapping protein-coding gene found
            return (gene_name, 0)
        else:
            # Calculate distance to nearest boundary
            if end < gene_start:
                distance = gene_start - end
            else:  # start > gene_end
                distance = start - gene_end
            
            if distance < min_distance:
                min_distance = distance
                closest_gene = gene_name
    
    return (closest_gene, min_distance if min_distance != float('inf') else None)


def analyze_bedgraph(bedgraph_path, threshold=100.0, gtf_path=None):
    """
    Find intervals with values above threshold and report statistics.
    
    Args:
        bedgraph_path: Path to bedGraph file
        threshold: Value threshold to report (default: 100.0)
        gtf_path: Optional path to gencode GTF for gene annotation
    """
    # Load gene annotations if provided
    genes_by_chrom = None
    if gtf_path:
        genes_by_chrom = load_gencode_genes(gtf_path)
        print()
    
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
        # Annotate with gene names if GTF was provided
        if genes_by_chrom:
            print("Annotating intervals with closest genes...")
            for interval in large_intervals:
                gene_name, distance = find_closest_gene(
                    interval['chrom'], 
                    interval['start'], 
                    interval['end'],
                    genes_by_chrom
                )
                interval['gene'] = gene_name if gene_name else 'intergenic'
                interval['gene_distance'] = distance
            print("✓ Gene annotation complete\n")
        
        # Sort by value descending
        large_intervals.sort(key=lambda x: x['value'], reverse=True)
        
        print(f"\nTop {min(50, len(large_intervals))} intervals with value > {threshold}:")
        print("-" * 100)
        if genes_by_chrom:
            print(f"{'Chrom':<10} {'Start':>12} {'End':>12} {'Length':>10} {'Value':>12} {'Gene':<15} {'Distance':>10}")
            print("-" * 100)
            
            for interval in large_intervals[:50]:
                dist_str = f"{interval['gene_distance']:,}" if interval['gene_distance'] is not None else "overlap"
                print(f"{interval['chrom']:<10} {interval['start']:>12,} {interval['end']:>12,} "
                      f"{interval['length']:>10,} {interval['value']:>12.2f} {interval['gene']:<15} {dist_str:>10}")
        else:
            print(f"{'Chrom':<10} {'Start':>12} {'End':>12} {'Length':>10} {'Value':>12}")
            print("-" * 100)
            
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
        output_file = os.path.join('exports', os.path.basename(bedgraph_path).replace('.bedgraph', f'_large_values_gt{int(threshold)}.bed'))
        os.makedirs('exports', exist_ok=True)
        print(f"\n\nSaving all {len(large_intervals):,} intervals to: {output_file}")
        with open(output_file, 'w') as out:
            out.write(f"# Intervals with value > {threshold} from {bedgraph_path}\n")
            if genes_by_chrom:
                out.write(f"# chrom\tstart\tend\tvalue\tlength\tgene\tgene_distance\n")
                for interval in large_intervals:
                    dist_str = str(interval['gene_distance']) if interval['gene_distance'] is not None else "0"
                    out.write(f"{interval['chrom']}\t{interval['start']}\t{interval['end']}\t"
                             f"{interval['value']:.6f}\t{interval['length']}\t{interval['gene']}\t{dist_str}\n")
            else:
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
    parser.add_argument('--gtf', type=str, default=None,
                       help='Gencode GTF file for gene annotation (optional)')
    
    args = parser.parse_args()
    
    # Use default GTF path if not provided
    if args.gtf is None:
        default_gtf = '/home/ray/AG_Finetune/data/resources/gencode.vM38.annotation.gtf.gz'
        if os.path.exists(default_gtf):
            args.gtf = default_gtf
            print(f"Using default GTF: {default_gtf}\n")
    
    analyze_bedgraph(args.bedgraph, args.threshold, args.gtf)


if __name__ == '__main__':
    main()
