"""
average_normalize_bigwigs.py — Average CUT&Tag BigWig replicates and normalise to 100 M insertions.

Workflow:
  1. Groups input BigWig files by cell type × epitope from a TSV manifest.
  2. Averages replicates at base-pair resolution.
  3. Scales the averaged signal so the genome-wide total equals 100 M insertions
     (matching the AlphaGenome pre-training normalisation).
  4. Writes one output BigWig per group.

Usage:
    python scripts/average_normalize_bigwigs.py \\
        --config samples.tsv \\
        --output-dir data/CUTTag/processed \\
        --chrom-sizes data/resources/mm10.chrom.sizes

samples.tsv columns: sample_id, cell_type, epitope, bigwig_path

Input BigWigs must already have the +4/-4 Tn5 shift applied
(e.g. produced by chrombpnet reads_to_bigwig.py).
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import pyBigWig
from collections import defaultdict
from pathlib import Path


def load_chrom_sizes(chrom_sizes_file: str) -> list[tuple[str, int]]:
    """Return list of (chrom, size) tuples in file order (for BigWig header)."""
    sizes = []
    with open(chrom_sizes_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            sizes.append((parts[0], int(parts[1])))
    return sizes


def average_and_normalize_bigwigs(
    bw_files: list[str],
    chrom_sizes: list[tuple[str, int]],
    output_path: str,
    normalize_to: float = 100_000_000,
):
    """
    Average base-resolution signal across replicates, normalize, and write BigWig.

    Args:
        bw_files:      List of input BigWig paths.
        chrom_sizes:   List of (chrom, size) in the desired output order.
        output_path:   Path for the output BigWig.
        normalize_to:  Target total insertion count (default 100 M).
    """
    n = len(bw_files)
    print(f"  Replicates : {n}")

    # Open all input BigWigs
    readers = [pyBigWig.open(p) for p in bw_files]

    chrom_dict = dict(chrom_sizes)
    canonical = [(c, s) for c, s in chrom_sizes
                 if c in chrom_dict and any(c in r.chroms() for r in readers)]

    # First pass: compute pre-normalization total across all chroms
    grand_total = 0.0
    for chrom, size in canonical:
        chrom_sum = 0.0
        for r in readers:
            if chrom not in r.chroms():
                continue
            vals = r.values(chrom, 0, size, numpy=True)
            if vals is not None:
                np.nan_to_num(vals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                chrom_sum += float(vals.sum())
        grand_total += chrom_sum / n  # average across replicates

    print(f"  Pre-norm total : {grand_total:,.0f}")
    if grand_total == 0:
        print("  WARNING: total signal is 0 — output will be all zeros")
        scale = 1.0
    else:
        scale = normalize_to / grand_total
    print(f"  Scale factor   : {scale:.6f}")

    # Second pass: write averaged + scaled signal to output BigWig
    bw_out = pyBigWig.open(output_path, 'w')
    bw_out.addHeader(canonical)

    for chrom, size in canonical:
        avg = np.zeros(size, dtype=np.float64)
        for r in readers:
            if chrom not in r.chroms():
                continue
            vals = r.values(chrom, 0, size, numpy=True)
            if vals is not None:
                np.nan_to_num(vals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                avg += vals
        avg /= n
        avg *= scale

        # Write as a base-resolution array (span=1, step=1)
        bw_out.addEntries(
            chrom, 0,
            values=avg.astype(np.float32),
            span=1, step=1,
        )

    bw_out.close()
    for r in readers:
        r.close()

    # Verify written total
    bw_check = pyBigWig.open(output_path)
    written_total = sum(
        float(np.nan_to_num(
            bw_check.values(c, 0, s, numpy=True), nan=0.0
        ).sum())
        for c, s in canonical
        if c in bw_check.chroms()
    )
    bw_check.close()
    print(f"  Written total  : {written_total:,.0f}  (target {normalize_to:,.0f})")
    print(f"  ✓ Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Average CUT&Tag BigWig replicates and normalize to 100 M insertions'
    )
    parser.add_argument('--config', required=True,
                        help='TSV with columns: sample_id, cell_type, epitope, bigwig_path')
    parser.add_argument('--output-dir', required=True,
                        help='Directory for output BigWig files')
    parser.add_argument('--chrom-sizes', required=True,
                        help='Chromosome sizes file (e.g. mm10.chrom.sizes)')
    parser.add_argument('--normalize-to', type=float, default=100_000_000,
                        help='Normalize to this many total insertions (default: 100 M)')
    args = parser.parse_args()

    # Load sample config
    print(f"Loading sample config: {args.config}")
    df = pd.read_csv(args.config, sep='\t')

    # Accept bigwig_path or legacy bedgraph_path column
    if 'bigwig_path' in df.columns:
        path_col = 'bigwig_path'
    elif 'bedgraph_path' in df.columns:
        path_col = 'bedgraph_path'
        print("  Note: 'bedgraph_path' column found — treating values as BigWig paths")
    else:
        sys.exit("Error: config TSV must have a 'bigwig_path' column")

    required = ['sample_id', 'cell_type', 'epitope', path_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"Error: missing columns in config: {missing}")

    print(f"  {len(df)} samples loaded")

    # Load chrom sizes
    print(f"\nLoading chrom sizes: {args.chrom_sizes}")
    chrom_sizes = load_chrom_sizes(args.chrom_sizes)
    print(f"  {len(chrom_sizes)} chromosomes")

    os.makedirs(args.output_dir, exist_ok=True)

    # Group by cell_type × epitope
    groups = defaultdict(list)
    for _, row in df.iterrows():
        groups[(row['cell_type'], row['epitope'])].append(row[path_col])

    print(f"\n{len(groups)} group(s) to process:")
    for (ct, ep), files in groups.items():
        print(f"  {ct} / {ep}  ({len(files)} replicate(s))")

    print("\n" + "=" * 60)
    print("PROCESSING")
    print("=" * 60)

    for (cell_type, epitope), bw_files in groups.items():
        print(f"\n{cell_type} / {epitope}")
        print("-" * 60)
        output_name = f"{cell_type}_{epitope}_averaged_normalized.bw"
        output_path = os.path.join(args.output_dir, output_name)
        average_and_normalize_bigwigs(
            bw_files, chrom_sizes, output_path, args.normalize_to
        )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
