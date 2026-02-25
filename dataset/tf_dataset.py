import numpy as np
import pandas as pd
import pyBigWig

class BigWigExtractor:
    def __init__(self, bw_path, chrom_sizes_path):
        self.bw = pyBigWig.open(bw_path)
        # If chrom_sizes_path is provided, use it; otherwise get from BigWig file
        if chrom_sizes_path and chrom_sizes_path is not None:
            self.chrom_sizes = pd.read_csv(chrom_sizes_path, sep="\t", header=None, names=["chrom", "size"])
            self.chroms = self.chrom_sizes["chrom"].tolist()
            self.sizes = dict(zip(self.chrom_sizes["chrom"], self.chrom_sizes["size"]))
        else:
            # Extract chromosome info directly from BigWig
            self.sizes = self.bw.chroms()
            self.chroms = list(self.sizes.keys())

    def get_signal(self, chrom, start, end):
        """Extract signal from BigWig, returning zeros for missing data."""
        if chrom not in self.sizes:
            return np.zeros(end - start, dtype=np.float32)
        try:
            values = np.array(self.bw.values(chrom, start, end), dtype=np.float32)
            # Handle NaN values
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            return values
        except Exception as e:
            print(f"Warning: Failed to extract BigWig signal for {chrom}:{start}-{end}: {e}")
            return np.zeros(end - start, dtype=np.float32)

    def close(self):
        self.bw.close()

class BedGraphExtractor:
    def __init__(self, bedgraph_path, chrom_sizes_path):
        if not chrom_sizes_path:
            raise ValueError("chrom_sizes_path is required for BedGraph format")
        self.chrom_sizes = pd.read_csv(chrom_sizes_path, sep="\t", header=None, names=["chrom", "size"])
        self.chroms = self.chrom_sizes["chrom"].tolist()
        self.sizes = dict(zip(self.chrom_sizes["chrom"], self.chrom_sizes["size"]))
        self.signal = self._load_bedgraph(bedgraph_path)

    def _load_bedgraph(self, path):
        signal = {chrom: np.zeros(size, dtype=np.float32) for chrom, size in self.sizes.items()}
        with open(path) as f:
            for line in f:
                if line.startswith("track") or line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                chrom, start, end, value = parts[0], int(parts[1]), int(parts[2]), float(parts[3])
                if chrom in signal:
                    signal[chrom][start:end] = value
        return signal

    def get_signal(self, chrom, start, end):
        """Extract signal from pre-loaded BedGraph data."""
        if chrom not in self.signal:
            return np.zeros(end - start, dtype=np.float32)
        return self.signal[chrom][start:end]

class CutAndTagDataPipelineTF:
    def __init__(self, signal_path, chrom_sizes_path):
        """
        Initialize signal extractor for either BigWig or BedGraph format.
        
        Args:
            signal_path: Path to signal file (.bigwig, .bw, .bedgraph, or .bed)
            chrom_sizes_path: Path to chrom.sizes file (required for BedGraph, optional for BigWig)
        
        Raises:
            ValueError: If signal format is unsupported or required args are missing
        """
        if signal_path.endswith(".bigwig") or signal_path.endswith(".bw"):
            self.extractor = BigWigExtractor(signal_path, chrom_sizes_path)
        elif signal_path.endswith(".bedgraph") or signal_path.endswith(".bed"):
            if not chrom_sizes_path:
                raise ValueError(
                    f"BedGraph format requires --chrom-sizes argument. "
                    f"Please provide a chrom.sizes file for {signal_path}"
                )
            self.extractor = BedGraphExtractor(signal_path, chrom_sizes_path)
        else:
            raise ValueError(
                f"Unsupported signal format: {signal_path}\n"
                f"Supported formats: .bigwig, .bw (BigWig), .bedgraph, .bed (BedGraph)"
            )

    def get_signal(self, chrom, start, end):
        return self.extractor.get_signal(chrom, start, end)

    def close(self):
        if hasattr(self.extractor, "close"):
            self.extractor.close()
