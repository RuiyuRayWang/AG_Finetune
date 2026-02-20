"""
Data loading utilities for CUT&Tag fine-tuning.

CUT&Tag uses Tn5 tagmentation (like ATAC-seq), so signal is maintained at
base-pair resolution (normalized insertion counts, e.g., scaled to 100M per track).

This module provides a PyTorch Dataset for:
1. Loading genomic sequences from reference FASTA
2. Loading CUT&Tag signal from BedGraph (base-pair resolution)
3. Data augmentation (following AlphaGenome training):
   - Shift augmentation: randomly shift windows by ±1024 bp
   - Reverse complementation: 50% probability (sequence + signal)

Use with PyTorch DataLoader for batching, shuffling, and parallel loading.
"""
import numpy as np
from typing import Tuple, Optional, Dict
import pyfaidx
from torch.utils.data import Dataset


class CutAndTagDataset(Dataset):
    """
    PyTorch Dataset for CUT&Tag fine-tuning.
    
    CUT&Tag uses Tn5 tagmentation, so data is treated like ATAC-seq:
    - Base-pair resolution insertion counts
    - Normalized to 100M insertions per track
    
    Expected workflow:
    1. Process FASTQ -> BAM using bowtie2/bwa
    2. BAM -> bedGraph using reads_to_bigwig.py (ChromBPNet) or similar
    3. Average replicates and normalize to 100M (process_cutandtag_bigwigs.py)
    4. Use with PyTorch DataLoader for batching and shuffling
    
    Example:
        dataset = CutAndTagDataset(fasta_path, signal_path, augment=True)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    """
    
    def __init__(
        self,
        fasta_path: str,
        signal_bedgraph: str,  # BedGraph file
        genome_windows: Optional[str] = None,  # BED file with training regions
        window_size: int = 262144,  # 2^18, can use smaller than 1Mb for fine-tuning
        organism_id: int = 0,  # 0 = human, 1 = mouse
        augment: bool = True,  # Enable data augmentation (shift + revcomp)
        max_shift: int = 1024,  # Maximum shift in bp for augmentation
    ):
        """
        Initialize dataset.
        
        Args:
            fasta_path: Path to reference genome FASTA
            signal_bedgraph: Path to CUT&Tag signal (BedGraph format, normalized)
            genome_windows: Path to BED file with regions to train on (optional)
            window_size: Sequence length in bp (default 262144 = 2^18).
                        AlphaGenome was pretrained on 2^20 (1Mb), but you can
                        use smaller windows for fine-tuning to reduce memory.
            organism_id: 0 for human, 1 for mouse
            augment: Enable data augmentation (True for training, False for validation)
            max_shift: Maximum shift in bp for shift augmentation (default 1024)
        
        Note: Batching is handled by PyTorch DataLoader, not this Dataset.
        """
        super().__init__()
        self.fasta = pyfaidx.Fasta(fasta_path)
        self.window_size = window_size
        self.organism_id = organism_id
        self.augment = augment
        self.max_shift = max_shift
        
        # Load signal data
        self.signal_data = self._load_signal(signal_bedgraph)
        
        # Load training regions
        if genome_windows:
            self.regions = self._load_regions(genome_windows)
        else:
            # Generate windows across genome
            self.regions = self._generate_windows()
            print(f"✓ Generated {len(self.regions)} training windows")

    def __len__(self) -> int:
        """Return number of training windows."""
        return len(self.regions)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, Dict[str, np.ndarray]]:
        """
        Get single training example with optional augmentation.
        
        Returns:
            (sequence, organism_id, targets) where:
                sequence: np.ndarray [window_size, 4], one-hot encoded DNA
                organism_id: int, organism identifier (0=human, 1=mouse)
                targets: dict with 'signal' key containing np.ndarray [window_size]
        """
        chrom, region_start, region_end = self.regions[idx]
        
        # Ensure region is exactly window_size
        region_len = region_end - region_start
        if region_len != self.window_size:
            # Adjust region to be exactly window_size
            region_end = region_start + self.window_size
        
        # Apply shift augmentation if enabled
        if self.augment:
            # Sample shift uniformly from [-max_shift, +max_shift]
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        else:
            shift = 0
        
        # Apply shift to window coordinates
        start = region_start + shift
        end = region_end + shift
        
        # Ensure we don't go out of chromosome bounds
        chrom_len = len(self.fasta[chrom])
        
        # If out of bounds, clip and adjust to maintain window_size
        if start < 0:
            start = 0
            end = self.window_size
        if end > chrom_len:
            end = chrom_len
            start = max(0, chrom_len - self.window_size)
        
        # Final safety check - ensure exactly window_size
        end = start + self.window_size
        if end > chrom_len:
            # Chromosome too short, pad will be needed
            end = chrom_len
        
        # Get sequence
        seq_str = str(self.fasta[chrom][start:end])
        seq_encoded = self._one_hot_encode(seq_str)
        
        # Get signal
        signal = self._get_signal_for_window(chrom, start, end)
        
        # Ensure exactly window_size (pad or truncate)
        current_len = len(seq_encoded)
        if current_len < self.window_size:
            # Pad to window_size
            pad_len = self.window_size - current_len
            pad_seq = np.zeros((pad_len, 4), dtype=np.float32)
            seq_encoded = np.vstack([seq_encoded, pad_seq])
            pad_signal = np.zeros(pad_len, dtype=np.float32)
            signal = np.concatenate([signal, pad_signal])
        elif current_len > self.window_size:
            # Truncate to window_size (shouldn't happen, but safety)
            seq_encoded = seq_encoded[:self.window_size]
            signal = signal[:self.window_size]
        
        # Apply reverse complementation with 50% probability if augmentation enabled
        if self.augment and np.random.rand() < 0.5:
            seq_encoded = self._reverse_complement_sequence(seq_encoded)
            signal = self._reverse_complement_signal(signal)
        
        # Final safety check: ensure no NaN/Inf in outputs
        # This catches any issues from augmentation or edge cases
        seq_encoded = np.where(np.isfinite(seq_encoded), seq_encoded, 0.25)
        signal = np.where(np.isfinite(signal), signal, 0.0)
        
        return (
            seq_encoded,  # np.ndarray [window_size, 4]
            self.organism_id,  # int
            {'signal': signal}  # dict with np.ndarray [window_size]
        )

    def _load_signal(self, bedgraph_path: str) -> Dict:
        """Load signal data from BedGraph into a dictionary for fast lookup."""
        import pandas as pd
        
        print(f"   Loading signal from {bedgraph_path}...")
        
        # Read BedGraph format: chr start end value
        df = pd.read_csv(
            bedgraph_path,
            sep='\t',
            header=None,
            names=['chrom', 'start', 'end', 'score'],
            comment='#'
        )
        
        # Check for NaN/Inf in ALL columns (not just score)
        initial_rows = len(df)
        
        # Remove rows with NaN in any column
        df = df.dropna()
        
        # Remove rows with Inf in numeric columns
        df = df[np.isfinite(df['start']) & np.isfinite(df['end']) & np.isfinite(df['score'])]
        
        # Ensure start/end are integers
        df['start'] = df['start'].astype(int)
        df['end'] = df['end'].astype(int)
        
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            print(f"   ⚠️  WARNING: Removed {rows_removed:,} rows containing NaN/Inf values")
            print(f"   Kept {len(df):,} valid rows")
        
        # Organize by chromosome for faster lookup
        signal_dict = {}
        for chrom in df['chrom'].unique():
            chrom_df = df[df['chrom'] == chrom].sort_values('start')
            signal_dict[chrom] = chrom_df
        
        print(f"   ✓ Loaded signal for {len(signal_dict)} chromosomes")
        return signal_dict
    
    def _load_regions(self, bed_path: str) -> list:
        """
        Load training regions from BED file and create windows.
        
        For peak-based training:
        - If peak < window_size: expand to window_size by centering
        - If peak >= window_size: split into non-overlapping windows
        
        This ensures every peak generates at least one training window.
        """
        import pandas as pd
        
        df = pd.read_csv(
            bed_path,
            sep='\t',
            header=None,
            names=['chrom', 'start', 'end'],
            comment='#'  # Skip comment lines
        )
        
        # Ensure start and end are integers
        df['start'] = df['start'].astype(int)
        df['end'] = df['end'].astype(int)
        
        windows = []
        total_bp = 0
        num_expanded = 0
        num_split = 0
        
        for _, row in df.iterrows():
            chrom = row['chrom']
            region_start = int(row['start'])
            region_end = int(row['end'])
            region_len = region_end - region_start
            total_bp += region_len
            
            # Get chromosome length for boundary checking
            if chrom in self.fasta.keys():
                chrom_len = len(self.fasta[chrom])
            else:
                continue  # Skip chromosomes not in reference
            
            if region_len < self.window_size:
                # Expand small peak to window_size by centering
                center = (region_start + region_end) // 2
                window_start = max(0, center - self.window_size // 2)
                window_end = window_start + self.window_size
                
                # Ensure within chromosome bounds
                if window_end > chrom_len:
                    window_end = chrom_len
                    window_start = max(0, window_end - self.window_size)
                
                windows.append((chrom, window_start, window_end))
                num_expanded += 1
            else:
                # Split large region into non-overlapping windows
                for start in range(region_start, region_end, self.window_size):
                    end = min(start + self.window_size, region_end, chrom_len)
                    if end - start >= self.window_size // 2:  # Only keep if at least half window size
                        windows.append((chrom, start, end))
                        num_split += 1
        
        print(f"   Loaded {len(df):,} regions ({total_bp:,} bp total)")
        print(f"   Expanded {num_expanded:,} small peaks to window_size")
        print(f"   Split {num_split:,} large regions into windows")
        print(f"   Total windows: {len(windows):,} of {self.window_size:,} bp each")
        
        return windows
    
    def _generate_windows(self) -> list:
        """Generate windows across the genome."""
        regions = []
        
        for chrom in self.fasta.keys():
            if chrom.startswith('chr'):
                chrom_len = len(self.fasta[chrom])
                
                # Generate non-overlapping windows
                for start in range(0, chrom_len - self.window_size, self.window_size):
                    end = start + self.window_size
                    regions.append((chrom, start, end))
        
        return regions
    
    def _one_hot_encode(self, sequence: str) -> np.ndarray:
        """
        One-hot encode DNA sequence.
        
        Returns: array of shape [seq_len, 4] with A=1000, C=0100, G=0010, T=0001
        """
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': -1}
        
        encoded = np.zeros((len(sequence), 4), dtype=np.float32)
        
        for i, base in enumerate(sequence.upper()):
            idx = mapping.get(base, -1)
            if idx >= 0:
                encoded[i, idx] = 1.0
            else:
                # N or unknown: set to 0.25 for all
                encoded[i, :] = 0.25
        
        return encoded
    
    def _reverse_complement_sequence(self, seq_encoded: np.ndarray) -> np.ndarray:
        """
        Reverse complement one-hot encoded sequence.
        
        Args:
            seq_encoded: One-hot encoded sequence [seq_len, 4] (A, C, G, T)
        
        Returns:
            Reverse complemented sequence [seq_len, 4]
        """
        # Reverse the sequence
        seq_rev = seq_encoded[::-1]
        # Complement: swap A<->T (0<->3) and C<->G (1<->2)
        seq_revcomp = seq_rev[:, [3, 2, 1, 0]]
        return seq_revcomp
    
    def _reverse_complement_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Reverse signal track.
        
        Args:
            signal: Signal array [seq_len]
        
        Returns:
            Reversed signal array [seq_len]
        """
        return signal[::-1]
    
    def _get_signal_for_window(self, chrom: str, start: int, end: int) -> np.ndarray:
        """
        Get CUT&Tag signal for a genomic window.
        
        Returns: array of shape [window_size] with signal values
        """
        # Initialize signal array
        signal = np.zeros(end - start, dtype=np.float32)
        
        # Get signal data for this chromosome
        if chrom not in self.signal_data:
            return signal  # Return zeros if no data for this chromosome
        
        chrom_signal = self.signal_data[chrom]
        
        # Find overlapping regions
        overlaps = chrom_signal[
            (chrom_signal['end'] > start) & (chrom_signal['start'] < end)
        ]
        
        # Fill in signal values
        for _, row in overlaps.iterrows():
            sig_start = max(0, row['start'] - start)
            sig_end = min(end - start, row['end'] - start)
            score = row['score']
            # Safety: replace NaN/Inf with 0
            if not np.isfinite(score):
                score = 0.0
            signal[sig_start:sig_end] = score
        
        # Final safety check: ensure no NaN/Inf in output
        signal = np.where(np.isfinite(signal), signal, 0.0)
        
        # Return raw signal (no log1p transform)
        # AlphaGenome/Borzoi loss handles scaling via targets_scaling function
        # which normalizes by track mean and applies smooth clipping
        return signal
    



def prepare_cut_and_tag_data(
    fastq_r1: str,
    fastq_r2: Optional[str],
    output_dir: str,
    genome_fasta: str,
    genome_index: str,
) -> str:
    """
    Helper function to process FASTQ files to BedGraph.
    
    This is a placeholder - you should use standard pipelines like:
    - bowtie2 or bwa for alignment
    - samtools for BAM processing  
    - ChromBPNet reads_to_bigwig.py for base-resolution signal
    
    Example pipeline:
    ```bash
    # Align with bowtie2
    bowtie2 --end-to-end --very-sensitive \
        --no-mixed --no-discordant \
        --phred33 -I 10 -X 700 \
        -p ${num_cores} \
        -x ${genome_index} \
        -1 ${fastq_r1} -2 ${fastq_r2} | \
        samtools sort -o aligned.bam
    
    # Index
    samtools index aligned.bam
    
    # Deduplicate (optional)
    samtools markdup -r aligned.bam deduped.bam
    samtools index deduped.bam
    
    # Generate base-resolution signal
    # Use ChromBPNet's reads_to_bigwig.py with +4/-4 shifts for Tn5
    python reads_to_bigwig.py \
        --bam deduped.bam \
        --output signal.bw \
        --chrom-sizes chrom.sizes \
        --smooth-width 0 \
        --plus-shift 4 \
        --minus-shift -4
    
    # Convert to bedGraph
    bigWigToBedGraph signal.bw signal.bedgraph
    
    # Average replicates and normalize (see process_cutandtag_bigwigs.py)
    ```
    
    Args:
        fastq_r1: Read 1 FASTQ file
        fastq_r2: Read 2 FASTQ file (optional for single-end)
        output_dir: Output directory
        genome_fasta: Reference genome FASTA
        genome_index: Bowtie2/BWA index prefix
        
    Returns:
        Path to output BedGraph file
    """
    raise NotImplementedError(
        "Please process FASTQ files using standard genomics pipelines. "
        "See function docstring for example commands."
    )
