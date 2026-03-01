"""
Fine-tuning script for AlphaGenome on CUT&Tag data using TensorFlow data loading.

This script follows the DeepMind finetuning approach from their official colab:
- TensorFlow (tf.data) for data loading (CPU only, JAX handles training)
- Support for both BigWig and BedGraph signal formats (auto-detected)
- bfloat16 for data storage (memory efficient)
- float32 for model computation (numerically stable)

Two strategies are supported (select with --strategy):

  head_replace (default)
    DeepMind-style full fine-tuning: replaces output heads with new task-specific
    heads (1 ATAC track for CUT&Tag) and fine-tunes ALL parameters jointly.
    Mirrors the official AlphaGenome fine-tuning colab exactly.

  lora
    Parameter-efficient fine-tuning: LoRA adapters on Q/K/V only (~428K params).
    Keeps pretrained heads; supervised against pretrained ATAC head.

Multi-GPU Support (8 GPUs tested):
  - JAX data parallelism with parameter replication  
  - Explicit sharding specifications in jax.jit() (prevents NCCL errors)
  - Validation statistics use jax.device_get() to avoid NCCL AllReduce

Usage:
    # Using config file (recommended):
    python tf_ft.py --config configs/config_bigwig.yaml
    
    # Config file with CLI overrides:
    python tf_ft.py --config configs/config_bigwig.yaml --batch-size 16 --lr 1e-4

    # Direct CLI arguments:
    python tf_ft.py --strategy head_replace \
        --checkpoint /path/to/alphagenome-fold-0 \
        --fasta /path/to/genome.fa \
        --signal /path/to/signal.bigwig \
        --output ./checkpoints/fine_tuned \
        --windows-bed /path/to/safe_windows.bed

Note: Signal files can be either BigWig (.bigwig, .bw) or BedGraph (.bedgraph, .bed) format.
      The script will auto-detect the format based on file extension.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Iterator, Any, Mapping
from functools import partial
import itertools
import yaml
import shutil
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Suppress noisy C++ (glog) rendezvous warnings ─────────────────────────
# The "acquire clique" E-level messages from rendezvous.cc are false positives:
# the thread self-resolves and prints "Thread is unstuck!" immediately after.
# These fire during normal compute time (the 28s/step is real forward+backward
# on 403M params, not a stall). GLOG levels: 0=INFO 1=WARNING 2=ERROR 3=FATAL.
os.environ.setdefault('GLOG_minloglevel', '3')
# ────────────────────────────────────────────────────────────────────────────

# TensorFlow: CPU only (for data loading), JAX handles training
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.config.set_visible_devices([], 'TPU')

import jax
jax.config.update('jax_compilation_cache_dir',
                  '/home/ray/AG_Finetune/.jax_cache')
jax.config.update('jax_persistent_cache_min_compile_time_secs', 5)
import jax.numpy as jnp
import haiku as hk
import jmp
import optax
import pandas as pd
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
from alphagenome_research.model import dna_model, model as alphagenome_model
from alphagenome_research.model.metadata import metadata as metadata_lib
from alphagenome_research.io import fasta as fasta_lib
from alphagenome_research.model import one_hot_encoder
from models.model import LoRAAlphaGenomeModel, HeadReplaceAlphaGenomeModel
from utils.lora_utils import get_attention_qkv_layers
from utils.borzoi_loss import cut_and_tag_loss_baskerville
from dataset.tf_dataset import CutAndTagDataPipelineTF
import json
import csv
import datetime
import contextlib
import sys as _sys


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config if config is not None else {}


def backup_config(config_path: str, output_dir: str) -> str:
    """
    Backup config file to output directory.
    
    Args:
        config_path: Path to original config file
        output_dir: Output directory for backup
    
    Returns:
        Path to backed up config file
    """
    os.makedirs(output_dir, exist_ok=True)
    config_filename = os.path.basename(config_path)
    backup_path = os.path.join(output_dir, config_filename)
    shutil.copy2(config_path, backup_path)
    return backup_path


class _Tee:
    """Mirror writes to both the original stream and a file."""
    def __init__(self, stream, filepath):
        self._stream = stream
        self._file = open(filepath, 'a', buffering=1)  # line-buffered

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    # Proxy everything else (isatty, fileno, …) to the original stream
    def __getattr__(self, name):
        return getattr(self._stream, name)


class CutAndTagDataPipeline:
    """TensorFlow data pipeline for CUT&Tag fine-tuning (supports BigWig and BedGraph)."""
    
    def __init__(
        self,
        fasta_path: str,
        signal_path: str,
        windows_bed: str,
        window_size: int,
        organism_id: int,
        chrom_sizes_path: str = None,
    ):
        self.fasta_extractor = fasta_lib.FastaExtractor(fasta_path)
        self.signal_extractor = CutAndTagDataPipelineTF(signal_path, chrom_sizes_path)
        self.window_size = window_size
        self.organism_id = organism_id
        self.one_hot_encoder = one_hot_encoder.DNAOneHotEncoder()
        
        # Load or generate training windows
        if windows_bed is not None:
            print(f"   Loading training windows from {windows_bed}...")
            self.intervals = pd.read_csv(
                windows_bed,
                sep='\t',
                header=None,
                names=['chrom', 'start', 'end'],
                comment='#'
            )
            self.intervals['start'] = self.intervals['start'].astype(int)
            self.intervals['end'] = self.intervals['end'].astype(int)
            print(f"   ✓ Loaded {len(self.intervals)} training windows")
        else:
            if chrom_sizes_path is None:
                raise ValueError(
                    "Either windows_bed or chrom_sizes_path must be provided "
                    "when windows_bed is None."
                )
            print(f"   No windows_bed provided — generating whole-genome windows "
                  f"from {chrom_sizes_path}...")
            self.intervals = generate_genome_windows(chrom_sizes_path, window_size)
            print(f"   ✓ Generated {len(self.intervals):,} whole-genome windows "
                  f"({window_size:,} bp, non-overlapping)")
    
    def get_element(self, idx: int) -> Dict[str, Any]:
        """Get a single training example."""
        row = self.intervals.iloc[idx]
        chrom = row['chrom']
        start = int(row['start'])
        end = int(row['end'])
        
        # Resize interval to exact window_size if needed
        if (end - start) != self.window_size:
            center = (start + end) // 2
            start = center - self.window_size // 2
            end = start + self.window_size
        
        # Extract sequence using DeepMind's FastaExtractor
        try:
            from alphagenome.data import genome
            interval = genome.Interval(chromosome=chrom, start=start, end=end)
            seq_str = self.fasta_extractor.extract(interval)
            seq_one_hot = self.one_hot_encoder.encode(seq_str)
        except Exception as e:
            print(f"Warning: Failed to extract sequence for {chrom}:{start}-{end}: {e}")
            seq_one_hot = np.zeros((self.window_size, 4), dtype=np.float32)
        
        # Extract signal (BigWig or BedGraph)
        try:
            signal = self.signal_extractor.get_signal(chrom, start, end)
            # Handle potential NaN or length mismatch
            if signal is None or len(signal) != self.window_size:
                signal = np.zeros(self.window_size, dtype=np.float32)
            else:
                signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        except Exception as e:
            print(f"Warning: Failed to extract signal for {chrom}:{start}-{end}: {e}")
            signal = np.zeros(self.window_size, dtype=np.float32)
        
        return {
            'dna_sequence': seq_one_hot,
            'organism_index': self.organism_id,
            'signal': signal,
        }
    
    def get_output_signature(self):
        """Return TensorFlow output signature."""
        return {
            'dna_sequence': tf.TensorSpec(shape=(self.window_size, 4), dtype=tf.float32),
            'organism_index': tf.TensorSpec(shape=[], dtype=tf.int32),
            'signal': tf.TensorSpec(shape=(self.window_size,), dtype=tf.float32),
        }
    
    def get_generator(self, shuffle: bool = True, seed: int = 0) -> Iterator[Dict[str, Any]]:
        """Generate training examples."""
        rng = np.random.default_rng(seed=seed)
        intervals = self.intervals.copy()
        
        while True:  # Infinite generator
            if shuffle:
                intervals = intervals.sample(frac=1, random_state=rng).reset_index(drop=True)
            
            for idx in range(len(intervals)):
                try:
                    yield self.get_element(idx)
                except Exception as e:
                    print(f"Warning: Failed to get element {idx}: {e}")
                    continue
    
    def close(self):
        self.signal_extractor.close()


def generate_genome_windows(
    chrom_sizes_path: str,
    window_size: int,
    stride: int = None,
) -> pd.DataFrame:
    """
    Generate non-overlapping windows across all canonical chromosomes.

    Only keeps chromosomes matching chr[0-9XY]+ (skips alt/random/chrM contigs)
    and large enough to contain at least one full window.

    Args:
        chrom_sizes_path: Path to a 2-column TSV (chrom\tsize).
        window_size: Window length in bp.
        stride: Step between window starts. Defaults to window_size (non-overlapping).

    Returns:
        DataFrame with columns ['chrom', 'start', 'end'].
    """
    if stride is None:
        stride = window_size
    chrom_sizes = pd.read_csv(
        chrom_sizes_path, sep='\t', header=None, names=['chrom', 'size']
    )
    canonical = chrom_sizes[
        chrom_sizes['chrom'].str.match(r'^chr[0-9XY]+$') &
        (chrom_sizes['size'] >= window_size)
    ]
    rows = []
    for _, row in canonical.iterrows():
        chrom, size = row['chrom'], int(row['size'])
        for start in range(0, size - window_size + 1, stride):
            rows.append({'chrom': chrom, 'start': start, 'end': start + window_size})
    return pd.DataFrame(rows)


def pearson_correlation(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    pred_mean = jnp.mean(predictions)
    target_mean = jnp.mean(targets)
    
    numerator = jnp.sum((predictions - pred_mean) * (targets - target_mean))
    denominator = jnp.sqrt(
        jnp.sum((predictions - pred_mean) ** 2) * 
        jnp.sum((targets - target_mean) ** 2)
    )
    
    return numerator / (denominator + 1e-8)


def train(
    checkpoint_path: str,
    fasta_path: str,
    signal_path: str,
    output_prefix: str,
    strategy: str = 'head_replace',
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    window_size: int = 131072,
    organism_id: int = 1,
    genome_windows: str = None,
    augment: bool = True,
    max_shift: int = 1024,
    num_workers: int = 0,
    precision_policy: str = 'params=float32,compute=float32,output=float32',
    config_path: str = None,
    chrom_sizes_path: str = None,
    checkpoint_interval: int = 500,
):
    """
    Fine-tune AlphaGenome on CUT&Tag data using TensorFlow data loading.
    
    This follows the official DeepMind finetuning approach with:
    - TensorFlow tf.data pipeline for efficient data loading
    - Support for both BigWig and BedGraph signal formats
    - float32 for model parameters (stable)
    - Requires pre-defined BED file of training windows

    Args:
        strategy: 'head_replace' (full fine-tuning with new ATAC head, recommended)
                  or 'lora' (LoRA adapters on Q/K/V, parameter-efficient alternative).
        precision_policy: JMP policy string. Default is full float32 (safe).
        checkpoint_path: Path to pre-trained AlphaGenome checkpoint.
        fasta_path: Path to reference genome FASTA.
        signal_path: Path to CUT&Tag signal file (BigWig or BedGraph, auto-detected).
        output_dir: Directory for checkpoints.
        learning_rate: Peak Adam LR (default 5e-4 for head_replace, 1e-5 for lora).
        lora_rank: LoRA rank (lora strategy only).
        lora_alpha: LoRA alpha (lora strategy only).
        window_size: bp per window. Must be divisible by 128.
        organism_id: 0=human, 1=mouse.
        genome_windows: REQUIRED path to BED file with training windows.
        chrom_sizes_path: Path to chrom.sizes file (required for BedGraph, optional for BigWig).
        config_path: Path to original config file (for backup purposes).
    """

    # ---- logging setup ---------------------------------------------------
    _ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    output_dir = f"{output_prefix}_{strategy}_{_ts}"
    os.makedirs(output_dir, exist_ok=True)
    
    log_path     = os.path.join(output_dir, 'train.log')
    metrics_path = os.path.join(output_dir, 'metrics.csv')
    _tee = _Tee(_sys.stdout, log_path)
    _sys.stdout = _tee
    print(f"[log] stdout → {log_path}")
    print(f"[log] metrics → {metrics_path}")
    
    # Backup config file if provided
    if config_path and os.path.exists(config_path):
        backup_path = backup_config(config_path, output_dir)
        print(f"[config] backed up → {backup_path}")
    _metrics_file   = open(metrics_path, 'a', newline='', buffering=1)
    _metrics_writer = csv.writer(_metrics_file)
    if os.path.getsize(metrics_path) == 0:
        _metrics_writer.writerow(['timestamp', 'epoch', 'batch', 'loss', 'grad_norm', 'nan_event', 'skipped'])

    # Redirect C++ stderr (rendezvous.cc false-positive warnings) to a file.
    # glog reads GLOG_minloglevel at C++ static-init time, before Python runs,
    # so os.environ has no effect. File-descriptor-level redirect is the only
    # reliable way to suppress these from the terminal.
    _stderr_log_path = os.path.join(output_dir, 'stderr.log')
    _stderr_log_fd   = os.open(_stderr_log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    _saved_stderr_fd = os.dup(2)
    os.dup2(_stderr_log_fd, 2)
    os.close(_stderr_log_fd)
    print(f"[log] stderr  → {_stderr_log_path}")
    # ----------------------------------------------------------------------

    num_devices = jax.device_count()
    
    if num_devices > 1:
        print("="*60)
        print(f"MULTI-GPU DETECTED: {num_devices} devices")
        print("="*60)
        print(f"Devices: {jax.devices()}")
        print(f"Using JAX data parallelism with parameter replication")
        print()
        
        # Create device mesh for data parallelism (official DeepMind approach)
        devices = mesh_utils.create_device_mesh((num_devices,))
        mesh = Mesh(devices, axis_names=('data',))
        # Use bare PartitionSpec — must be used inside jax.set_mesh() context
        data_sharding = P('data')
        replicated_sharding = P()
    else:
        print("Single GPU mode")
        mesh = None
        data_sharding = None
        replicated_sharding = None
    
    # Validate window size
    if window_size % 128 != 0:
        raise ValueError(f"window_size must be divisible by 128, got {window_size}")
    
    print("="*60)
    print(f"FINE-TUNING SETUP  [strategy={strategy}]")
    print("="*60)
    print(f"Precision policy:  {precision_policy}")
    print(f"Window size: {window_size:,} bp (2^{int(np.log2(window_size))})")
    if window_size < 1048576:
        print(f"  Note: Using smaller windows than pretraining (1Mb)")
        print(f"        This reduces memory but may limit long-range context")
    print(f"Augmentation: {'enabled' if augment else 'disabled'}")
    if augment:
        print(f"  - Shift augmentation: ±{max_shift} bp")
        print(f"  - Reverse complement: 50% probability")
        print(f"  WARNING: TensorFlow pipeline does not support augmentation yet")
    
    # Convert paths to absolute
    checkpoint_path = os.path.abspath(checkpoint_path)
    fasta_path      = os.path.abspath(fasta_path)
    signal_path     = os.path.abspath(signal_path)
    output_dir      = os.path.abspath(output_dir)
    
    # Detect signal format
    signal_format = "BigWig" if signal_path.endswith(('.bigwig', '.bw')) else "BedGraph"
    print(f"Signal format: {signal_format} (auto-detected from extension)")
    
    # Resolve genome_windows path if provided
    if genome_windows:
        genome_windows = os.path.abspath(genome_windows)
    else:
        print("   Note: windows_bed not set — whole-genome windows will be generated from chrom_sizes")

    # ------------------------------------------------------------------
    # Data pipeline — TensorFlow dataset (following DeepMind approach)
    # ------------------------------------------------------------------
    print(f"\n1. Setting up TensorFlow data pipeline...")
    pipeline = CutAndTagDataPipeline(
        fasta_path=fasta_path,
        signal_path=signal_path,
        windows_bed=genome_windows,
        window_size=window_size,
        organism_id=organism_id,
        chrom_sizes_path=chrom_sizes_path,
    )
    
    # Create TensorFlow dataset
    tf_dataset = tf.data.Dataset.from_generator(
        pipeline.get_generator,
        output_signature=pipeline.get_output_signature(),
    )
    tf_dataset = (
        tf_dataset
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    
    # Get dataset size for scheduling
    num_windows = len(pipeline.intervals)
    steps_per_epoch = num_windows // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(0.05 * total_steps)  # 5% warmup
    print(f"   TF dataset: {num_windows} windows, {steps_per_epoch} steps/epoch")
    print(f"   LR schedule: {warmup_steps} warmup, {total_steps} total steps")

    # Load base model
    print(f"\n2. Loading base model from {checkpoint_path}...")
    base_model = dna_model.create(
        checkpoint_path=checkpoint_path,
        organism_settings=None,
        device=None
    )
    print(f"   ✓ Model loaded on {base_model._device_context._device}")

    # ------------------------------------------------------------------
    # Model construction (outside mesh context — no device_put happens here).
    # params/state/opt_state stay as plain arrays; they are materialised as
    # replicated by in_shardings=P() in jax.jit on the first training call.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Strategy: head_replace  (DeepMind official approach)
    # ------------------------------------------------------------------
    mesh_ctx = jax.set_mesh(mesh) if mesh is not None else contextlib.nullcontext()
    if strategy == 'head_replace':
        print(f"\n3. [head_replace] Building new ATAC output head for CUT&Tag...")
        organism_enum = (
            dna_model.Organism.HOMO_SAPIENS if organism_id == 0
            else dna_model.Organism.MUS_MUSCULUS
        )
        track_df = pd.DataFrame([{
            'output_type': 'ATAC',
            'name': 'CUT&Tag H3K27ac',
            'strand': '.',
            'file_path': signal_path,
        }])
        new_metadata = metadata_lib.AlphaGenomeOutputMetadata(atac=track_df)
        new_output_metadata = {organism_enum: new_metadata}

        jmp_policy_obj = jmp.get_policy(precision_policy)

        @hk.transform_with_state
        def _forward_new(dna_sequence, organism_index):
            with hk.mixed_precision.push_policy(alphagenome_model.AlphaGenome, jmp_policy_obj):
                return alphagenome_model.AlphaGenome(
                    new_output_metadata,
                    num_splice_sites=512,
                    splice_site_threshold=0.1,
                )(dna_sequence, organism_index)

        def new_apply_fn(params, state, dna_sequence, organism_index):
            (predictions, _), _ = _forward_new.apply(
                params, state, None, dna_sequence, organism_index
            )
            return predictions

        print(f"   Initialising new head params (dummy forward pass, no JIT)...")
        dummy_seq = jnp.zeros((1, window_size, 4), dtype=jnp.float32)
        dummy_org = jnp.zeros((1,), dtype=jnp.int32)
        new_params, _new_state = _forward_new.init(
            jax.random.PRNGKey(0), dummy_seq, dummy_org
        )

        # Weight surgery: trunk from pretrained, heads freshly initialised
        new_head_params = hk.data_structures.filter(
            lambda module_name, *_: 'head' in module_name, new_params
        )
        base_trunk_params = hk.data_structures.filter(
            lambda module_name, *_: 'head' not in module_name, base_model._params
        )
        merged_params = hk.data_structures.merge(base_trunk_params, new_head_params)
        print(f"   ✓ Weight surgery complete — trunk from pretrained, head freshly init'd")

        model = HeadReplaceAlphaGenomeModel(
            params=merged_params,
            state=base_model._state,
            apply_fn=new_apply_fn,
            mesh=mesh,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

    # ------------------------------------------------------------------
    # Strategy: lora  (parameter-efficient alternative)
    # ------------------------------------------------------------------
    elif strategy == 'lora':
        print(f"\n3. [lora] Identifying LoRA target layers...")
        target_layers = get_attention_qkv_layers()
        print(f"   ✓ Targeting {len(target_layers)} Q/K/V layers")

        model = LoRAAlphaGenomeModel(
            base_model=base_model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            target_layers=target_layers,
            learning_rate=learning_rate,
            mesh=mesh,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            precision_policy=precision_policy,
        )
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'head_replace' or 'lora'.")

    # ------------------------------------------------------------------
    # Training loop  (shared between both strategies)
    # ------------------------------------------------------------------
    # IMPORTANT: When using multi-GPU with sharded arrays, global reductions
    # (jnp.sum, jnp.max, jnp.mean) trigger NCCL AllReduce operations which
    # can fail with "invalid argument" errors. For validation/logging purposes,
    # use jax.device_get() to move data to host BEFORE computing statistics.
    # This bypasses NCCL and works reliably.
    # Element-wise operations and the loss function don't need device_get.
    # ------------------------------------------------------------------
    print(f"\n4. Starting training for {epochs} epochs  [strategy={strategy}]...")
    loss_fn = cut_and_tag_loss_baskerville
    best_loss = float('inf')

    # Wrap training loop in mesh context (CRITICAL for multi-GPU).
    # jax.set_mesh() provides the axis context for bare P()/P('data') specs
    # used in jax.device_put (batch sharding) and jax.jit in_shardings/out_shardings
    # (param replication). Matches the official DeepMind colab pattern exactly:
    #   with jax.set_mesh(mesh):
    #       batch = jax.device_put(batch, data_sharding)
    #       params, state, opt_state, scalars = train_step(params, state, opt_state, batch)
    with mesh_ctx:
        for epoch in range(epochs):
            epoch_losses = []
            grad_norms = []

            print(f"\nEpoch {epoch + 1}/{epochs}  [strategy={strategy}]")
            print("-" * 60)

            # Create numpy iterator from TensorFlow dataset.
            # NOTE: get_generator() is infinite (while True loop + reshuffle),
            # so we MUST cap iteration at steps_per_epoch via islice.
            ds_iter = tf_dataset.as_numpy_iterator()
            
            for batch_idx, batch in enumerate(itertools.islice(ds_iter, steps_per_epoch)):
                # Progress indicator for first batch (JAX compilation happens here)
                if batch_idx == 0:
                    print(f"  Compiling JAX functions (this may take 2-5 minutes on first batch)...")
                    start_time = time.time()
                
                # Convert TensorFlow batch to JAX arrays
                sequences = jnp.array(batch['dna_sequence'], dtype=jnp.float32)
                organism_ids = jnp.array(batch['organism_index'], dtype=jnp.int32)
                targets = {'signal': jnp.array(batch['signal'], dtype=jnp.float32)}
                
                # Shard batch across devices (inside jax.set_mesh() context)
                if data_sharding is not None:
                    sequences = jax.device_put(sequences, data_sharding)
                    organism_ids = jax.device_put(organism_ids, data_sharding)
                    targets = {
                        'signal': jax.device_put(targets['signal'], data_sharding)
                    }
                
                # Training step
                loss_value, grad_norm = model.train_step(
                    sequences, organism_ids, targets, loss_fn
                )
                
                # Convert to host scalars for logging
                loss_value_host = float(jax.device_get(loss_value))
                grad_norm_host = float(jax.device_get(grad_norm))
                
                epoch_losses.append(loss_value_host)
                grad_norms.append(grad_norm_host)
                _metrics_writer.writerow([datetime.datetime.now().isoformat(), epoch + 1, batch_idx + 1, loss_value_host, grad_norm_host, False, False])
                
                # Log progress
                if batch_idx == 0:
                    compile_time = time.time() - start_time
                    print(f"  ✓ Compilation complete! Took {compile_time:.1f}s")
                    
                    # Visualize array sharding for debugging (multi-GPU only)
                    if data_sharding is not None:
                        print(f"\n  Array Sharding Visualization:")
                        print(f"  Sequences shape: {sequences.shape}")
                        print(f"  Sequences sharding: {sequences.sharding}")
                        # Visualize a 2D slice since visualize_array_sharding only supports 1D/2D
                        try:
                            print(f"  Sequences[:, 0, 0] (batch dimension):")
                            jax.debug.visualize_array_sharding(sequences[:, 0, 0])
                        except Exception as e:
                            print(f"  (Visualization skipped: {e})")
                        print(f"  Loss value: {loss_value}")
                        print()
                    
                    print(f"  Training on {steps_per_epoch} batches...")
                
                # Print progress every 10 batches (or every batch if < 20 total batches)
                print_interval = 1 if steps_per_epoch < 20 else 10
                if (batch_idx + 1) % print_interval == 0:
                    avg_loss = sum(epoch_losses[-print_interval:]) / min(print_interval, len(epoch_losses))
                    avg_gn   = sum(grad_norms[-print_interval:]) / max(1, min(print_interval, len(grad_norms)))
                    print(f"  Batch {batch_idx + 1}/{steps_per_epoch}: Loss = {avg_loss:.6f}")

                # Periodic mid-epoch checkpoint (guards against long epochs with no save)
                if checkpoint_interval > 0 and (batch_idx + 1) % checkpoint_interval == 0:
                    global_step = epoch * steps_per_epoch + batch_idx + 1
                    step_dir = os.path.join(output_dir, f'step_{global_step:06d}')
                    if strategy == 'lora':
                        model.save_lora(step_dir)
                    else:
                        model.save(step_dir)
                    print(f"  [ckpt] Step {global_step}: saved → {step_dir}")
            
            # Epoch summary
            if len(epoch_losses) > 0:
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"  Average Loss: {avg_epoch_loss:.6f}")
                print(f"  Batches processed: {len(epoch_losses)}")
                
                # Save best checkpoint (strategy-aware)
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    best_dir = os.path.join(output_dir, 'best_checkpoint')
                    if strategy == 'lora':
                        model.save_lora(best_dir)
                    else:
                        model.save(best_dir)
                    print(f"  ✓ Saved best checkpoint (loss={best_loss:.6f}) → {best_dir}")
            else:
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"  ⚠️  No batches completed (training stopped early due to NaN/Inf)")
                avg_epoch_loss = float('inf')
            
            # Save epoch checkpoint
            epoch_dir = os.path.join(output_dir, f'epoch_{epoch + 1:03d}')
            if strategy == 'lora':
                model.save_lora(epoch_dir)
            else:
                model.save(epoch_dir)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {output_dir}")
    
    # Save training config
    config = {
        'strategy': strategy,
        'checkpoint_path': checkpoint_path,
        'precision_policy': precision_policy,
        'lora_rank': lora_rank if strategy == 'lora' else None,
        'lora_alpha': lora_alpha if strategy == 'lora' else None,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'window_size': window_size,
        'augment': augment,
        'max_shift': max_shift,
        'best_loss': best_loss,
        'signal_format': signal_format,
    }
    
    with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # ---- restore stdout / stderr / close log files ----------------------
    _metrics_file.close()
    os.dup2(_saved_stderr_fd, 2)
    os.close(_saved_stderr_fd)
    _sys.stdout = _tee._stream
    _tee.close()
    # ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune AlphaGenome on CUT&Tag data (supports BigWig and BedGraph)'
    )

    # Configuration file (optional, but recommended)
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file. CLI arguments override config values.')

    # Required arguments (or can be specified in config)
    parser.add_argument('--checkpoint', required=False,
                       help='Path to pre-trained AlphaGenome checkpoint')
    parser.add_argument('--fasta', required=False,
                       help='Path to reference genome FASTA')
    parser.add_argument('--signal', required=False,
                       help='Path to CUT&Tag signal file (BigWig or BedGraph, auto-detected)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for checkpoints')

    # Strategy
    parser.add_argument('--strategy', type=str, default='head_replace',
                       choices=['head_replace', 'lora'],
                       help='Fine-tuning strategy: head_replace (default) or lora')
    parser.add_argument('--precision-policy', type=str,
                       default='params=float32,compute=float32,output=float32',
                       help='JMP mixed-precision policy (default: full float32)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Total batch size across all GPUs (default: 8)')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Peak learning rate (default: 5e-4)')

    # LoRA-specific (only used when --strategy lora)
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=float, default=16.0)

    # Data / window
    parser.add_argument('--window-size', type=int, default=131072,
                       help='Sequence window size in bp (default: 131072 = 2^17)')
    parser.add_argument('--organism', type=int, default=1,
                       help='Organism ID: 0=human, 1=mouse (default: 1=mouse)')
    parser.add_argument('--windows-bed', type=str, default=None,
                       help='REQUIRED: BED file with training windows (use safe windows to avoid artifacts)')
    parser.add_argument('--chrom-sizes', type=str, default=None,
                       help='Path to chrom.sizes file (required for BedGraph, optional for BigWig)')
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable data augmentation (NOTE: TensorFlow pipeline does not support augmentation yet)')
    parser.add_argument('--max-shift', type=int, default=1024)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--checkpoint-interval', type=int, default=None,
                       help='Save a checkpoint every N steps within an epoch (default: 500). Set 0 to disable.')

    args = parser.parse_args()

    # Load config file if provided
    config = {}
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}")
    
    # Merge config with CLI args (CLI args take precedence)
    # Helper function to get value with priority: CLI arg > config > default
    def get_value(cli_arg, config_key, default):
        if cli_arg is not None:
            return cli_arg
        return config.get(config_key, default)
    
    # Extract values with proper precedence
    checkpoint = get_value(args.checkpoint, 'checkpoint', None)
    fasta = get_value(args.fasta, 'fasta', None)
    signal = get_value(args.signal, 'signal', None)
    output = get_value(args.output, 'output', '/home/ray/AG_Finetune/checkpoints/fine_tuned')
    strategy = get_value(args.strategy, 'strategy', 'head_replace')
    epochs = get_value(args.epochs, 'epochs', 10)
    batch_size = get_value(args.batch_size, 'batch_size', 8)
    lr = get_value(args.lr, 'learning_rate', 5e-4)
    lora_rank = get_value(args.lora_rank, 'lora_rank', 8)
    lora_alpha = get_value(args.lora_alpha, 'lora_alpha', 16.0)
    window_size = get_value(args.window_size, 'window_size', 131072)
    organism = get_value(args.organism, 'organism_id', 1)
    windows_bed = get_value(args.windows_bed, 'windows_bed', None)
    chrom_sizes = get_value(args.chrom_sizes, 'chrom_sizes', None)
    max_shift = get_value(args.max_shift, 'max_shift', 1024)
    num_workers = get_value(args.num_workers, 'num_workers', 0)
    precision_policy = get_value(args.precision_policy, 'precision_policy', 
                                  'params=float32,compute=float32,output=float32')
    
    # Handle augment flag (inverted logic for --no-augment)
    if args.no_augment:
        augment = False
    else:
        augment = config.get('augment', True)
    checkpoint_interval = get_value(args.checkpoint_interval, 'checkpoint_interval', 500)

    # Validate required arguments
    if not checkpoint:
        raise ValueError("--checkpoint is required (or specify 'checkpoint' in config file)")
    if not fasta:
        raise ValueError("--fasta is required (or specify 'fasta' in config file)")
    if not signal:
        raise ValueError("--signal is required (or specify 'signal' in config file)")

    train(
        checkpoint_path=checkpoint,
        fasta_path=fasta,
        signal_path=signal,
        output_prefix=output,
        strategy=strategy,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        window_size=window_size,
        organism_id=organism,
        genome_windows=windows_bed,
        augment=augment,
        max_shift=max_shift,
        num_workers=num_workers,
        precision_policy=precision_policy,
        config_path=args.config,
        chrom_sizes_path=chrom_sizes,
        checkpoint_interval=checkpoint_interval,
    )


if __name__ == '__main__':
    main()
