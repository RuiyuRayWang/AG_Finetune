"""
Training script for fine-tuning AlphaGenome on CUT&Tag data.

Two strategies are supported (select with --strategy):

  head_replace (default)
    DeepMind-style full fine-tuning: replaces output heads with new task-specific
    heads (1 ATAC track for CUT&Tag) and fine-tunes ALL parameters jointly.
    Mirrors the official AlphaGenome fine-tuning colab exactly.

  lora
    Parameter-efficient fine-tuning: LoRA adapters on Q/K/V only (~428K params).
    Keeps pretrained heads; supervised against pretrained ATAC head.

Both strategies use float32 compute (no bfloat16) to avoid the forward-NaN
issue observed at bf16 precision.

Multi-GPU Support (8 GPUs tested):
  - JAX data parallelism with parameter replication
  - Explicit sharding specifications in jax.jit() (prevents NCCL errors)
  - Validation statistics use jax.device_get() to avoid NCCL AllReduce

Data augmentation (following AlphaGenome pretraining):
  - Shift augmentation: randomly shift windows by ±1024 bp
  - Reverse complementation: 50% probability (both sequence and signal)

Usage:
    # Using config file (recommended):
    python train.py --config configs/config.yaml
    
    # Config file with CLI overrides:
    python train.py --config configs/config.yaml --batch-size 16 --lr 1e-4

    # Direct CLI arguments (head-replacement):
    python train.py --strategy head_replace \
        --checkpoint /path/to/alphagenome-fold-0 \
        --fasta /path/to/genome.fa \
        --signal /path/to/signal.bedGraph \  # or signal.bw for BigWig
        --output ./checkpoints/fine_tuned

    # LoRA (parameter-efficient alternative):
    python train.py --strategy lora \
        --checkpoint /path/to/alphagenome-fold-0 \
        --fasta /path/to/genome.fa \
        --signal /path/to/signal.bedGraph \
        --lora-rank 8

Config file:
    The config file (YAML format) is automatically backed up to the output directory
    with a timestamp (e.g., config_20260223_083657.yaml). This ensures reproducibility
    and tracks the exact settings used for each training run.

Known Issues:
    PyTorch→JAX data pipeline exhibits corruption after ~1700 batches in multi-GPU
    setups (independent of signal format - occurs with both BedGraph and BigWig).
    Signal values spike to 10-4700x expected ranges. If you encounter unexpectedly
    large signal values (>50,000/bp), consider using tf_ft.py (TensorFlow pipeline).
    consider using tf_ft.py (TensorFlow data pipeline) as an alternative.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from functools import partial
import yaml
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import haiku as hk
import jmp
import pandas as pd
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from torch.utils.data import DataLoader
from alphagenome_research.model import dna_model, model as alphagenome_model
from alphagenome_research.model.metadata import metadata as metadata_lib
from models.model import LoRAAlphaGenomeModel, HeadReplaceAlphaGenomeModel
from dataset.dataset import CutAndTagDataset
from utils.lora_utils import get_attention_qkv_layers
from utils.borzoi_loss import cut_and_tag_loss_baskerville
import json
import csv
import datetime
import sys as _sys
import contextlib


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


def collate_to_jax(batch: List[Tuple[np.ndarray, int, Dict]]) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    """
    Custom collate function to convert batch of numpy arrays to JAX arrays.

    Both sequences and targets are kept as float32 throughout — the model
    runs in full float32 (no bfloat16) to avoid forward-NaN instability.

    Args:
        batch: List of (sequence, organism_id, targets) tuples from Dataset

    Returns:
        (sequences, organism_ids, targets) as JAX arrays:
            sequences: [batch, window_size, 4] - float32
            organism_ids: [batch] - int32
            targets: dict with 'signal' [batch, window_size] - float32 (raw counts)
    """
    sequences = [item[0] for item in batch]
    organism_ids = [item[1] for item in batch]
    signals = [item[2]['signal'] for item in batch]
    
    return (
        jnp.array(sequences, dtype=jnp.float32),
        jnp.array(organism_ids, dtype=jnp.int32),
        {'signal': jnp.array(signals, dtype=jnp.float32)}
    )


# cut_and_tag_loss function is now imported from utils.borzoi_loss
# It implements the Borzoi-style multinomial + Poisson NLL loss


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
):
    """
    Fine-tune AlphaGenome on CUT&Tag data.

    Args:
        strategy: 'head_replace' (full fine-tuning with new ATAC head, recommended)
                  or 'lora' (LoRA adapters on Q/K/V, parameter-efficient alternative).
        precision_policy: JMP policy string. Default is full float32 (safe).
        checkpoint_path: Path to pre-trained AlphaGenome checkpoint.
        fasta_path: Path to reference genome FASTA.
        signal_path: Path to CUT&Tag signal (BedGraph or BigWig, normalized, base-resolution).
        output_prefix: Prefix for output directory.
        learning_rate: Peak Adam LR (default 5e-4 for head_replace, 1e-5 for lora).
        lora_rank: LoRA rank (lora strategy only).
        lora_alpha: LoRA alpha (lora strategy only).
        window_size: bp per window. Must be divisible by 128.
        organism_id: 0=human, 1=mouse.
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
        # Simplified sharding: only specify first dimension, let JAX infer the rest
        data_sharding = NamedSharding(mesh, P('data'))
        batch_sharding = data_sharding  # alias for backward compatibility
        organism_sharding = data_sharding
        signal_sharding = data_sharding
    else:
        print("Single GPU mode")
        mesh = None
        batch_sharding = None
        organism_sharding = None
        signal_sharding = None
    
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
    
    # Convert paths to absolute
    checkpoint_path = os.path.abspath(checkpoint_path)
    fasta_path      = os.path.abspath(fasta_path)
    signal_path     = os.path.abspath(signal_path)
    output_dir      = os.path.abspath(output_dir)
    if genome_windows:
        genome_windows = os.path.abspath(genome_windows)

    # ------------------------------------------------------------------
    # Data loader — MUST be set up before model init so that
    # total_steps / warmup_steps are known for the LR schedule.
    # ------------------------------------------------------------------
    print(f"\n1. Setting up data loader...")
    dataset = CutAndTagDataset(
        fasta_path=fasta_path,
        signal_path=signal_path,
        genome_windows=genome_windows,
        window_size=window_size,
        organism_id=organism_id,
        augment=augment,
        max_shift=max_shift,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_to_jax,
        drop_last=True,
    )
    steps_per_epoch = len(data_loader)
    total_steps     = steps_per_epoch * epochs
    warmup_steps    = int(0.05 * total_steps)  # 5% warmup
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
    # Strategy: head_replace  (DeepMind official approach)
    # ------------------------------------------------------------------
    if strategy == 'head_replace':
        print(f"\n2. [head_replace] Building new ATAC output head for CUT&Tag...")
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

        print(f"   Initialising new head params (dummy forward pass)...")
        dummy_seq = jnp.zeros((1, window_size, 4), dtype=jnp.float32)
        dummy_org = jnp.zeros((1,), dtype=jnp.int32)
        new_params, _new_state = jax.jit(_forward_new.init)(
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
        print(f"\n2. [lora] Identifying LoRA target layers...")
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
    print(f"\n3. Starting training for {epochs} epochs  [strategy={strategy}]...")
    loss_fn = cut_and_tag_loss_baskerville
    best_loss = float('inf')
    grad_norms = []

    # Wrap training loop in mesh context (CRITICAL for multi-GPU)
    # This tells JAX the active mesh for all operations inside
    mesh_context = jax.set_mesh(mesh) if mesh is not None else None
    
    with mesh_context if mesh_context is not None else contextlib.nullcontext():
        for epoch in range(epochs):
            epoch_losses = []
            grad_norms = []

            print(f"\nEpoch {epoch + 1}/{epochs}  [strategy={strategy}]")
            print("-" * 60)

            for batch_idx, (sequences, organism_ids, targets) in enumerate(data_loader):
            # Progress indicator for first batch (JAX compilation happens here)
            if batch_idx == 0:
                print(f"  Compiling JAX functions (this may take 2-5 minutes on first batch)...")
                import time
                start_time = time.time()
            
            # Shard batch across devices using JAX automatic parallelization
            if batch_sharding is not None:
                # Use jax.device_put to physically shard arrays across devices
                # JAX will automatically parallelize computation based on this sharding
                sequences = jax.device_put(sequences, batch_sharding)
                organism_ids = jax.device_put(organism_ids, organism_sharding)
                targets = {
                    'signal': jax.device_put(targets['signal'], signal_sharding)
                }
            
            # Skip batches with extremely low signal (reduces sparse signal instability)
            # Use device_get() to avoid NCCL errors with sharded arrays
            target_sum = float(jnp.sum(jax.device_get(targets['signal'])))
            if target_sum < 1e-6:
                # Skip this batch - no signal to learn from
                _metrics_writer.writerow([datetime.datetime.now().isoformat(), epoch + 1, batch_idx + 1, '', '', False, True])
                continue

            # Skip batches where any signal value is physically impossible.
            # Bedgraph is normalised to 100M insertions; >50 000/bp is a data
            # corruption artifact. Catches the 1.5e24 class of overflows BEFORE
            # they can poison gradients or model parameters.
            target_max_raw = float(jnp.max(jax.device_get(targets['signal'])))
            if target_max_raw > 50000.0:
                print(f"\n  ⚠️  WARNING: Corrupt target at batch {batch_idx + 1} (max={target_max_raw:.3e}), skipping")
                _metrics_writer.writerow([datetime.datetime.now().isoformat(), epoch + 1, batch_idx + 1, '', '', True, True])
                continue
            
            # Training step (automatically parallelized based on input sharding)
            loss_value, predictions, grad_norm = model.train_step(
                sequences, organism_ids, targets, loss_fn
            )
            
            # Check for zero gradient (indicates NaN was sanitized)
            if grad_norm < 1e-10:
                print(f"\n  ⚠️  WARNING: Zero gradient at batch {batch_idx + 1}")
                print(f"  Gradient norm: {grad_norm:.2e}")
                print(f"  Loss value: {loss_value:.6f}")
                
                # Debug: check predictions to understand the issue
                # Use device_get() to avoid NCCL errors with sharded arrays
                if isinstance(predictions, dict) and 'atac' in predictions:
                    atac_preds = predictions['atac'].get('scaled_predictions_1bp', None)
                    if atac_preds is not None:
                        atac_preds_host = jax.device_get(atac_preds)
                        pred_min = float(jnp.min(atac_preds_host))
                        pred_max = float(jnp.max(atac_preds_host))
                        pred_mean = float(jnp.mean(atac_preds_host))
                        print(f"  Predictions: min={pred_min:.6f}, max={pred_max:.6f}, mean={pred_mean:.6f}")
                
                targets_host = jax.device_get(targets['signal'])
                target_min = float(jnp.min(targets_host))
                target_max = float(jnp.max(targets_host))
                target_mean = float(jnp.mean(targets_host))
                print(f"  Targets: min={target_min:.6f}, max={target_max:.6f}, mean={target_mean:.6f}")
                print(f"  This suggests model predictions are producing NaN in backward pass")
                print(f"  Skipping this batch...")
                _metrics_writer.writerow([datetime.datetime.now().isoformat(), epoch + 1, batch_idx + 1, float(loss_value), float(grad_norm), True, True])
                continue  # Skip update for this batch
            
            # Handle sentinel grad_norm codes (encoded in models/model.py)
            # -1 = NaN detected in gradients
            # -2 = Inf detected in gradients
            # -3 = NaN in params  → fatal, stop training
            # -4 = Inf in params  → fatal, stop training
            CLIP_VALUE = 0.5
            if grad_norm < 0:
                sentinel_map = {-1: 'grad NaN', -2: 'grad Inf',
                                -3: 'param NaN ❌', -4: 'param Inf ❌'}
                label = sentinel_map.get(int(grad_norm), f'unknown sentinel {grad_norm}')
                print(f"  ⚠️  Gradient issue at batch {batch_idx + 1}: {label}")
                if int(grad_norm) in (-3, -4):
                    print(f"  Model parameters are corrupted — stopping training.")
                    break
                # grad NaN/Inf: warn but let the (clipped) update proceed

            # Warn about large gradients (before clip_by_global_norm(0.5))
            if grad_norm > 2.0 * CLIP_VALUE:
                print(f"  ⚠️  Large gradient at batch {batch_idx + 1}: norm={grad_norm:.4f} (clipped to {CLIP_VALUE})")
            
            # Check for NaN/Inf loss
            if jnp.isnan(loss_value) or jnp.isinf(loss_value):
                print(f"\n  ⚠️  WARNING: NaN/Inf loss detected at batch {batch_idx + 1}")
                print(f"  Gradient norm: {grad_norm:.6f}")
                print(f"  Model may be corrupted - stopping training.")
                print(f"\n  Stopping training to prevent further issues.")
                break
            
            epoch_losses.append(float(loss_value))
            grad_norms.append(float(grad_norm))
            _metrics_writer.writerow([datetime.datetime.now().isoformat(), epoch + 1, batch_idx + 1, float(loss_value), float(grad_norm), False, False])
            
            # Log progress
            if batch_idx == 0:
                compile_time = time.time() - start_time
                print(f"  ✓ Compilation complete! Took {compile_time:.1f}s")
                
                # Visualize array sharding for debugging (multi-GPU only)
                if batch_sharding is not None:
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
                    print(f"  Gradient norm: {grad_norm}")
                    print()
                
                print(f"  Training on {len(data_loader)} batches...")
            
            # Print progress every 10 batches (or every batch if < 20 total batches)
            print_interval = 1 if len(data_loader) < 20 else 10
            if (batch_idx + 1) % print_interval == 0:
                    avg_loss = sum(epoch_losses[-print_interval:]) / min(print_interval, len(epoch_losses))
                    avg_gn   = sum(grad_norms[-print_interval:]) / max(1, min(print_interval, len(grad_norms)))
                    print(f"  Batch {batch_idx + 1}/{len(data_loader)}: Loss = {avg_loss:.6f}, Grad Norm = {avg_gn:.6f}")
            
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
    }
    
    with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # ---- restore stdout / close log files --------------------------------
    _metrics_file.close()
    _sys.stdout = _tee._stream
    _tee.close()
    # ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune AlphaGenome on CUT&Tag data (head_replace or lora strategy)'
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
                       help='Path to CUT&Tag signal file (BedGraph or BigWig)')
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
    parser.add_argument('--windows-bed', type=str, default=None)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--max-shift', type=int, default=1024)
    parser.add_argument('--num-workers', type=int, default=None)

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
    max_shift = get_value(args.max_shift, 'max_shift', 1024)
    num_workers = get_value(args.num_workers, 'num_workers', 0)
    precision_policy = get_value(args.precision_policy, 'precision_policy', 
                                  'params=float32,compute=float32,output=float32')
    
    # Handle augment flag (inverted logic for --no-augment)
    if args.no_augment:
        augment = False
    else:
        augment = config.get('augment', True)
    
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
    )


if __name__ == '__main__':
    main()
