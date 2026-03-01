"""
Fine-tuning script for AlphaGenome with a trunk-freeze → full-unfreeze curriculum.

Identical to tf_ft.py except:

  Phase 1  (first `freeze_epochs` epochs)
    The pretrained trunk is frozen.  Only the new ATAC output head receives
    gradient updates.  Because JAX never differentiates through the ~88% of
    parameters that live in the trunk, the backward pass is ~4–6× cheaper,
    cutting per-step wall time from ~28 s to roughly ~8–12 s.  The head
    converges quickly before full joint training begins.

  Phase 2  (remaining epochs)
    All parameters are trained jointly — same as plain head_replace in tf_ft.py.
    A fresh full-parameter optimizer is initialised at the start of this phase
    (no stale momentum from the head-only phase bleeds into trunk updates).

Controlled by:
    --freeze-epochs N      CLI flag (default 1)
    freeze_epochs: N       YAML config key

Usage:
    LD_PRELOAD=/home/ray/miniforge3/envs/alphagenome/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2 \\
        python scripts/tf_ft_freeze.py --config configs/config_bigwig_freeze.yaml
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Iterator, Any, Mapping, Callable, Optional
from functools import partial
import itertools
import yaml
import shutil
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Suppress noisy C++ rendezvous warnings (false positives) ──────────────
os.environ.setdefault('GLOG_minloglevel', '3')

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
import pandas as pd
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import optax
import orbax.checkpoint as ocp

from alphagenome_research.model import dna_model, model as alphagenome_model
from alphagenome_research.model.metadata import metadata as metadata_lib
from alphagenome_research.io import fasta as fasta_lib
from alphagenome_research.model import one_hot_encoder

from models.model import HeadReplaceAlphaGenomeModel
from utils.borzoi_loss import cut_and_tag_loss_baskerville
from dataset.tf_dataset import CutAndTagDataPipelineTF

import json
import csv
import datetime
import contextlib
import sys as _sys


# ---------------------------------------------------------------------------
# Config / logging helpers  (identical to tf_ft.py)
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config if config is not None else {}


def backup_config(config_path: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    dest = os.path.join(output_dir, os.path.basename(config_path))
    shutil.copy2(config_path, dest)
    return dest


class _Tee:
    """Mirror writes to both the original stream and a file."""
    def __init__(self, stream, filepath):
        self._stream = stream
        self._file = open(filepath, 'a', buffering=1)

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    def __getattr__(self, name):
        return getattr(self._stream, name)


# ---------------------------------------------------------------------------
# Data pipeline  (identical to tf_ft.py)
# ---------------------------------------------------------------------------

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


class CutAndTagDataPipeline:
    """TensorFlow data pipeline for CUT&Tag fine-tuning (BigWig and BedGraph)."""

    def __init__(self, fasta_path, signal_path, windows_bed, window_size,
                 organism_id, chrom_sizes_path=None):
        self.fasta_extractor = fasta_lib.FastaExtractor(fasta_path)
        self.signal_extractor = CutAndTagDataPipelineTF(signal_path, chrom_sizes_path)
        self.window_size = window_size
        self.organism_id = organism_id
        self.one_hot_encoder = one_hot_encoder.DNAOneHotEncoder()

        # Load or generate training windows
        if windows_bed is not None:
            print(f"   Loading training windows from {windows_bed}...")
            self.intervals = pd.read_csv(
                windows_bed, sep='\t', header=None,
                names=['chrom', 'start', 'end'], comment='#'
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
        row = self.intervals.iloc[idx]
        chrom, start, end = row['chrom'], int(row['start']), int(row['end'])
        if (end - start) != self.window_size:
            center = (start + end) // 2
            start = center - self.window_size // 2
            end = start + self.window_size
        try:
            from alphagenome.data import genome
            interval = genome.Interval(chromosome=chrom, start=start, end=end)
            seq_str = self.fasta_extractor.extract(interval)
            seq_one_hot = self.one_hot_encoder.encode(seq_str)
        except Exception as e:
            print(f"Warning: sequence {chrom}:{start}-{end}: {e}")
            seq_one_hot = np.zeros((self.window_size, 4), dtype=np.float32)
        try:
            signal = self.signal_extractor.get_signal(chrom, start, end)
            if signal is None or len(signal) != self.window_size:
                signal = np.zeros(self.window_size, dtype=np.float32)
            else:
                signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        except Exception as e:
            print(f"Warning: signal {chrom}:{start}-{end}: {e}")
            signal = np.zeros(self.window_size, dtype=np.float32)
        return {'dna_sequence': seq_one_hot, 'organism_index': self.organism_id, 'signal': signal}

    def get_output_signature(self):
        return {
            'dna_sequence': tf.TensorSpec(shape=(self.window_size, 4), dtype=tf.float32),
            'organism_index': tf.TensorSpec(shape=[], dtype=tf.int32),
            'signal': tf.TensorSpec(shape=(self.window_size,), dtype=tf.float32),
        }

    def get_generator(self, shuffle: bool = True, seed: int = 0):
        rng = np.random.default_rng(seed=seed)
        intervals = self.intervals.copy()
        while True:
            if shuffle:
                intervals = intervals.sample(frac=1, random_state=rng).reset_index(drop=True)
            for idx in range(len(intervals)):
                try:
                    yield self.get_element(idx)
                except Exception as e:
                    print(f"Warning: element {idx}: {e}")
                    continue

    def close(self):
        self.signal_extractor.close()


# ---------------------------------------------------------------------------
# FreezeHeadReplaceAlphaGenomeModel
# ---------------------------------------------------------------------------

class FreezeHeadReplaceAlphaGenomeModel:
    """
    Head-replace fine-tuning with a two-phase training curriculum.

    Phase 1 — trunk frozen
        Only the new output head is optimised.  Gradients for the trunk
        (~88% of parameters) are never computed, making the backward pass
        ~4–6× cheaper than full training.

    Phase 2 — trunk unfrozen (full training)
        All parameters are jointly optimised.  A fresh full-parameter
        optimizer is initialised to avoid stale momentum from Phase 1.

    Call ``model.unfreeze()`` to switch from Phase 1 to Phase 2.
    """

    def __init__(
        self,
        params: Dict,
        state: Dict,
        apply_fn: Callable,
        mesh: Optional[Mesh] = None,
        learning_rate: float = 5e-4,
        warmup_steps: int = 0,
        total_steps: int = 0,
        freeze_warmup_steps: int = 0,
        freeze_total_steps: int = 0,
    ):
        """
        Args:
            params: Merged parameters (trunk pretrained + fresh head).
            state:  Model state (BatchNorm EMAs from pretrained checkpoint).
            apply_fn: ``apply_fn(params, state, seq, org) -> predictions``.
            mesh:   JAX device mesh for data-parallel training.
            learning_rate: Peak Adam LR for both phases.
            warmup_steps:  Warmup steps for *full* (Phase 2) cosine schedule.
            total_steps:   Total steps for *full* (Phase 2) cosine schedule.
            freeze_warmup_steps: Warmup steps for *frozen* (Phase 1) schedule.
            freeze_total_steps:  Total steps for *frozen* (Phase 1) schedule.
        """
        self.apply_fn = apply_fn
        self.state = state
        self.mesh = mesh
        self.num_devices = jax.device_count()
        self._frozen = True  # Start in frozen phase

        # Guard against YAML parsing 5e-4 as a string on some PyYAML versions
        learning_rate  = float(learning_rate)
        warmup_steps   = int(warmup_steps)
        total_steps    = int(total_steps)
        freeze_warmup_steps = int(freeze_warmup_steps)
        freeze_total_steps  = int(freeze_total_steps)

        # ── Split params into head and trunk ──────────────────────────────
        self.head_params = hk.data_structures.filter(
            lambda module_name, *_: 'head' in module_name, params
        )
        self.trunk_params = hk.data_structures.filter(
            lambda module_name, *_: 'head' not in module_name, params
        )
        n_head  = sum(x.size for x in jax.tree_util.tree_leaves(self.head_params))
        n_trunk = sum(x.size for x in jax.tree_util.tree_leaves(self.trunk_params))
        n_total = n_head + n_trunk

        # ── Frozen-phase optimizer  (head only) ───────────────────────────
        if freeze_warmup_steps > 0 and freeze_total_steps > 0:
            freeze_lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=freeze_warmup_steps,
                decay_steps=freeze_total_steps,
                end_value=learning_rate * 0.1,
            )
            print(f"  [Freeze phase] LR warmup: {freeze_warmup_steps} steps, "
                  f"cosine over {freeze_total_steps} steps")
        else:
            freeze_lr_schedule = learning_rate

        self._head_optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(freeze_lr_schedule),
        )
        self._head_opt_state = self._head_optimizer.init(self.head_params)

        # ── Full-phase optimizer  (all params) — init lazily on unfreeze() ─
        if warmup_steps > 0 and total_steps > 0:
            full_lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=total_steps,
                end_value=learning_rate * 0.1,
            )
            print(f"  [Full phase]   LR warmup: {warmup_steps} steps, "
                  f"cosine over {total_steps} steps")
        else:
            full_lr_schedule = learning_rate

        self._full_optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(full_lr_schedule),
        )
        self._full_opt_state = None   # Created in unfreeze()
        self.params = None            # Populated in unfreeze()

        # ── JIT-compile training steps ────────────────────────────────────
        if mesh is not None:
            _data = P('data')
            _repl = P()

            # Frozen step: head_params + head_opt_state are replicated;
            # trunk_params lives in the Python closure — never passed to JIT.
            self._frozen_train_step_jit = jax.jit(
                self._frozen_train_step,
                static_argnames=['loss_fn'],
                in_shardings=(
                    _data,  # dna_sequence
                    _data,  # organism_index
                    _data,  # targets dict
                    _repl,  # head_params
                    _repl,  # head_opt_state
                ),
                out_shardings=(_repl, _repl, _repl),
            )

            # Full step: all params replicated (mirrors HeadReplaceAlphaGenomeModel)
            self._full_train_step_jit = jax.jit(
                self._full_train_step,
                static_argnames=['loss_fn'],
                in_shardings=(
                    _data,  # dna_sequence
                    _data,  # organism_index
                    _data,  # targets dict
                    _repl,  # params
                    _repl,  # state
                    _repl,  # opt_state
                ),
                out_shardings=(_repl, _repl, _repl),
            )
        else:
            self._frozen_train_step_jit = jax.jit(
                self._frozen_train_step, static_argnames=['loss_fn']
            )
            self._full_train_step_jit = jax.jit(
                self._full_train_step, static_argnames=['loss_fn']
            )

        print(f"✓ FreezeHeadReplaceAlphaGenomeModel initialized")
        print(f"  Trunk params (frozen): {n_trunk:,}  ({100*n_trunk/n_total:.1f}%)")
        print(f"  Head  params (active): {n_head:,}  ({100*n_head/n_total:.1f}%)")
        print(f"  Total params:          {n_total:,}")
        if self.num_devices > 1:
            print(f"  Multi-GPU: {self.num_devices} devices")

    # ------------------------------------------------------------------
    # Phase 1: frozen trunk, head-only gradient
    # ------------------------------------------------------------------

    def _frozen_train_step(
        self,
        dna_sequence: jnp.ndarray,
        organism_index: jnp.ndarray,
        targets: Dict,
        head_params: Dict,
        head_opt_state: optax.OptState,
        loss_fn: Callable,
    ):
        """
        Computes gradients only for head_params.

        trunk_params is accessed from ``self`` (Python closure) — JAX never
        differentiates through it because it is not an argument to value_and_grad.
        """
        # Capture trunk_params in closure so JAX doesn't diff through it
        trunk_params = self.trunk_params

        def compute_loss(hp):
            full_params = hk.data_structures.merge(trunk_params, hp)
            predictions = self.apply_fn(
                full_params, self.state, dna_sequence, organism_index
            )
            return loss_fn(predictions, targets['signal'])

        loss_value, head_grads = jax.value_and_grad(compute_loss)(head_params)

        updates, new_head_opt_state = self._head_optimizer.update(head_grads, head_opt_state)
        new_head_params = optax.apply_updates(head_params, updates)

        return loss_value, new_head_params, new_head_opt_state

    def _frozen_train_wrapper(
        self,
        dna_sequence: jnp.ndarray,
        organism_index: jnp.ndarray,
        targets: Dict,
        loss_fn: Callable,
    ):
        loss_value, new_head_params, new_head_opt_state = self._frozen_train_step_jit(
            dna_sequence, organism_index, targets,
            self.head_params, self._head_opt_state, loss_fn,
        )
        self.head_params = new_head_params
        self._head_opt_state = new_head_opt_state
        return loss_value

    # ------------------------------------------------------------------
    # Phase 2: all params, full gradient
    # ------------------------------------------------------------------

    def _full_train_step(
        self,
        dna_sequence: jnp.ndarray,
        organism_index: jnp.ndarray,
        targets: Dict,
        params: Dict,
        state: Dict,
        opt_state: optax.OptState,
        loss_fn: Callable,
    ):
        def compute_loss(p):
            predictions = self.apply_fn(p, state, dna_sequence, organism_index)
            return loss_fn(predictions, targets['signal'])

        loss_value, grads = jax.value_and_grad(compute_loss)(params)
        updates, new_opt_state = self._full_optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return loss_value, new_params, new_opt_state

    def _full_train_wrapper(
        self,
        dna_sequence: jnp.ndarray,
        organism_index: jnp.ndarray,
        targets: Dict,
        loss_fn: Callable,
    ):
        loss_value, new_params, new_opt_state = self._full_train_step_jit(
            dna_sequence, organism_index, targets,
            self.params, self.state, self._full_opt_state, loss_fn,
        )
        self.params = new_params
        self._full_opt_state = new_opt_state
        return loss_value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def unfreeze(self):
        """
        Switch to Phase 2: all parameters are now trainable.

        Merges back trunk + head params and initialises a fresh full-parameter
        optimizer (no stale momentum from Phase 1 bleeds into trunk updates).
        """
        if not self._frozen:
            print("  [unfreeze] Already in full-training mode — no-op.")
            return
        self._frozen = False
        self.params = hk.data_structures.merge(self.trunk_params, self.head_params)
        self._full_opt_state = self._full_optimizer.init(self.params)
        n = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print(f"  ✓ Trunk unfrozen — now optimising all {n:,} parameters")
        print(f"    (Fresh optimizer state initialised — no momentum bleed from Phase 1)")

    def train_step(
        self,
        dna_sequence: jnp.ndarray,
        organism_index: jnp.ndarray,
        targets: Dict,
        loss_fn: Callable,
    ):
        """Unified interface: dispatches to frozen or full step automatically."""
        if self._frozen:
            loss_value = self._frozen_train_wrapper(
                dna_sequence, organism_index, targets, loss_fn
            )
        else:
            loss_value = self._full_train_wrapper(
                dna_sequence, organism_index, targets, loss_fn
            )
        return loss_value, jnp.float32(0.0)

    def save(self, dirpath: str):
        """
        Save params+state via orbax.  Compatible with dna_model.create() loading.

        In frozen phase: merges trunk+head first, then saves.
        In full phase:   saves self.params directly.
        """
        import shutil
        if os.path.exists(dirpath):
            shutil.rmtree(dirpath)
        checkpointer = ocp.StandardCheckpointer()
        if self._frozen:
            save_params = hk.data_structures.merge(self.trunk_params, self.head_params)
        else:
            save_params = self.params
        # Save as list [params, state] matching the original kagglehub checkpoint
        # format so dna_model.create() can load via `params, state = restore(path)`.
        # jax.device_get() gathers all arrays to numpy, stripping any 8-GPU
        # NamedSharding so the checkpoint loads cleanly on a single GPU.
        checkpointer.save(dirpath, jax.device_get([save_params, self.state]))
        checkpointer.wait_until_finished()

    def load(self, dirpath: str):
        """Restore params+state via orbax. Reinitialises optimizer state."""
        checkpointer = ocp.StandardCheckpointer()
        restored = checkpointer.restore(dirpath)
        # Checkpoint saved as [params, state] list
        all_params, self.state = restored
        # Re-split
        self.head_params = hk.data_structures.filter(
            lambda module_name, *_: 'head' in module_name, all_params
        )
        self.trunk_params = hk.data_structures.filter(
            lambda module_name, *_: 'head' not in module_name, all_params
        )
        self._head_opt_state = self._head_optimizer.init(self.head_params)
        if not self._frozen:
            self.params = all_params
            self._full_opt_state = self._full_optimizer.init(self.params)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    checkpoint_path: str,
    fasta_path: str,
    signal_path: str,
    output_prefix: str,
    strategy: str = 'head_replace',
    epochs: int = 10,
    freeze_epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
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
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
):
    """
    Fine-tune AlphaGenome on CUT&Tag data with trunk-freeze curriculum.

    Only `head_replace` strategy supports the freeze/unfreeze curriculum.
    `lora` strategy falls back to plain LoRA training (freeze has no meaning
    since the trunk is never directly optimised in LoRA either).
    """

    # ── Logging setup ───────────────────────────────────────────────────────
    _ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = f"{output_prefix}_{strategy}_freeze{freeze_epochs}_{_ts}"
    os.makedirs(output_dir, exist_ok=True)

    log_path     = os.path.join(output_dir, 'train.log')
    metrics_path = os.path.join(output_dir, 'metrics.csv')
    _tee = _Tee(_sys.stdout, log_path)
    _sys.stdout = _tee
    print(f"[log] stdout  → {log_path}")
    print(f"[log] metrics → {metrics_path}")

    if config_path and os.path.exists(config_path):
        print(f"[config] backed up → {backup_config(config_path, output_dir)}")

    _metrics_file   = open(metrics_path, 'a', newline='', buffering=1)
    _metrics_writer = csv.writer(_metrics_file)
    if os.path.getsize(metrics_path) == 0:
        _metrics_writer.writerow(
            ['timestamp', 'epoch', 'batch', 'loss', 'frozen', 'nan_event', 'skipped']
        )

    _stderr_log_path = os.path.join(output_dir, 'stderr.log')
    _stderr_log_fd   = os.open(_stderr_log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    _saved_stderr_fd = os.dup(2)
    os.dup2(_stderr_log_fd, 2)
    os.close(_stderr_log_fd)
    print(f"[log] stderr  → {_stderr_log_path}")
    # ────────────────────────────────────────────────────────────────────────

    num_devices = jax.device_count()
    if num_devices > 1:
        print("=" * 60)
        print(f"MULTI-GPU DETECTED: {num_devices} devices")
        print("=" * 60)
        print(f"Devices: {jax.devices()}")
        devices = mesh_utils.create_device_mesh((num_devices,))
        mesh = Mesh(devices, axis_names=('data',))
        data_sharding = P('data')
    else:
        print("Single GPU mode")
        mesh = None
        data_sharding = None

    if window_size % 128 != 0:
        raise ValueError(f"window_size must be divisible by 128, got {window_size}")

    print("=" * 60)
    print(f"FINE-TUNING SETUP  [strategy={strategy}]")
    print("=" * 60)
    print(f"Precision policy : {precision_policy}")
    print(f"Window size      : {window_size:,} bp (2^{int(np.log2(window_size))})")
    print(f"Freeze/unfreeze  : trunk frozen for {freeze_epochs} epoch(s), "
          f"then full training for {epochs - freeze_epochs} epoch(s)")

    checkpoint_path = os.path.abspath(checkpoint_path)
    fasta_path      = os.path.abspath(fasta_path)
    signal_path     = os.path.abspath(signal_path)
    output_dir      = os.path.abspath(output_dir)

    signal_format = "BigWig" if signal_path.endswith(('.bigwig', '.bw')) else "BedGraph"
    print(f"Signal format    : {signal_format}")

    # Resolve genome_windows path if provided
    if genome_windows:
        genome_windows = os.path.abspath(genome_windows)
    else:
        print("   Note: windows_bed not set — whole-genome windows will be generated from chrom_sizes")

    # ── Data pipeline ───────────────────────────────────────────────────────
    print(f"\n1. Setting up TensorFlow data pipeline...")
    pipeline = CutAndTagDataPipeline(
        fasta_path=fasta_path,
        signal_path=signal_path,
        windows_bed=genome_windows,
        window_size=window_size,
        organism_id=organism_id,
        chrom_sizes_path=chrom_sizes_path,
    )
    tf_dataset = (
        tf.data.Dataset.from_generator(
            pipeline.get_generator,
            output_signature=pipeline.get_output_signature(),
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    num_windows      = len(pipeline.intervals)
    steps_per_epoch  = num_windows // batch_size

    # Schedule for full-training phase (Phase 2)
    full_phase_epochs = max(0, epochs - freeze_epochs)
    full_total_steps  = steps_per_epoch * max(1, full_phase_epochs)
    full_warmup_steps = int(0.05 * full_total_steps)

    # Schedule for frozen phase (Phase 1) — shorter
    freeze_phase_steps  = steps_per_epoch * max(1, freeze_epochs)
    freeze_warmup_steps = int(0.05 * freeze_phase_steps)

    print(f"   TF dataset        : {num_windows} windows, {steps_per_epoch} steps/epoch")
    print(f"   Frozen phase      : {freeze_epochs} epoch(s), "
          f"{freeze_phase_steps} steps, {freeze_warmup_steps} warmup")
    print(f"   Full-train phase  : {full_phase_epochs} epoch(s), "
          f"{full_total_steps} steps, {full_warmup_steps} warmup")

    # ── Load base model ─────────────────────────────────────────────────────
    print(f"\n2. Loading base model from {checkpoint_path}...")
    base_model = dna_model.create(
        checkpoint_path=checkpoint_path,
        organism_settings=None,
        device=None,
    )
    print(f"   ✓ Model loaded on {base_model._device_context._device}")

    # ── Build head_replace model ────────────────────────────────────────────
    mesh_ctx = jax.set_mesh(mesh) if mesh is not None else contextlib.nullcontext()

    if strategy == 'head_replace':
        print(f"\n3. [head_replace + freeze] Building new ATAC output head...")
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
        new_head_params = hk.data_structures.filter(
            lambda module_name, *_: 'head' in module_name, new_params
        )
        base_trunk_params = hk.data_structures.filter(
            lambda module_name, *_: 'head' not in module_name, base_model._params
        )
        merged_params = hk.data_structures.merge(base_trunk_params, new_head_params)
        print(f"   ✓ Weight surgery complete")

        model = FreezeHeadReplaceAlphaGenomeModel(
            params=merged_params,
            state=base_model._state,
            apply_fn=new_apply_fn,
            mesh=mesh,
            learning_rate=learning_rate,
            warmup_steps=full_warmup_steps,
            total_steps=full_total_steps,
            freeze_warmup_steps=freeze_warmup_steps,
            freeze_total_steps=freeze_phase_steps,
        )

    else:
        raise ValueError(
            f"Strategy '{strategy}' is not supported by tf_ft_freeze.py. "
            "Only 'head_replace' supports the freeze/unfreeze curriculum. "
            "Use tf_ft.py for LoRA."
        )

    # ── Training loop ───────────────────────────────────────────────────────
    print(f"\n4. Starting training for {epochs} epochs  "
          f"[freeze_epochs={freeze_epochs}]...")
    loss_fn  = cut_and_tag_loss_baskerville
    best_loss = float('inf')

    with mesh_ctx:
        for epoch in range(epochs):
            # ── Phase transition ──────────────────────────────────────────
            if epoch == freeze_epochs and model.is_frozen:
                print(f"\n{'=' * 60}")
                print(f"PHASE TRANSITION: Epoch {epoch + 1} — unfreezing trunk")
                print(f"{'=' * 60}")
                model.unfreeze()
                # Recompile: the full-train JIT hasn't been traced yet
                print(f"  (Full-parameter JIT will recompile on first batch of this epoch)")

            frozen_label = " [TRUNK FROZEN]" if model.is_frozen else " [FULL TRAIN]"
            print(f"\nEpoch {epoch + 1}/{epochs}{frozen_label}")
            print("-" * 60)

            epoch_losses = []
            ds_iter = tf_dataset.as_numpy_iterator()

            for batch_idx, batch in enumerate(
                itertools.islice(ds_iter, steps_per_epoch)
            ):
                if batch_idx == 0:
                    print(f"  Compiling JAX functions "
                          f"(this may take 2–5 minutes on first batch)...")
                    start_time = time.time()

                sequences   = jnp.array(batch['dna_sequence'], dtype=jnp.float32)
                organism_ids = jnp.array(batch['organism_index'], dtype=jnp.int32)
                targets     = {'signal': jnp.array(batch['signal'], dtype=jnp.float32)}

                if data_sharding is not None:
                    sequences    = jax.device_put(sequences,    data_sharding)
                    organism_ids = jax.device_put(organism_ids, data_sharding)
                    targets = {'signal': jax.device_put(targets['signal'], data_sharding)}

                loss_value, _ = model.train_step(
                    sequences, organism_ids, targets, loss_fn
                )
                loss_host = float(jax.device_get(loss_value))
                epoch_losses.append(loss_host)

                _metrics_writer.writerow([
                    datetime.datetime.now().isoformat(),
                    epoch + 1, batch_idx + 1,
                    loss_host, model.is_frozen, False, False,
                ])

                if batch_idx == 0:
                    print(f"  ✓ Compilation complete! Took {time.time() - start_time:.1f}s")
                    if data_sharding is not None:
                        print(f"\n  Array Sharding Visualization:")
                        print(f"  Sequences shape: {sequences.shape}")
                        try:
                            print(f"  Sequences[:, 0, 0] (batch dimension):")
                            jax.debug.visualize_array_sharding(sequences[:, 0, 0])
                        except Exception:
                            pass
                    print(f"  Training on {steps_per_epoch} batches...")

                print_interval = 1 if steps_per_epoch < 20 else 10
                if (batch_idx + 1) % print_interval == 0:
                    avg = sum(epoch_losses[-print_interval:]) / min(
                        print_interval, len(epoch_losses)
                    )
                    print(f"  Batch {batch_idx + 1}/{steps_per_epoch}: Loss = {avg:.6f}")

                # Mid-epoch periodic checkpoint
                if checkpoint_interval > 0 and (batch_idx + 1) % checkpoint_interval == 0:
                    global_step = epoch * steps_per_epoch + batch_idx + 1
                    step_dir = os.path.join(output_dir, f'step_{global_step:06d}')
                    model.save(step_dir)
                    phase = "frozen" if model.is_frozen else "full"
                    print(f"  [ckpt] Step {global_step} ({phase}): saved → {step_dir}")

            # ── Epoch summary ─────────────────────────────────────────────
            if epoch_losses:
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                phase_label = "frozen" if model.is_frozen else "full-train"
                print(f"\nEpoch {epoch + 1} Summary  [{phase_label}]:")
                print(f"  Average Loss : {avg_epoch_loss:.6f}")
                print(f"  Batches      : {len(epoch_losses)}")

                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    best_dir = os.path.join(output_dir, 'best_checkpoint')
                    model.save(best_dir)
                    print(f"  ✓ Saved best checkpoint (loss={best_loss:.6f}) → {best_dir}")
            else:
                avg_epoch_loss = float('inf')
                print(f"\nEpoch {epoch + 1}: ⚠ No batches completed")

            epoch_dir = os.path.join(output_dir, f'epoch_{epoch + 1:03d}')
            model.save(epoch_dir)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints: {output_dir}")

    config_out = {
        'strategy': strategy,
        'freeze_epochs': freeze_epochs,
        'checkpoint_path': checkpoint_path,
        'precision_policy': precision_policy,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'batch_size': batch_size,
        'window_size': window_size,
        'best_loss': best_loss,
        'signal_format': signal_format,
    }
    with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
        json.dump(config_out, f, indent=2)

    _metrics_file.close()
    os.dup2(_saved_stderr_fd, 2)
    os.close(_saved_stderr_fd)
    _sys.stdout = _tee._stream
    _tee.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='AlphaGenome CUT&Tag fine-tuning with trunk-freeze curriculum'
    )
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', required=False)
    parser.add_argument('--fasta', required=False)
    parser.add_argument('--signal', required=False)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--strategy', type=str, default='head_replace',
                        choices=['head_replace'])
    parser.add_argument('--precision-policy', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--freeze-epochs', type=int, default=None,
                        help='Number of epochs to freeze trunk (default: 1). '
                             'Set 0 to disable freeze (same as tf_ft.py).')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--window-size', type=int, default=None)
    parser.add_argument('--organism', type=int, default=None)
    parser.add_argument('--windows-bed', type=str, default=None)
    parser.add_argument('--chrom-sizes', type=str, default=None)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--max-shift', type=int, default=None)
    parser.add_argument('--checkpoint-interval', type=int, default=None)

    args = parser.parse_args()

    config = {}
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config not found: {args.config}")
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}")

    def get_value(cli_arg, config_key, default):
        if cli_arg is not None:
            return cli_arg
        return config.get(config_key, default)

    checkpoint    = get_value(args.checkpoint,   'checkpoint',    None)
    fasta         = get_value(args.fasta,         'fasta',         None)
    signal        = get_value(args.signal,        'signal',        None)
    output        = get_value(args.output,        'output',
                              '/home/ray/AG_Finetune/checkpoints/fine_tuned')
    strategy      = get_value(args.strategy,      'strategy',      'head_replace')
    epochs        = get_value(args.epochs,        'epochs',        10)
    freeze_epochs = get_value(args.freeze_epochs, 'freeze_epochs', 1)
    batch_size    = get_value(args.batch_size,    'batch_size',    8)
    lr            = get_value(args.lr,            'learning_rate', 5e-4)
    window_size   = get_value(args.window_size,   'window_size',   131072)
    organism      = get_value(args.organism,      'organism_id',   1)
    windows_bed   = get_value(args.windows_bed,   'windows_bed',   None)
    chrom_sizes   = get_value(args.chrom_sizes,   'chrom_sizes',   None)
    max_shift     = get_value(args.max_shift,     'max_shift',     1024)
    precision_policy = get_value(
        args.precision_policy, 'precision_policy',
        'params=float32,compute=float32,output=float32'
    )
    augment = False if args.no_augment else config.get('augment', True)
    checkpoint_interval = get_value(
        args.checkpoint_interval, 'checkpoint_interval', 500
    )

    if not checkpoint:
        raise ValueError("--checkpoint is required")
    if not fasta:
        raise ValueError("--fasta is required")
    if not signal:
        raise ValueError("--signal is required")

    train(
        checkpoint_path=checkpoint,
        fasta_path=fasta,
        signal_path=signal,
        output_prefix=output,
        strategy=strategy,
        epochs=epochs,
        freeze_epochs=freeze_epochs,
        batch_size=batch_size,
        learning_rate=lr,
        window_size=window_size,
        organism_id=organism,
        genome_windows=windows_bed,
        augment=augment,
        max_shift=max_shift,
        precision_policy=precision_policy,
        config_path=args.config,
        chrom_sizes_path=chrom_sizes,
        checkpoint_interval=checkpoint_interval,
    )


if __name__ == '__main__':
    main()
