"""
Fine-tuning script for AlphaGenome with a two-phase hybrid curriculum.

  Phase 1  (first ``freeze_epochs`` epochs)  — Head-only training
    The pretrained trunk is fully frozen.  Only the new ATAC output head
    receives gradient updates (~9 K parameters).  Per-step time ~0.15 s.

  Phase 2  (remaining epochs)  — Head + LoRA on trunk
    Trunk base weights remain frozen.  Low-rank adaptation (LoRA) matrices
    are added to the Q/K/V projection layers of every transformer block
    (~428 K parameters by default at rank 8) and trained jointly with the
    head.  Per-step time is intermediate between Phase 1 and full training,
    with much less risk of catastrophic forgetting than fine-tuning all
    450 M parameters.

Why this beats full unfreeze for histone-mark data
---------------------------------------------------
CUT&Tag H3K27ac / H3K27me3 signals differ substantially from the ATAC-seq
signal patterns the trunk was pre-trained on.  LoRA adapts the attention
layers just enough for the new biochemistry while preserving the general
genomic representations learned during pretraining.

Controlled by:
    freeze_epochs: N      YAML key  (Phase 1 length)
    lora_rank: R          YAML key  (default 8)
    lora_alpha: A         YAML key  (default 16.0)

Usage:
    LD_PRELOAD=/home/ray/miniforge3/envs/alphagenome/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2 \\
        python scripts/tf_ft_hybrid.py --config configs/config_bigwig_hybrid.yaml
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
import optax
import pandas as pd
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils
import orbax.checkpoint as ocp

from alphagenome_research.model import dna_model, model as alphagenome_model
from alphagenome_research.model.metadata import metadata as metadata_lib
from alphagenome_research.io import fasta as fasta_lib
from alphagenome_research.model import one_hot_encoder

from utils.borzoi_loss import (
    cut_and_tag_loss_alphagenome,
    compute_track_mean_from_bigwig,
)
from utils.lora_utils import (
    initialize_lora_params,
    apply_lora_to_params,
    get_attention_qkv_layers,
    count_lora_parameters,
)
from dataset.tf_dataset import CutAndTagDataPipelineTF

import json
import csv
import datetime
import contextlib
import sys as _sys


# ---------------------------------------------------------------------------
# Config / logging helpers
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
# Data pipeline  (identical to tf_ft_freeze.py)
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
        chrom_sizes_path: Path to a 2-column TSV (chrom\\tsize).
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

    def __init__(self, fasta_path, signal_paths, windows_bed, window_size,
                 organism_id, chrom_sizes_path=None,
                 exclude_chroms=None, include_chroms=None):
        self.fasta_extractor = fasta_lib.FastaExtractor(fasta_path)
        if isinstance(signal_paths, str):
            signal_paths = [signal_paths]
        self.signal_extractors = [
            CutAndTagDataPipelineTF(p, chrom_sizes_path) for p in signal_paths
        ]
        self.num_tracks = len(self.signal_extractors)
        self.window_size = window_size
        self.organism_id = organism_id
        self.one_hot_encoder = one_hot_encoder.DNAOneHotEncoder()

        if windows_bed is not None:
            print(f"   Loading training windows from {windows_bed}...")
            self.intervals = pd.read_csv(
                windows_bed, sep='\t', header=None,
                names=['chrom', 'start', 'end'], comment='#'
            )
            self.intervals['start'] = self.intervals['start'].astype(int)
            self.intervals['end'] = self.intervals['end'].astype(int)
            print(f"   ✓ Loaded {len(self.intervals)} windows")
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

        # Chromosome-based filtering for train/val/test splits
        if exclude_chroms:
            before = len(self.intervals)
            self.intervals = self.intervals[
                ~self.intervals['chrom'].isin(exclude_chroms)
            ].reset_index(drop=True)
            print(f"   Excluded {before - len(self.intervals)} windows on "
                  f"{exclude_chroms} → {len(self.intervals)} remaining")
        if include_chroms:
            before = len(self.intervals)
            self.intervals = self.intervals[
                self.intervals['chrom'].isin(include_chroms)
            ].reset_index(drop=True)
            print(f"   Kept {len(self.intervals)} windows on {include_chroms} "
                  f"(dropped {before - len(self.intervals)})")

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
        signals = []
        for extractor in self.signal_extractors:
            try:
                s = extractor.get_signal(chrom, start, end)
                if s is None or len(s) != self.window_size:
                    s = np.zeros(self.window_size, dtype=np.float32)
                else:
                    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            except Exception as e:
                print(f"Warning: signal {chrom}:{start}-{end}: {e}")
                s = np.zeros(self.window_size, dtype=np.float32)
            signals.append(s)
        signal = np.stack(signals, axis=-1)  # [window_size, N]
        return {
            'dna_sequence': seq_one_hot,
            'organism_index': self.organism_id,
            'signal': signal,
        }

    def get_output_signature(self):
        return {
            'dna_sequence': tf.TensorSpec(shape=(self.window_size, 4), dtype=tf.float32),
            'organism_index': tf.TensorSpec(shape=[], dtype=tf.int32),
            'signal': tf.TensorSpec(shape=(self.window_size, self.num_tracks), dtype=tf.float32),
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
        for extractor in self.signal_extractors:
            extractor.close()


# ---------------------------------------------------------------------------
# HybridAlphaGenomeModel
# ---------------------------------------------------------------------------

class HybridAlphaGenomeModel:
    """
    Two-phase hybrid fine-tuning:

    Phase 1 — trunk frozen, head-only training
        Only the new ATAC head (~9 K params) receives gradients.
        Per-step time: ~0.15 s.

    Phase 2 — LoRA on trunk, head + LoRA trained jointly
        Trunk base weights stay frozen.  LoRA A/B matrices for the Q/K/V
        projections of all transformer blocks (~428 K params at rank 8) are
        trained together with the head.  Trunk is never directly modified.
        Per-step time: intermediate (~1–5 s estimated, much less than full).

    Call ``model.activate_lora()`` to transition from Phase 1 to Phase 2.
    """

    def __init__(
        self,
        params: Dict,
        state: Dict,
        apply_fn: Callable,
        mesh: Optional[Mesh] = None,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.04,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_init_std: float = None,
        warmup_steps: int = 0,
        total_steps: int = 0,
        freeze_warmup_steps: int = 0,
        freeze_total_steps: int = 0,
    ):
        """
        Args:
            params:              Merged params (pretrained trunk + fresh head).
            state:               Model state (BatchNorm EMAs).
            apply_fn:            ``apply_fn(params, state, seq, org) -> predictions``.
            mesh:                JAX device mesh for data-parallel training.
            learning_rate:       Peak AdamW LR used in both phases.
            weight_decay:        AdamW weight decay (paper pre-training: 0.4,
                                 distillation: 0.04; default 0.04 for fine-tuning).
            lora_rank:           LoRA rank for Phase 2 (default 8).
            lora_alpha:          LoRA alpha scaling for Phase 2 (default 16.0).
            lora_init_std:       If set, initialize both LoRA A and B matrices as
                                 N(0, lora_init_std).  If None (default), uses the
                                 standard asymmetric init (Kaiming A, zeros B).
            warmup_steps:        Warmup steps for Phase 2 cosine schedule.
            total_steps:         Total steps for Phase 2 cosine schedule.
            freeze_warmup_steps: Warmup steps for Phase 1 cosine schedule.
            freeze_total_steps:  Total steps for Phase 1 cosine schedule.
        """
        self.apply_fn = apply_fn
        self.state = state
        self.mesh = mesh
        self.num_devices = jax.device_count()
        self._phase = 'frozen'  # 'frozen' | 'lora'

        # Guard against YAML parsing 5e-4 as a string on some PyYAML versions
        learning_rate        = float(learning_rate)
        weight_decay         = float(weight_decay)
        lora_rank            = int(lora_rank)
        lora_alpha           = float(lora_alpha)
        warmup_steps         = int(warmup_steps)
        total_steps          = int(total_steps)
        freeze_warmup_steps  = int(freeze_warmup_steps)
        freeze_total_steps   = int(freeze_total_steps)

        # ── Split trunk and head ──────────────────────────────────────────
        self.head_params = hk.data_structures.filter(
            lambda module_name, *_: 'head' in module_name, params
        )
        self.trunk_params = hk.data_structures.filter(
            lambda module_name, *_: 'head' not in module_name, params
        )
        n_head  = sum(x.size for x in jax.tree_util.tree_leaves(self.head_params))
        n_trunk = sum(x.size for x in jax.tree_util.tree_leaves(self.trunk_params))
        n_total = n_head + n_trunk

        # ── Initialize LoRA params for QKV layers in all transformer blocks ─
        target_layers = get_attention_qkv_layers()
        lora_init = initialize_lora_params(
            base_params=self.trunk_params,
            target_layers=target_layers,
            rank=lora_rank,
            alpha=lora_alpha,
            init_std=lora_init_std,
        )
        # Trainable arrays — separated from static metadata
        self.lora_A      = lora_init['lora_A']        # dict[str, array]
        self.lora_B      = lora_init['lora_B']        # dict[str, array]
        self._lora_scaling      = lora_init['scaling']       # float (static)
        self._lora_target_layers = lora_init['target_layers'] # list[str] (static)
        self._lora_rank  = lora_rank
        self._lora_alpha = lora_alpha

        n_lora = count_lora_parameters(lora_init)

        # ── Phase 1 optimizer: head only ──────────────────────────────────
        if freeze_warmup_steps > 0 and freeze_total_steps > 0:
            freeze_lr = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=freeze_warmup_steps,
                decay_steps=freeze_total_steps,
                end_value=0.0,
            )
            print(f"  [Phase 1] LR warmup: {freeze_warmup_steps} steps, "
                  f"cosine over {freeze_total_steps} steps")
        else:
            freeze_lr = learning_rate

        self._head_optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adamw(freeze_lr, weight_decay=weight_decay),
        )
        self._head_opt_state = self._head_optimizer.init(self.head_params)

        # ── Phase 2 optimizer: head + LoRA (init lazily on activate_lora) ─
        if warmup_steps > 0 and total_steps > 0:
            lora_lr = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=total_steps,
                end_value=0.0,
            )
            print(f"  [Phase 2] LR warmup: {warmup_steps} steps, "
                  f"cosine over {total_steps} steps")
        else:
            lora_lr = learning_rate

        self._lora_optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adamw(lora_lr, weight_decay=weight_decay),
        )
        self._lora_opt_state = None  # Created in activate_lora()

        # ── JIT-compile steps ─────────────────────────────────────────────
        if mesh is not None:
            _data = P('data')
            _repl = P()

            # Phase 1: frozen trunk, head-only grad
            # trunk_params lives in Python closure — not a JIT argument
            self._frozen_train_step_jit = jax.jit(
                self._frozen_train_step,
                static_argnames=['loss_fn'],
                in_shardings=(
                    _data,  # dna_sequence
                    _data,  # organism_index
                    _data,  # targets
                    _repl,  # head_params
                    _repl,  # head_opt_state
                ),
                # loss_value, per_track_losses, new_head_params, new_head_opt_state
                out_shardings=(_repl, _repl, _repl, _repl),
            )

            # Val step (Phase 1): forward-only, trunk in closure — same
            # memory pattern as training (trunk NOT passed as JIT arg).
            self._frozen_val_step_jit = jax.jit(
                self._frozen_val_step,
                static_argnames=['loss_fn'],
                in_shardings=(
                    _data,  # dna_sequence
                    _data,  # organism_index
                    _data,  # targets
                    _repl,  # head_params
                ),
                out_shardings=(_repl, _repl),  # total_loss, per_track_losses
            )

            # Val step (Phase 2): forward-only, trunk in closure, lora small args.
            self._lora_val_step_jit = jax.jit(
                self._lora_val_step,
                static_argnames=['loss_fn'],
                in_shardings=(
                    _data,  # dna_sequence
                    _data,  # organism_index
                    _data,  # targets
                    _repl,  # trainable: {'head', 'lora_A', 'lora_B'}
                ),
                out_shardings=(_repl, _repl),  # total_loss, per_track_losses
            )

            # Phase 2: head + lora_A + lora_B gradients
            # trunk_params still in closure
            self._lora_train_step_jit = jax.jit(
                self._lora_train_step,
                static_argnames=['loss_fn'],
                in_shardings=(
                    _data,  # dna_sequence
                    _data,  # organism_index
                    _data,  # targets
                    _repl,  # trainable: {'head', 'lora_A', 'lora_B'}
                    _repl,  # opt_state
                ),
                # loss_value, per_track_losses, new_trainable, new_opt_state
                out_shardings=(_repl, _repl, _repl, _repl),
            )
        else:
            self._frozen_train_step_jit = jax.jit(
                self._frozen_train_step, static_argnames=['loss_fn']
            )
            self._lora_train_step_jit = jax.jit(
                self._lora_train_step, static_argnames=['loss_fn']
            )
            self._frozen_val_step_jit = jax.jit(
                self._frozen_val_step, static_argnames=['loss_fn']
            )
            self._lora_val_step_jit = jax.jit(
                self._lora_val_step, static_argnames=['loss_fn']
            )

        print(f"✓ HybridAlphaGenomeModel initialized")
        print(f"  Trunk params (always frozen base): {n_trunk:,}  "
              f"({100*n_trunk/n_total:.1f}%)")
        print(f"  Head  params (Phase 1+2 trainable): {n_head:,}  "
              f"({100*n_head/n_total:.1f}%)")
        print(f"  LoRA  params (Phase 2 trainable):  {n_lora:,}  "
              f"(rank={lora_rank}, alpha={lora_alpha})")
        print(f"  Phase 2 trainable total:           {n_head + n_lora:,}  "
              f"({100*(n_head+n_lora)/n_total:.2f}% of all params)")
        if self.num_devices > 1:
            print(f"  Multi-GPU: {self.num_devices} devices")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_lora_inline(self, trunk_params, lora_A, lora_B):
        """
        Apply LoRA deltas to trunk_params using self._lora_* metadata.

        This is a pure JAX function — safe to call inside value_and_grad.
        lora_A / lora_B are differentiable JAX arrays; scaling and
        target_layers are Python scalars/lists captured in the closure.
        """
        scaling = self._lora_scaling
        target_layers = self._lora_target_layers
        # Deep copy via tree_map so we don't mutate trunk_params
        merged = jax.tree_util.tree_map(lambda x: x, trunk_params)
        for layer_path in target_layers:
            w = merged[layer_path]['w']
            a = lora_A[layer_path].astype(jnp.float32)
            b = lora_B[layer_path].astype(jnp.float32)
            delta = (a @ b * scaling).astype(w.dtype)
            merged[layer_path] = dict(merged[layer_path], w=w + delta)
        return merged

    def _apply_lora_numpy(self, trunk_params, lora_A, lora_B):
        """
        Apply LoRA deltas using numpy — safe to call from save() inside a
        mesh context because device_get() pulls all arrays to host first,
        avoiding the single-device vs. multi-device mesh conflict.
        """
        import numpy as np
        scaling = self._lora_scaling
        # Pull everything to host numpy
        trunk_host = jax.device_get(trunk_params)
        lora_A_host = jax.device_get(lora_A)
        lora_B_host = jax.device_get(lora_B)
        # Deep copy (already numpy after device_get)
        merged = jax.tree_util.tree_map(lambda x: np.array(x), trunk_host)
        for layer_path in self._lora_target_layers:
            w = merged[layer_path]['w']
            a = lora_A_host[layer_path].astype(np.float32)
            b = lora_B_host[layer_path].astype(np.float32)
            delta = (a @ b * scaling).astype(w.dtype)
            merged[layer_path] = dict(merged[layer_path], w=w + delta)
        return merged

    # ------------------------------------------------------------------
    # Phase 1: frozen trunk, head-only gradient
    # ------------------------------------------------------------------

    def _frozen_train_step(
        self,
        dna_sequence,
        organism_index,
        targets,
        head_params,
        head_opt_state,
        loss_fn: Callable,
    ):
        trunk_params = self.trunk_params  # closure — not differentiated

        def compute_loss(hp):
            full_params = hk.data_structures.merge(trunk_params, hp)
            predictions = self.apply_fn(
                full_params, self.state, dna_sequence, organism_index
            )
            return loss_fn(predictions, targets['signal'])

        # loss_fn returns (total_loss, per_track_losses); has_aux=True treats
        # the second element as auxiliary data not differentiated through.
        (loss_value, per_track_losses), head_grads = jax.value_and_grad(
            compute_loss, has_aux=True
        )(head_params)
        updates, new_head_opt_state = self._head_optimizer.update(
            head_grads, head_opt_state, head_params
        )
        new_head_params = optax.apply_updates(head_params, updates)
        return loss_value, per_track_losses, new_head_params, new_head_opt_state

    def _frozen_train_wrapper(self, dna_sequence, organism_index, targets, loss_fn):
        loss_value, per_track_losses, new_head_params, new_head_opt_state = (
            self._frozen_train_step_jit(
                dna_sequence, organism_index, targets,
                self.head_params, self._head_opt_state, loss_fn,
            )
        )
        self.head_params = new_head_params
        self._head_opt_state = new_head_opt_state
        return loss_value, per_track_losses

    # ------------------------------------------------------------------
    # Phase 2: head + LoRA, trunk base stays frozen
    # ------------------------------------------------------------------

    def _lora_train_step(
        self,
        dna_sequence,
        organism_index,
        targets,
        trainable: Dict,   # {'head': head_params, 'lora_A': ..., 'lora_B': ...}
        opt_state,
        loss_fn: Callable,
    ):
        trunk_params = self.trunk_params  # closure — not differentiated

        def compute_loss(t):
            lora_trunk = self._apply_lora_inline(trunk_params, t['lora_A'], t['lora_B'])
            full_params = hk.data_structures.merge(lora_trunk, t['head'])
            predictions = self.apply_fn(
                full_params, self.state, dna_sequence, organism_index
            )
            return loss_fn(predictions, targets['signal'])

        (loss_value, per_track_losses), grads = jax.value_and_grad(
            compute_loss, has_aux=True
        )(trainable)
        updates, new_opt_state = self._lora_optimizer.update(grads, opt_state, trainable)
        new_trainable = optax.apply_updates(trainable, updates)
        return loss_value, per_track_losses, new_trainable, new_opt_state

    def _lora_train_wrapper(self, dna_sequence, organism_index, targets, loss_fn):
        trainable = {
            'head':   self.head_params,
            'lora_A': self.lora_A,
            'lora_B': self.lora_B,
        }
        loss_value, per_track_losses, new_trainable, new_opt_state = (
            self._lora_train_step_jit(
                dna_sequence, organism_index, targets,
                trainable, self._lora_opt_state, loss_fn,
            )
        )
        self.head_params  = new_trainable['head']
        self.lora_A       = new_trainable['lora_A']
        self.lora_B       = new_trainable['lora_B']
        self._lora_opt_state = new_opt_state
        return loss_value, per_track_losses

    # ------------------------------------------------------------------
    # Validation: forward-only (no gradient)
    # ------------------------------------------------------------------

    def _frozen_val_step(self, dna_sequence, organism_index, targets, head_params, loss_fn):
        """Pure forward pass for validation (Phase 1) — trunk in closure, no grad."""
        trunk_params = self.trunk_params  # closure — same pattern as training
        full_params = hk.data_structures.merge(trunk_params, head_params)
        predictions = self.apply_fn(full_params, self.state, dna_sequence, organism_index)
        return loss_fn(predictions, targets['signal'])

    def _lora_val_step(self, dna_sequence, organism_index, targets, trainable, loss_fn):
        """Pure forward pass for validation (Phase 2) — trunk in closure, lora applied."""
        trunk_params = self.trunk_params  # closure — base weights never replicated
        lora_trunk = self._apply_lora_inline(trunk_params, trainable['lora_A'], trainable['lora_B'])
        full_params = hk.data_structures.merge(lora_trunk, trainable['head'])
        predictions = self.apply_fn(full_params, self.state, dna_sequence, organism_index)
        return loss_fn(predictions, targets['signal'])

    def val_step(self, dna_sequence, organism_index, targets, loss_fn):
        """Compute validation loss for one batch. Trunk stays in closure.

        Returns:
            (total_loss, per_track_losses) — mirrors train_step return signature.
        """
        if self._phase == 'frozen':
            return self._frozen_val_step_jit(
                dna_sequence, organism_index, targets, self.head_params, loss_fn
            )
        else:
            trainable = {
                'head':   self.head_params,
                'lora_A': self.lora_A,
                'lora_B': self.lora_B,
            }
            return self._lora_val_step_jit(
                dna_sequence, organism_index, targets, trainable, loss_fn
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def phase(self) -> str:
        """Returns 'frozen' or 'lora'."""
        return self._phase

    def activate_lora(self):
        """
        Transition from Phase 1 (head-only) to Phase 2 (head + LoRA).

        Initialises a fresh combined optimizer over
        {'head': head_params, 'lora_A': ..., 'lora_B': ...}.
        No optimizer momentum bleeds from Phase 1 to Phase 2.
        """
        if self._phase == 'lora':
            print("  [activate_lora] Already in LoRA phase — no-op.")
            return
        self._phase = 'lora'
        trainable = {
            'head':   self.head_params,
            'lora_A': self.lora_A,
            'lora_B': self.lora_B,
        }
        self._lora_opt_state = self._lora_optimizer.init(trainable)
        n_lora   = sum(x.size for x in jax.tree_util.tree_leaves(self.lora_A))
        n_lora  += sum(x.size for x in jax.tree_util.tree_leaves(self.lora_B))
        n_head   = sum(x.size for x in jax.tree_util.tree_leaves(self.head_params))
        print(f"  ✓ LoRA phase activated — now training {n_head:,} head params "
              f"+ {n_lora:,} LoRA params ({len(self._lora_target_layers)} layers, "
              f"rank={self._lora_rank})")
        print(f"    (Fresh optimizer state — no momentum bleed from Phase 1)")

    def train_step(
        self,
        dna_sequence,
        organism_index,
        targets: Dict,
        loss_fn: Callable,
    ):
        """Unified interface: dispatches to frozen or LoRA step automatically.

        Returns:
            (loss_value, per_track_losses) — loss_value is a scalar; 
            per_track_losses is a JAX array of shape [N] (one entry per track).
        """
        if self._phase == 'frozen':
            loss_value, per_track_losses = self._frozen_train_wrapper(
                dna_sequence, organism_index, targets, loss_fn
            )
        else:
            loss_value, per_track_losses = self._lora_train_wrapper(
                dna_sequence, organism_index, targets, loss_fn
            )
        return loss_value, per_track_losses

    def save(self, dirpath: str):
        """
        Save checkpoint via orbax.

        The saved format is compatible with ``dna_model.create()`` loading:
        trunk params have LoRA deltas merged in (not stored separately as a
        diff), plus the fresh ATAC head.  LoRA raw A/B matrices are also
        saved alongside as ``lora_raw/`` for reproducibility.

        In frozen phase: trunk has no LoRA (B is zero-initialized — equivalent
        to no change), so the save is identical to FreezeHeadReplaceAlphaGenomeModel.
        """
        if os.path.exists(dirpath):
            shutil.rmtree(dirpath)
        os.makedirs(dirpath, exist_ok=True)
        checkpointer = ocp.StandardCheckpointer()

        # Merge LoRA into trunk params before saving.
        # Use numpy path (_apply_lora_numpy) to avoid the JAX mesh-context
        # conflict: trunk_params lives on GPU 0 only (loaded before the mesh
        # was established), while inside `with jax.set_mesh(mesh)` JAX expects
        # all arrays to be sharded across all devices.
        lora_trunk = self._apply_lora_numpy(
            self.trunk_params, self.lora_A, self.lora_B
        )
        merged_params = hk.data_structures.merge(lora_trunk, self.head_params)

        # Primary checkpoint: compatible with dna_model.create() API.
        # Must be a list [params, state] — the original kagglehub checkpoints
        # are stored this way so that `params, state = checkpointer.restore(path)`
        # unpacks correctly (dna_model.create() uses that exact pattern).
        #
        # IMPORTANT: jax.device_get() gathers all arrays to numpy before saving.
        # Without this, arrays have 8-GPU NamedSharding baked into the orbax
        # metadata, which (a) forces all 8 GPUs to be visible at load time and
        # (b) causes a slow cross-GPU gather before single-GPU inference.
        checkpointer.save(
            os.path.join(dirpath, 'model'),
            jax.device_get([merged_params, self.state])
        )

        # Secondary checkpoint: raw LoRA A/B for Phase 2 resumption
        lora_raw_dir = os.path.join(dirpath, 'lora_raw')
        checkpointer.save(
            lora_raw_dir,
            {'lora_A': self.lora_A, 'lora_B': self.lora_B}
        )

        checkpointer.wait_until_finished()

        # Save LoRA metadata as JSON for human inspection
        with open(os.path.join(dirpath, 'lora_metadata.json'), 'w') as f:
            json.dump({
                'rank':          self._lora_rank,
                'alpha':         self._lora_alpha,
                'scaling':       self._lora_scaling,
                'target_layers': self._lora_target_layers,
                'phase':         self._phase,
            }, f, indent=2)

    def load(self, dirpath: str):
        """
        Restore from a checkpoint saved by ``save()``.

        Re-splits trunk/head, re-initialises optimizers from scratch.
        """
        checkpointer = ocp.StandardCheckpointer()
        restored = checkpointer.restore(os.path.join(dirpath, 'model'))
        # Checkpoint saved as [params, state] list (compatible with kagglehub format)
        all_params, self.state = restored

        self.head_params = hk.data_structures.filter(
            lambda module_name, *_: 'head' in module_name, all_params
        )
        self.trunk_params = hk.data_structures.filter(
            lambda module_name, *_: 'head' not in module_name, all_params
        )

        # Restore phase from lora_metadata.json (written by save())
        meta_path = os.path.join(dirpath, 'lora_metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            saved_phase = meta.get('phase', 'frozen')
            if saved_phase == 'lora' and self._phase != 'lora':
                # Re-activate LoRA so that lora_A/B matrices are initialised
                # before we overwrite them from the checkpoint below.
                self.activate_lora()

        # Restore raw LoRA arrays if Phase 2 checkpoint exists
        lora_raw_dir = os.path.join(dirpath, 'lora_raw')
        if os.path.exists(lora_raw_dir):
            lora_restored = checkpointer.restore(lora_raw_dir)
            self.lora_A = lora_restored['lora_A']
            self.lora_B = lora_restored['lora_B']

        # Re-initialise optimizers
        self._head_opt_state = self._head_optimizer.init(self.head_params)
        if self._phase == 'lora':
            trainable = {
                'head':   self.head_params,
                'lora_A': self.lora_A,
                'lora_B': self.lora_B,
            }
            self._lora_opt_state = self._lora_optimizer.init(trainable)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    checkpoint_path: str,
    fasta_path: str,
    signal_paths,
    output_prefix: str,
    strategy: str = 'head_replace',
    epochs: int = 10,
    freeze_epochs: int = 2,
    batch_size: int = 8,
    learning_rate: float = 5e-4,
    window_size: int = 8192,
    organism_id: int = 1,
    genome_windows: str = None,
    augment: bool = True,
    max_shift: int = 1024,
    num_workers: int = 0,
    precision_policy: str = 'params=float32,compute=float32,output=float32',
    config_path: str = None,
    chrom_sizes_path: str = None,
    checkpoint_epoch_interval: int = 1,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_init_std: float = None,
    weight_decay: float = 0.04,
    val_chroms: list = None,
    test_chroms: list = None,
    resume_checkpoint_dir: str = None,
    resume_start_epoch: int = 0,
):
    """Fine-tune AlphaGenome with the two-phase hybrid curriculum.

    Args:
        checkpoint_epoch_interval: Epoch-level checkpoint save interval.
            Saves ``epoch_NNN/`` when ``epoch % checkpoint_epoch_interval == 0``.
            Set to 0 to disable epoch checkpoints (best checkpoint is still saved).
    """

    if val_chroms is None:
        val_chroms = []
    if test_chroms is None:
        test_chroms = []

    # ── Logging setup ───────────────────────────────────────────────────────
    _ts = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = f"{output_prefix}_hybrid_freeze{freeze_epochs}_{_ts}"
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
    # Header is written after num_tracks is resolved (see below).

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

    lora_phase_epochs = max(0, epochs - freeze_epochs)

    checkpoint_path = os.path.abspath(checkpoint_path)
    fasta_path      = os.path.abspath(fasta_path)
    output_dir      = os.path.abspath(output_dir)

    if isinstance(signal_paths, str):
        signal_paths = [signal_paths]
    signal_paths  = [os.path.abspath(p) for p in signal_paths]
    num_tracks    = len(signal_paths)
    signal_format = 'BigWig' if signal_paths[0].endswith(('.bigwig', '.bw')) else 'BedGraph'

    # Write metrics CSV header now that num_tracks is known.
    if os.path.getsize(metrics_path) == 0:
        per_track_cols = [f'track_{i}_loss' for i in range(num_tracks)]
        _metrics_writer.writerow(
            ['timestamp', 'epoch', 'batch', 'loss', 'phase', 'nan_event', 'skipped']
            + per_track_cols
        )

    print("=" * 60)
    print(f"FINE-TUNING SETUP  [hybrid curriculum]")
    print("=" * 60)
    print(f"Precision policy : {precision_policy}")
    print(f"Window size      : {window_size:,} bp (2^{int(np.log2(window_size))})")
    print(f"Phase 1 (frozen) : {freeze_epochs} epoch(s)  — head only (~0.15 s/step)")
    print(f"Phase 2 (LoRA)   : {lora_phase_epochs} epoch(s)  — head + LoRA "
          f"(rank={lora_rank}, alpha={lora_alpha})")
    print(f"Ckpt epoch intvl : {checkpoint_epoch_interval} "
          f"(0=disabled, saves epoch_NNN every N epochs)")
    print(f"Signal format    : {signal_format}")
    print(f"Tracks           : {num_tracks}")
    for p in signal_paths:
        print(f"   {os.path.basename(p)}")

    if genome_windows:
        genome_windows = os.path.abspath(genome_windows)
    else:
        print("   Note: windows_bed not set — "
              "whole-genome windows will be generated from chrom_sizes")

    # ── Data pipeline ───────────────────────────────────────────────────────
    print(f"\n1. Setting up TensorFlow data pipeline...")

    # Chromosome-based train/val/test split
    train_exclude = list(val_chroms) + list(test_chroms)
    if train_exclude:
        print(f"   Train split: excluding val chroms {list(val_chroms)} "
              f"and test chroms {list(test_chroms)}")
    print(f"   Building training pipeline...")
    train_pipeline = CutAndTagDataPipeline(
        fasta_path=fasta_path,
        signal_paths=signal_paths,
        windows_bed=genome_windows,
        window_size=window_size,
        organism_id=organism_id,
        chrom_sizes_path=chrom_sizes_path,
        exclude_chroms=train_exclude or None,
    )
    train_tf_dataset = (
        tf.data.Dataset.from_generator(
            train_pipeline.get_generator,
            output_signature=train_pipeline.get_output_signature(),
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Validation pipeline (if val chroms provided)
    if val_chroms:
        print(f"   Building validation pipeline ({list(val_chroms)})...")
        val_pipeline = CutAndTagDataPipeline(
            fasta_path=fasta_path,
            signal_paths=signal_paths,
            windows_bed=genome_windows,
            window_size=window_size,
            organism_id=organism_id,
            chrom_sizes_path=chrom_sizes_path,
            include_chroms=list(val_chroms),
        )
        if len(val_pipeline.intervals) > 0:
            val_tf_dataset = (
                tf.data.Dataset.from_generator(
                    lambda: val_pipeline.get_generator(shuffle=False),
                    output_signature=val_pipeline.get_output_signature(),
                )
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            steps_per_val = len(val_pipeline.intervals) // batch_size
            print(f"   ✓ Val pipeline: {len(val_pipeline.intervals)} windows, "
                  f"{steps_per_val} steps")
        else:
            print(f"   ⚠ No intervals found for val_chroms {list(val_chroms)} — "
                  f"skipping validation")
            val_tf_dataset = None
            steps_per_val = 0
    else:
        print(f"   No val_chroms set — validation disabled (best ckpt by train loss)")
        val_tf_dataset = None
        steps_per_val = 0

    num_train_windows = len(train_pipeline.intervals)
    steps_per_epoch   = num_train_windows // batch_size

    print(f"   Computing track_means from BigWig(s) (for AlphaGenome targets_scaling)...")
    track_means_list = []
    for p in signal_paths:
        tm = compute_track_mean_from_bigwig(p, chrom_sizes_path=chrom_sizes_path)
        print(f"   track_mean = {tm:.6f}  ({os.path.basename(p)})")
        track_means_list.append(tm)
    track_means = np.array(track_means_list, dtype=np.float32)  # shape [N]

    # LR schedules
    freeze_phase_steps  = steps_per_epoch * max(1, freeze_epochs)
    freeze_warmup_steps = int(0.05 * freeze_phase_steps)

    lora_total_steps    = steps_per_epoch * max(1, lora_phase_epochs)
    lora_warmup_steps   = int(0.05 * lora_total_steps)

    print(f"   Train windows   : {num_train_windows}, {steps_per_epoch} steps/epoch")
    print(f"   Phase 1 (frozen): {freeze_epochs} epoch(s), "
          f"{freeze_phase_steps} steps, {freeze_warmup_steps} warmup")
    print(f"   Phase 2 (LoRA)  : {lora_phase_epochs} epoch(s), "
          f"{lora_total_steps} steps, {lora_warmup_steps} warmup")

    # ── Load base model ─────────────────────────────────────────────────────
    print(f"\n2. Loading base model from {checkpoint_path}...")
    # Important: do not use default_organism_settings() here, because it points
    # to remote GTF/PAS/splice Feather files and can block on network I/O.
    # Training only needs params/state + local FASTA; annotation files are not
    # required for the fine-tuning loop.
    train_organism = (
        dna_model.Organism.HOMO_SAPIENS if organism_id == 0
        else dna_model.Organism.MUS_MUSCULUS
    )
    base_model = dna_model.create(
        checkpoint_path=checkpoint_path,
        organism_settings={
            train_organism: dna_model.OrganismSettings(
                metadata=None,
                fasta_path=fasta_path,
                gtf_feather_path=None,
                pas_feather_path=None,
                splice_site_starts_feather_path=None,
                splice_site_ends_feather_path=None,
            )
        },
        device=None,
    )
    print(f"   ✓ Model loaded on {base_model._device_context._device}")

    # ── Build the model with a fresh ATAC head ──────────────────────────────
    mesh_ctx = jax.set_mesh(mesh) if mesh is not None else contextlib.nullcontext()

    print(f"\n3. Building new ATAC output head ({num_tracks} track(s))...")
    organism_enum = (
        dna_model.Organism.HOMO_SAPIENS if organism_id == 0
        else dna_model.Organism.MUS_MUSCULUS
    )
    track_df = pd.DataFrame([
        {
            'output_type': 'ATAC',
            'name': os.path.splitext(os.path.basename(p))[0],
            'strand': '.',
            'file_path': p,
        }
        for p in signal_paths
    ])
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

    model = HybridAlphaGenomeModel(
        params=merged_params,
        state=base_model._state,
        apply_fn=new_apply_fn,
        mesh=mesh,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_init_std=lora_init_std,
        warmup_steps=lora_warmup_steps,
        total_steps=lora_total_steps,
        freeze_warmup_steps=freeze_warmup_steps,
        freeze_total_steps=freeze_phase_steps,
    )

    # ── Resume from checkpoint if requested ────────────────────────────────
    if resume_checkpoint_dir:
        print(f"\n   [resume] Loading checkpoint from: {resume_checkpoint_dir}")
        model.load(resume_checkpoint_dir)
        print(f"   [resume] ✓ Restored params/state/LoRA  "
              f"(phase={model.phase}, start_epoch={resume_start_epoch + 1})")

    # ── Training loop ───────────────────────────────────────────────────────
    print(f"\n4. Starting training for {epochs} epochs  "
          f"[freeze_epochs={freeze_epochs}, lora_epochs={lora_phase_epochs}]...")
    # Bind track_means into a plain closure — must be a numpy/JAX array for
    # multi-track broadcasting in alphagenome_targets_scaling.
    def loss_fn(predictions, targets):
        return cut_and_tag_loss_alphagenome(predictions, targets, track_means=track_means)
    best_loss = float('inf')

    with mesh_ctx:
        # Persistent iterator — created ONCE so data order advances continuously
        # across epochs (fixes the periodic-valley bug caused by restarting the
        # generator with seed=0 at every epoch).
        ds_iter = iter(train_tf_dataset)

        for epoch in range(resume_start_epoch, epochs):
            # ── Phase transition ──────────────────────────────────────────
            if epoch == freeze_epochs and model.phase == 'frozen':
                print(f"\n{'=' * 60}")
                print(f"PHASE TRANSITION: Epoch {epoch + 1} — activating LoRA")
                print(f"{'=' * 60}")
                model.activate_lora()
                print(f"  (LoRA JIT will recompile on first batch of this epoch)")

            phase_label = (
                " [HEAD ONLY — TRUNK FROZEN]" if model.phase == 'frozen'
                else f" [HEAD + LoRA rank={lora_rank}]"
            )
            print(f"\nEpoch {epoch + 1}/{epochs}{phase_label}")
            print("-" * 60)

            epoch_losses = []

            for batch_idx, batch in enumerate(
                itertools.islice(ds_iter, steps_per_epoch)
            ):
                if batch_idx == 0:
                    print(f"  Compiling JAX functions "
                          f"(this may take 2–5 minutes on first batch)...")
                    start_time = time.time()

                sequences    = jnp.array(batch['dna_sequence'], dtype=jnp.float32)
                organism_ids = jnp.array(batch['organism_index'], dtype=jnp.int32)
                targets      = {'signal': jnp.array(batch['signal'], dtype=jnp.float32)}

                if data_sharding is not None:
                    sequences    = jax.device_put(sequences,    data_sharding)
                    organism_ids = jax.device_put(organism_ids, data_sharding)
                    targets = {'signal': jax.device_put(targets['signal'], data_sharding)}

                loss_value, per_track_losses = model.train_step(
                    sequences, organism_ids, targets, loss_fn
                )
                loss_host = float(jax.device_get(loss_value))
                per_track_host = np.asarray(
                    jax.device_get(per_track_losses)
                ).flatten().tolist()
                epoch_losses.append(loss_host)

                _metrics_writer.writerow([
                    datetime.datetime.now().isoformat(),
                    epoch + 1, batch_idx + 1,
                    loss_host, model.phase, False, False,
                    *per_track_host,
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
                    track_str = '  '.join(
                        f'track[{i}]={v:.6f}' for i, v in enumerate(per_track_host)
                    )
                    print(
                        f"  Batch {batch_idx + 1}/{steps_per_epoch}: "
                        f"Loss = {avg:.6f}  [{track_str}]"
                    )

            # ── Validation loop ───────────────────────────────────────────
            avg_val_loss = None
            if val_tf_dataset is not None and steps_per_val > 0:
                val_iter = val_tf_dataset.as_numpy_iterator()
                val_losses = []
                for val_batch in itertools.islice(val_iter, steps_per_val):
                    val_seq = jnp.array(val_batch['dna_sequence'], dtype=jnp.float32)
                    val_org = jnp.array(val_batch['organism_index'], dtype=jnp.int32)
                    val_tgt = {'signal': jnp.array(val_batch['signal'], dtype=jnp.float32)}
                    if data_sharding is not None:
                        val_seq = jax.device_put(val_seq, data_sharding)
                        val_org = jax.device_put(val_org, data_sharding)
                        val_tgt = {'signal': jax.device_put(val_tgt['signal'], data_sharding)}
                    val_loss, _val_per_track = model.val_step(
                        val_seq, val_org, val_tgt, loss_fn
                    )
                    val_losses.append(float(jax.device_get(val_loss)))
                avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')

            # ── Epoch summary ─────────────────────────────────────────────
            if epoch_losses:
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                print(f"\nEpoch {epoch + 1} Summary  [{model.phase}]:")
                print(f"  Train Loss   : {avg_epoch_loss:.6f}")
                if avg_val_loss is not None:
                    print(f"  Val Loss     : {avg_val_loss:.6f}  ({steps_per_val} batches)")
                print(f"  Batches      : {len(epoch_losses)}")

                # Best checkpoint: use val loss when available, else train loss
                score = avg_val_loss if avg_val_loss is not None else avg_epoch_loss
                if score < best_loss:
                    best_loss = score
                    best_dir = os.path.join(output_dir, 'best_checkpoint')
                    model.save(best_dir)
                    score_label = 'val_loss' if avg_val_loss is not None else 'train_loss'
                    print(f"  ✓ Saved best checkpoint ({score_label}={best_loss:.6f}) → {best_dir}")
            else:
                avg_epoch_loss = float('inf')
                print(f"\nEpoch {epoch + 1}: ⚠ No batches completed")

            # Epoch-level checkpoint saves (configurable interval).
            if checkpoint_epoch_interval > 0:
                should_save_epoch_ckpt = ((epoch + 1) % checkpoint_epoch_interval == 0)
                if should_save_epoch_ckpt:
                    epoch_dir = os.path.join(output_dir, f'epoch_{epoch + 1:03d}')
                    model.save(epoch_dir)
                    print(f"  [ckpt] epoch save → {epoch_dir}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints: {output_dir}")

    config_out = {
        'strategy':        'hybrid',
        'freeze_epochs':   freeze_epochs,
        'lora_rank':       lora_rank,
        'lora_alpha':      lora_alpha,
        'lora_init_std':   lora_init_std,
        'checkpoint_path': checkpoint_path,
        'precision_policy': precision_policy,
        'learning_rate':   learning_rate,
        'weight_decay':    weight_decay,
        'epochs':          epochs,
        'checkpoint_epoch_interval': checkpoint_epoch_interval,
        'batch_size':      batch_size,
        'window_size':     window_size,
        'best_loss':       best_loss,
        'signal_format':   signal_format,
        'signal_paths':    signal_paths,
        'track_means':     track_means.tolist(),
        'val_chroms':      list(val_chroms),
        'test_chroms':     list(test_chroms),
        'resumed_from':    resume_checkpoint_dir,
        'resumed_from_epoch': resume_start_epoch,
    }
    with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
        json.dump(config_out, f, indent=2)

    _metrics_file.close()
    os.dup2(_saved_stderr_fd, 2)
    os.close(_saved_stderr_fd)
    _sys.stdout = _tee._stream
    _tee.close()


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _resolve_resume_checkpoint(resume_from: str):
    """Resolve a resume-from path to (checkpoint_dir, start_epoch).

    Accepts:
    - An explicit ``epoch_NNN/`` directory  → start_epoch = NNN
    - A run directory (contains epoch_NNN/ sub-dirs) → uses highest epoch sub-dir
    - ``best_checkpoint`` directory inside a run dir
    """
    import re as _re

    if not os.path.isdir(resume_from):
        raise FileNotFoundError(f"--resume-from path not found: {resume_from}")

    # Direct epoch_NNN directory?
    m = _re.search(r'epoch_(\d+)$', resume_from)
    if m and os.path.isdir(os.path.join(resume_from, 'model')):
        return os.path.abspath(resume_from), int(m.group(1))

    # best_checkpoint inside run dir?
    best = os.path.join(resume_from, 'best_checkpoint')
    if os.path.isdir(os.path.join(best, 'model')):
        import glob as _glob
        start_epoch = 0
        epoch_dirs = sorted(_glob.glob(os.path.join(resume_from, 'epoch_*')))
        if epoch_dirs:
            m2 = _re.search(r'epoch_(\d+)$', epoch_dirs[-1])
            if m2:
                start_epoch = int(m2.group(1))
        return os.path.abspath(best), start_epoch

    # Run directory: find highest epoch_NNN/
    import glob as _glob
    epoch_dirs = sorted(_glob.glob(os.path.join(resume_from, 'epoch_*')))
    valid = [d for d in epoch_dirs if os.path.isdir(os.path.join(d, 'model'))]
    if not valid:
        raise FileNotFoundError(
            f"No valid epoch_NNN checkpoint found in: {resume_from}"
        )
    last = valid[-1]
    m3 = _re.search(r'epoch_(\d+)$', last)
    start_epoch = int(m3.group(1)) if m3 else 0
    return os.path.abspath(last), start_epoch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='AlphaGenome CUT&Tag fine-tuning — hybrid head+LoRA curriculum'
    )
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', required=False)
    parser.add_argument('--fasta', required=False)
    parser.add_argument('--signal', required=False)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--precision-policy', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--freeze-epochs', type=int, default=None,
                        help='Epochs to keep trunk fully frozen (Phase 1, head-only). '
                             'Default: 2.')
    parser.add_argument('--lora-rank', type=int, default=None)
    parser.add_argument('--lora-alpha', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--window-size', type=int, default=None)
    parser.add_argument('--organism', type=int, default=None)
    parser.add_argument('--windows-bed', type=str, default=None)
    parser.add_argument('--chrom-sizes', type=str, default=None)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--max-shift', type=int, default=None)
    parser.add_argument('--checkpoint-epoch-interval', type=int, default=None,
                        help='Save epoch_NNN checkpoint every N epochs (0 disables).')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume training from a previous run directory or '
                             'epoch_NNN checkpoint. Training continues from the '
                             'next epoch after the resolved checkpoint.')

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

    # ── Resume-from resolution ───────────────────────────────────────────────
    resume_checkpoint_dir = None
    resume_start_epoch    = 0
    if args.resume_from:
        resume_checkpoint_dir, resume_start_epoch = _resolve_resume_checkpoint(args.resume_from)
        print(f"[resume] Resolved checkpoint : {resume_checkpoint_dir}")
        print(f"[resume] Resuming from epoch : {resume_start_epoch + 1}")
    # ────────────────────────────────────────────────────────────────────────

    checkpoint    = get_value(args.checkpoint,   'checkpoint',    None)
    fasta         = get_value(args.fasta,         'fasta',         None)
    output        = get_value(args.output,        'output',
                              '/home/ray/AG_Finetune/checkpoints/fine_tuned')
    epochs        = get_value(args.epochs,        'epochs',        10)
    freeze_epochs = get_value(args.freeze_epochs, 'freeze_epochs', 2)
    lora_rank     = get_value(args.lora_rank,     'lora_rank',     8)
    lora_alpha    = get_value(args.lora_alpha,    'lora_alpha',    16.0)
    lora_init_std = get_value(None,               'lora_init_std', None)
    weight_decay  = get_value(None,               'weight_decay',  0.04)
    val_chroms    = config.get('val_chroms',  [])
    test_chroms   = config.get('test_chroms', [])
    batch_size    = get_value(args.batch_size,    'batch_size',    8)
    lr            = get_value(args.lr,            'learning_rate', 5e-4)
    window_size   = get_value(args.window_size,   'window_size',   8192)
    organism      = get_value(args.organism,      'organism_id',   1)
    windows_bed   = get_value(args.windows_bed,   'windows_bed',   None)
    chrom_sizes   = get_value(args.chrom_sizes,   'chrom_sizes',   None)
    max_shift     = get_value(args.max_shift,     'max_shift',     1024)
    precision_policy = get_value(
        args.precision_policy, 'precision_policy',
        'params=float32,compute=float32,output=float32'
    )
    augment = False if args.no_augment else config.get('augment', True)
    checkpoint_epoch_interval = get_value(
        args.checkpoint_epoch_interval, 'checkpoint_epoch_interval', 1
    )

    # Resolve signal paths: support both 'signal' (string/list) and 'signals' (list)
    # CLI --signal takes a single path; config 'signals' takes a list.
    cli_signal = getattr(args, 'signal', None)
    if cli_signal:
        signal_paths = [cli_signal]
    elif 'signals' in config:
        signal_paths = config['signals']
        if isinstance(signal_paths, str):
            signal_paths = [signal_paths]
    elif 'signal' in config:
        v = config['signal']
        signal_paths = v if isinstance(v, list) else [v]
    else:
        signal_paths = None

    if not checkpoint:
        raise ValueError("--checkpoint is required")
    if not fasta:
        raise ValueError("--fasta is required")
    if not signal_paths:
        raise ValueError("--signal or config 'signal'/'signals' is required")

    train(
        checkpoint_path=checkpoint,
        fasta_path=fasta,
        signal_paths=signal_paths,
        output_prefix=output,
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
        checkpoint_epoch_interval=checkpoint_epoch_interval,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_init_std=lora_init_std,
        weight_decay=weight_decay,
        val_chroms=val_chroms,
        test_chroms=test_chroms,
        resume_checkpoint_dir=resume_checkpoint_dir,
        resume_start_epoch=resume_start_epoch,
    )


if __name__ == '__main__':
    main()
