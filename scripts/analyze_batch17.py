"""Analyze the first zero/NaN-gradient batch from a single deterministic training pass."""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from torch.utils.data import DataLoader

from alphagenome_research.model import dna_model
from dataset.dataset import CutAndTagDataset
from models.lora_model import LoRAAlphaGenomeModel
from utils.borzoi_loss_v2 import baskerville_target_transform, cut_and_tag_loss_baskerville
from utils.lora_utils import apply_lora_to_params, get_attention_qkv_layers


def collate_to_jax(batch: List[Tuple[np.ndarray, int, Dict]]) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    sequences = [item[0] for item in batch]
    organism_ids = [item[1] for item in batch]
    signals = [item[2]['signal'] for item in batch]
    return (
        jnp.array(sequences, dtype=jnp.float32),
        jnp.array(organism_ids, dtype=jnp.int32),
        {'signal': jnp.array(signals, dtype=jnp.float32)},
    )


def array_stats(name: str, arr: jnp.ndarray):
    arr_f32 = arr.astype(jnp.float32)
    finite_mask = jnp.isfinite(arr_f32)
    finite_count = int(jnp.sum(finite_mask))
    total = arr_f32.size
    nan_count = int(jnp.sum(jnp.isnan(arr_f32)))
    inf_count = int(jnp.sum(jnp.isinf(arr_f32)))
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"  finite={finite_count}/{total}, nan={nan_count}, inf={inf_count}")
    if finite_count > 0:
        finite_vals = arr_f32[finite_mask]
        print(
            f"  min={float(jnp.min(finite_vals)):.6e}, "
            f"max={float(jnp.max(finite_vals)):.6e}, "
            f"mean={float(jnp.mean(finite_vals)):.6e}, "
            f"std={float(jnp.std(finite_vals)):.6e}"
        )


def summarize_prediction_heads(predictions: Dict):
    print("\n1) Prediction head summary (only leaves with NaN/Inf are shown)")
    print("-" * 80)
    found_problem = False
    for head_name, head_val in predictions.items():
        if not isinstance(head_val, dict):
            continue
        for leaf_name, leaf_val in head_val.items():
            if not isinstance(leaf_val, jnp.ndarray):
                continue
            has_nan = bool(jnp.any(jnp.isnan(leaf_val)))
            has_inf = bool(jnp.any(jnp.isinf(leaf_val)))
            if has_nan or has_inf:
                found_problem = True
                print(f"{head_name}.{leaf_name}: has_nan={has_nan}, has_inf={has_inf}, shape={leaf_val.shape}")
    if not found_problem:
        print("No NaN/Inf found in any prediction leaf.")


def summarize_grad_tree(name: str, tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        print(f"{name}: empty tree")
        return

    nan_count = 0
    inf_count = 0
    finite_count = 0
    total_count = 0
    finite_samples = []

    for leaf in leaves:
        leaf_f32 = leaf.astype(jnp.float32)
        total_count += leaf_f32.size
        nan_count += int(jnp.sum(jnp.isnan(leaf_f32)))
        inf_count += int(jnp.sum(jnp.isinf(leaf_f32)))
        mask = jnp.isfinite(leaf_f32)
        finite_count += int(jnp.sum(mask))
        if bool(jnp.any(mask)):
            finite_samples.append(leaf_f32[mask])

    print(f"{name} gradients: total={total_count}, finite={finite_count}, nan={nan_count}, inf={inf_count}")
    if finite_samples:
        finite_vals = jnp.concatenate([x.reshape(-1) for x in finite_samples])
        print(
            f"  finite min={float(jnp.min(finite_vals)):.6e}, "
            f"max={float(jnp.max(finite_vals)):.6e}, "
            f"mean={float(jnp.mean(finite_vals)):.6e}, "
            f"std={float(jnp.std(finite_vals)):.6e}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze first zero/NaN-gradient batch in a single pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for numpy/torch')
    parser.add_argument('--max-batches', type=int, default=80, help='How many batches to scan')
    parser.add_argument('--no-augment', action='store_true', help='Disable dataset augmentation')
    parser.add_argument('--no-shuffle', action='store_true', help='Disable dataloader shuffling')
    parser.add_argument(
        '--precision-policy',
        type=str,
        default='params=float32,compute=bfloat16,output=bfloat16',
        help='JMP precision policy, e.g. params=float32,compute=float32,output=float32',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("ANALYZE FIRST ZERO/NaN-GRAD BATCH (SINGLE PASS)")
    print("=" * 80)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    batch_size = 8
    num_devices = jax.device_count()

    print("\nLoading AlphaGenome...")
    base_model = dna_model.create(
        checkpoint_path='/home/ray/AG_Finetune/checkpoints/pre_trained/alphagenome-all-folds',
        organism_settings=None,
        device=None,
    )

    dataset = CutAndTagDataset(
        fasta_path='/home/ray/AG_Finetune/data/resources/mm10.fa',
        signal_bedgraph='/home/ray/AG_Finetune/data/CUTTag/processed_bedgraph/D1MSN_H3K27ac_averaged_normalized.bedgraph',
        genome_windows=None,
        window_size=131072,
        organism_id=1,
        augment=not args.no_augment,
        max_shift=1024,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not args.no_shuffle,
        num_workers=0,
        collate_fn=collate_to_jax,
        drop_last=True,
    )

    target_layers = get_attention_qkv_layers()
    if num_devices > 1:
        mesh = Mesh(jax.devices(), ('batch',))
        batch_sharding = NamedSharding(mesh, P('batch', None, None))
        organism_sharding = NamedSharding(mesh, P('batch'))
        signal_sharding = NamedSharding(mesh, P('batch', None))
    else:
        mesh = None
        batch_sharding = None
        organism_sharding = None
        signal_sharding = None

    lora_model = LoRAAlphaGenomeModel(
        base_model=base_model,
        lora_rank=8,
        lora_alpha=16.0,
        target_layers=target_layers,
        learning_rate=1e-5,
        mesh=mesh,
        warmup_steps=1300,
        total_steps=26010,
        precision_policy=args.precision_policy,
    )
    loss_fn = cut_and_tag_loss_baskerville

    failing_batch = None
    failing_idx = None

    print(
        f"\nScanning first {args.max_batches} batches "
        f"(seed={args.seed}, augment={not args.no_augment}, shuffle={not args.no_shuffle})..."
    )
    print(f"Precision policy: {args.precision_policy}")
    for batch_idx, (sequences, organism_ids, targets) in enumerate(data_loader):
        if batch_idx >= args.max_batches:
            break

        if batch_sharding is not None:
            sequences = jax.device_put(sequences, batch_sharding)
            organism_ids = jax.device_put(organism_ids, organism_sharding)
            targets = {'signal': jax.device_put(targets['signal'], signal_sharding)}

        loss_value, _, grad_norm = lora_model.train_step(sequences, organism_ids, targets, loss_fn)
        grad_norm_val = float(jax.device_get(grad_norm))
        loss_val = float(jax.device_get(loss_value))

        target_has_nan = bool(jnp.any(jnp.isnan(targets['signal'])))
        print(
            f"Batch {batch_idx + 1:>2}: loss={loss_val:.6f}, grad={grad_norm_val:.6e}, "
            f"target_has_nan={target_has_nan}"
        )

        if grad_norm_val < 1e-10:
            failing_batch = (sequences, organism_ids, targets)
            failing_idx = batch_idx + 1
            break

    if failing_batch is None:
        print(f"\nNo zero/NaN-gradient batch found in first {args.max_batches} batches.")
        return

    sequences, organism_ids, targets = failing_batch
    target_signal = targets['signal']

    print("\n" + "=" * 80)
    print(f"ANALYZING FAILING BATCH {failing_idx}")
    print("=" * 80)

    # Recompute forward on exact same failing batch (current model state)
    full_lora_params = {
        'lora_A': lora_model.lora_params['lora_A'],
        'lora_B': lora_model.lora_params['lora_B'],
        'scaling': lora_model.lora_params['scaling'],
        'rank': lora_model.lora_params['rank'],
        'alpha': lora_model.lora_params['alpha'],
        'target_layers': lora_model.lora_params['target_layers'],
    }
    merged_params = apply_lora_to_params(lora_model.base_params, full_lora_params)
    predictions = lora_model.apply_fn(merged_params, lora_model.base_state, sequences, organism_ids)

    if isinstance(predictions, dict):
        summarize_prediction_heads(predictions)

    print("\n2) ATAC input to loss")
    print("-" * 80)
    if isinstance(predictions, dict) and 'atac' in predictions and 'scaled_predictions_1bp' in predictions['atac']:
        pred_signal = jnp.mean(predictions['atac']['scaled_predictions_1bp'], axis=-1)
        array_stats("ATAC scaled_predictions_1bp(mean over tracks)", pred_signal)
        per_sample_nan = jnp.sum(jnp.isnan(pred_signal), axis=1)
        per_sample_finite = jnp.sum(jnp.isfinite(pred_signal), axis=1)
        print(f"  per-sample NaN counts: {[int(x) for x in per_sample_nan]}")
        print(f"  per-sample finite counts: {[int(x) for x in per_sample_finite]}")
    else:
        print("ATAC scaled_predictions_1bp not found.")

    print("\n3) Target signal")
    print("-" * 80)
    array_stats("targets['signal']", target_signal)
    target_per_sample_nan = jnp.sum(jnp.isnan(target_signal), axis=1)
    print(f"  target per-sample NaN counts: {[int(x) for x in target_per_sample_nan]}")

    print("\n4) Transformed targets")
    print("-" * 80)
    targets_transformed = baskerville_target_transform(
        target_signal,
        scale=1.0,
        clip_soft=64.0,
        clip_hard=384.0,
        use_sqrt=True,
    )
    array_stats("targets_transformed", targets_transformed)

    print("\n5) Loss and raw gradients (pre-sanitization)")
    print("-" * 80)

    def compute_loss_for_grad(lora_trainable):
        full_lora = {
            'lora_A': lora_trainable['lora_A'],
            'lora_B': lora_trainable['lora_B'],
            'scaling': lora_model.lora_params['scaling'],
            'rank': lora_model.lora_params['rank'],
            'alpha': lora_model.lora_params['alpha'],
            'target_layers': lora_model.lora_params['target_layers'],
        }
        merged = apply_lora_to_params(lora_model.base_params, full_lora)
        preds = lora_model.apply_fn(merged, lora_model.base_state, sequences, organism_ids)
        return loss_fn(preds, target_signal)

    lora_trainable = {
        'lora_A': lora_model.lora_params['lora_A'],
        'lora_B': lora_model.lora_params['lora_B'],
    }
    loss_val, grads = jax.value_and_grad(compute_loss_for_grad)(lora_trainable)
    print(f"loss from grad() = {float(loss_val):.6f}, finite={bool(jnp.isfinite(loss_val))}")

    summarize_grad_tree("lora_A", grads['lora_A'])
    summarize_grad_tree("lora_B", grads['lora_B'])

    print("\n" + "=" * 80)
    print("DIAGNOSIS HINT")
    print("=" * 80)
    print(
        "If targets are finite but predictions contain NaN in many heads, the instability "
        "starts in model forward state/activations rather than in dataset collation."
    )


if __name__ == '__main__':
    main()
