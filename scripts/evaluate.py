"""
evaluate.py — standalone evaluation script for a hybrid fine-tuned AlphaGenome.

Run with:
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
        --run-dir checkpoints/fine_tuned_hybrid_freeze1_2026-02-28_14-00-57

All outputs (plots + metrics CSV) are written to <run-dir>/.
"""

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# ── Force single-GPU BEFORE any JAX/CUDA import ────────────────────────────
# Prevents NCCL dual-instance deadlock (no LD_PRELOAD in terminal inference).
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"[env] CUDA_VISIBLE_DEVICES not set — defaulting to '0'")
else:
    print(f"[env] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")

sys.path.insert(0, "/home/ray/alphagenome_research")
sys.path.insert(0, "/home/ray/AG_Finetune")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in terminal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import pyBigWig

sys.path.insert(0, "/home/ray/AG_Finetune")
from utils.borzoi_loss import (
    alphagenome_predictions_scaling,
    compute_track_mean_from_bigwig,
)

# ── JAX (imported after CUDA_VISIBLE_DEVICES) ──────────────────────────────
import jax
import jax.numpy as jnp
jax.config.update("jax_compilation_cache_dir",
                  "/home/ray/AG_Finetune/.jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)

from alphagenome.data import genome as genome_lib
from alphagenome.models import dna_output
from alphagenome_research.model import dna_model
from alphagenome_research.model.metadata import metadata as metadata_lib

# ── Defaults (override via CLI) ────────────────────────────────────────────
DEFAULT_RUN_DIR = (
    "/home/ray/AG_Finetune/checkpoints/"
    "fine_tuned_hybrid_freeze1_2026-02-28_14-00-57"
)
DEFAULT_SIGNAL = (
    "/home/ray/AG_Finetune/data/CUTTag/ZHY-250227-1/"
    "ZHY-250227-1_ACTGCATATAAGGCGA/alignment/bigwig/"
    "ZHY-250227-1_ACTGCATATAAGGCGA_bowtie2"
    ".chrombpnet_reads_to_bw_unstranded.bw"
)
DEFAULT_FASTA       = "/home/ray/AG_Finetune/data/resources/mm10.fa"
DEFAULT_CHROM_SIZES = "/home/ray/AG_Finetune/data/resources/mm10.chrom.sizes"
DEFAULT_TEST_CHROMS = ["chr17", "chr18", "chr19"]
DEFAULT_WINDOWS_PER_CHROM = 5
DEFAULT_WINDOW_SIZE = 131_072   # 2^17 bp


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _banner(msg: str) -> None:
    print(f"\n{'='*60}\n{msg}\n{'='*60}", flush=True)


def _load_chrom_sizes(path: str) -> dict[str, int]:
    sizes = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                sizes[parts[0]] = int(parts[1])
    return sizes


def _build_eval_windows(
    chrom_sizes: dict[str, int],
    test_chroms: list[str],
    window_size: int,
    max_per_chrom: int,
) -> list[tuple[str, int, int]]:
    windows = []
    for chrom in test_chroms:
        size = chrom_sizes.get(chrom, 0)
        if size == 0:
            print(f"  WARNING: {chrom} not in chrom.sizes, skipping")
            continue
        starts = np.linspace(
            window_size, size - 2 * window_size, max_per_chrom, dtype=int
        )
        for s in starts:
            e = int(s) + window_size
            if e <= size:
                windows.append((chrom, int(s), e))
    return windows


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_dir: str, signal_path: str, fasta_path: str):
    """Load fine-tuned AlphaGenome with 1-track ATAC metadata."""
    _banner("Loading fine-tuned model")
    print(f"  Checkpoint : {checkpoint_dir}")
    print(f"  FASTA      : {fasta_path}")
    print(f"  JAX devices: {jax.devices()}", flush=True)

    # Reconstruct the exact 1-track metadata used during training
    # (matches tf_ft_hybrid.py lines 869–875)
    track_df = pd.DataFrame([{
        "output_type": "ATAC",
        "name":        "CUT&Tag",
        "strand":      ".",
        "file_path":   signal_path,
    }])
    fine_tune_metadata = metadata_lib.AlphaGenomeOutputMetadata(atac=track_df)

    organism = dna_model.Organism.MUS_MUSCULUS
    device   = jax.devices()[0]

    model = dna_model.create(
        checkpoint_path=checkpoint_dir,
        organism_settings={
            organism: dna_model.OrganismSettings(
                metadata=fine_tune_metadata,
                fasta_path=fasta_path,
            )
        },
        device=device,
    )
    print(f"  ✓ Model loaded on {device}", flush=True)
    return model, organism


# ══════════════════════════════════════════════════════════════════════════════
# Sanity-check inference
# ══════════════════════════════════════════════════════════════════════════════

def sanity_check(model, organism, window_size: int) -> None:
    """Run one prediction on chr1 to verify model + trigger XLA compilation."""
    _banner("Sanity-check inference (chr1:3,000,000)")
    chrom, start = "chr1", 3_000_000
    end = start + window_size
    print(f"  Interval   : {chrom}:{start:,}–{end:,}")
    print("  Running predict_interval …", flush=True)

    t0       = time.time()
    interval = genome_lib.Interval(chrom, start, end)
    preds    = model.predict_interval(
        interval,
        organism=organism,
        requested_outputs=[dna_output.OutputType.ATAC],
        ontology_terms=None,
    )
    elapsed = time.time() - t0

    # preds.atac is TrackData; .values shape = [window_size, num_tracks]
    raw = preds.atac.values
    print(f"  Output shape  : {raw.shape}")
    print(f"  Output dtype  : {raw.dtype}")
    print(f"  Value range   : [{float(raw.min()):.4f}, {float(raw.max()):.4f}]")
    print(f"  Elapsed       : {elapsed:.1f} s")
    print("  ✓ Sanity check PASSED", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def run_predictions(
    model,
    organism,
    eval_windows: list[tuple[str, int, int]],
    signal_path: str,
    window_size: int,
    track_mean: float = 1.0,
) -> list[dict]:
    """Predict + fetch ground-truth for every eval window.

    ``track_mean`` (from training_config.json) is used to apply the inverse
    AlphaGenome targets_scaling transform so predictions are in the same
    count space as the BigWig signal.
    """
    _banner(f"Running predictions on {len(eval_windows)} windows")
    print(f"  track_mean = {track_mean:.6f}  (inverse-scaling to count space)",
          flush=True)
    results = []
    bw = pyBigWig.open(signal_path)

    for i, (chrom, start, end) in enumerate(eval_windows):
        print(f"  [{i+1}/{len(eval_windows)}] {chrom}:{start:,}–{end:,} …",
              end="  ", flush=True)
        t0       = time.time()
        interval = genome_lib.Interval(chrom, start, end)
        preds    = model.predict_interval(
            interval,
            organism=organism,
            requested_outputs=[dna_output.OutputType.ATAC],
            ontology_terms=None,
        )

        # TrackData.values shape: [window_size, num_tracks]
        raw      = preds.atac.values
        # Model outputs are in the transformed space (after targets_scaling);
        # apply the inverse to get back to original count units.
        pred_scaled = np.array(raw[:, 0], dtype=np.float32)
        pred_arr = np.array(
            alphagenome_predictions_scaling(
                jnp.array(pred_scaled), track_mean=track_mean
            ),
            dtype=np.float32,
        )

        obs_raw  = bw.values(chrom, start, end, numpy=True)
        obs_arr  = np.nan_to_num(obs_raw, nan=0.0).astype(np.float32)

        assert len(pred_arr) == window_size, f"pred len={len(pred_arr)}"
        assert len(obs_arr)  == window_size, f"obs  len={len(obs_arr)}"

        elapsed = time.time() - t0
        print(f"pred_max={pred_arr.max():.4f}  obs_max={obs_arr.max():.4f}"
              f"  ({elapsed:.1f}s)", flush=True)

        results.append({
            "chrom": chrom,
            "start": start,
            "end":   end,
            "pred":  pred_arr,
            "obs":   obs_arr,
        })

    bw.close()
    print(f"\n  ✓ Done: {len(results)} windows.", flush=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(results: list[dict], window_size: int, bin_size: int = 128):
    rows      = []
    all_pred  = []
    all_obs   = []

    for r in results:
        n_bins = window_size // bin_size
        pred_b = r["pred"].reshape(n_bins, bin_size).mean(axis=1)
        obs_b  = r["obs"].reshape(n_bins, bin_size).mean(axis=1)

        pearson_r,  _ = stats.pearsonr(pred_b, obs_b)
        spearman_r, _ = stats.spearmanr(pred_b, obs_b)
        rows.append({
            "chrom":      r["chrom"],
            "start":      r["start"],
            "end":        r["end"],
            "pearson_r":  pearson_r,
            "spearman_r": spearman_r,
            "mse":        float(np.mean((pred_b - obs_b) ** 2)),
            "mae":        float(np.mean(np.abs(pred_b - obs_b))),
        })
        all_pred.append(pred_b)
        all_obs.append(obs_b)

    all_pred_flat = np.concatenate(all_pred)
    all_obs_flat  = np.concatenate(all_obs)
    global_r,   _ = stats.pearsonr(all_pred_flat, all_obs_flat)
    global_rho, _ = stats.spearmanr(all_pred_flat, all_obs_flat)

    df = pd.DataFrame(rows)
    return df, global_r, global_rho, all_pred_flat, all_obs_flat


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

PHASE_COLORS  = {"frozen": "#4273c8", "lora": "#e07b39"}
CHROM_COLORS  = {"chr17": "#4273c8", "chr18": "#e07b39", "chr19": "#4caf50"}


def plot_training_curves(metrics_csv: str, run_dir: str) -> None:
    df = pd.read_csv(metrics_csv, parse_dates=["timestamp"])
    df = df[~df["skipped"]].reset_index(drop=True)
    df["global_step"] = df.index + 1
    ROLL = 50
    df["loss_rolling"] = df["loss"].rolling(ROLL, min_periods=1).mean()

    lora_mask       = df["phase"] == "lora"
    transition_step = (df.loc[lora_mask, "global_step"].iloc[0]
                       if lora_mask.any() else None)
    x_max = df["global_step"].max()

    epoch_stats = (df.groupby(["epoch", "phase"])["loss"]
                   .agg(["mean", "min", "std"])
                   .reset_index())

    fig, (ax_loss, ax_ep) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=False,
        gridspec_kw={"height_ratios": [3, 1]})

    for phase, grp in df.groupby("phase", sort=False):
        ax_loss.scatter(grp["global_step"], grp["loss"],
                        s=2, alpha=0.18, color=PHASE_COLORS[phase])
    ax_loss.plot(df["global_step"], df["loss_rolling"],
                 color="k", lw=1.4, label=f"Rolling mean (w={ROLL})")

    ax_loss.axvspan(1, transition_step or x_max,
                    alpha=0.06, color=PHASE_COLORS["frozen"])
    if transition_step:
        ax_loss.axvspan(transition_step, x_max,
                        alpha=0.06, color=PHASE_COLORS["lora"])
        ax_loss.axvline(transition_step, color="red", lw=1.5,
                        linestyle="--", label="Phase transition")
        ylo, yhi = ax_loss.get_ylim()
        ax_loss.text(transition_step + x_max * 0.005, yhi * 0.88,
                     "Phase 2\n(Head + LoRA)", color="red",
                     fontsize=9, va="top")
        ax_loss.text(transition_step - x_max * 0.005, yhi * 0.88,
                     "Phase 1\n(Head only)", color=PHASE_COLORS["frozen"],
                     fontsize=9, va="top", ha="right")

    legend_handles = [
        mpatches.Patch(color=PHASE_COLORS["frozen"], alpha=0.6,
                       label="Phase 1 — Head only (frozen trunk, 1 epoch)"),
        mpatches.Patch(color=PHASE_COLORS["lora"],   alpha=0.6,
                       label="Phase 2 — Head + LoRA (rank=8, α=16, 2 epochs)"),
    ]
    h2, l2 = ax_loss.get_legend_handles_labels()
    ax_loss.legend(handles=legend_handles + h2, fontsize=9, loc="upper right")
    ax_loss.set_ylabel("Loss (Borzoi Poisson+Multinomial NLL)", fontsize=11)
    ax_loss.set_title(
        "Hybrid Curriculum Training Loss\n"
        f"Run: fine_tuned_hybrid_freeze1_2026-02-28_14-00-57  |  "
        f"Best loss: {df['loss'].min():.4f}", fontsize=12)
    ax_loss.set_yscale("log")
    ax_loss.grid(True, which="both", alpha=0.3)

    for _, row in epoch_stats.iterrows():
        c = PHASE_COLORS[row["phase"]]
        ax_ep.bar(row["epoch"], row["mean"], color=c, alpha=0.7,
                  edgecolor="k", linewidth=0.5)
        ax_ep.errorbar(row["epoch"], row["mean"], yerr=row["std"],
                       color="k", capsize=3, linewidth=0.8)
        ax_ep.scatter(row["epoch"], row["min"], color="k", s=20, zorder=5)
    ax_ep.set_xlabel("Epoch", fontsize=10)
    ax_ep.set_ylabel("Avg Loss / epoch", fontsize=9)
    ax_ep.set_xticks(epoch_stats["epoch"])
    ax_ep.set_xticklabels(
        [f"E{int(e)} ({p})"
         for e, p in zip(epoch_stats["epoch"], epoch_stats["phase"])],
        fontsize=8)
    ax_ep.set_yscale("log")
    ax_ep.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(run_dir, "training_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}", flush=True)


def plot_tracks(results: list[dict], window_size: int,
                run_dir: str, n: int = 4, bin_size: int = 128) -> None:
    show = results[:n]
    fig, axes = plt.subplots(len(show), 1,
                             figsize=(16, 3.2 * len(show)), sharex=False)
    if len(show) == 1:
        axes = [axes]
    for ax, r in zip(axes, show):
        nb    = window_size // bin_size
        pred  = r["pred"].reshape(nb, bin_size).mean(axis=1)
        obs   = r["obs"].reshape(nb, bin_size).mean(axis=1)
        bpos  = r["start"] + np.arange(nb) * bin_size + bin_size // 2
        ax.fill_between(bpos, obs,  alpha=0.55, color="#e07b39", label="Observed")
        ax.fill_between(bpos, pred, alpha=0.55, color="#4273c8", label="Predicted")
        rv = stats.pearsonr(pred, obs)[0]
        rh = stats.spearmanr(pred, obs)[0]
        ax.set_title(f"{r['chrom']}:{r['start']:,}–{r['end']:,}  |  "
                     f"Pearson r={rv:.3f}  Spearman ρ={rh:.3f}", fontsize=10)
        ax.set_ylabel("Signal (128-bp avg)", fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel(f"Position on {r['chrom']} (bp)", fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Predicted vs Observed CUT&Tag Signal  [Hybrid AlphaGenome · mm10]",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = os.path.join(run_dir, "pred_vs_obs_tracks.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}", flush=True)


def plot_peak_tracks(results: list[dict], window_size: int,
                     run_dir: str, n: int = 4, bin_size: int = 128) -> None:
    top = sorted(results, key=lambda r: r["obs"].max(), reverse=True)[:n]
    fig, axes = plt.subplots(len(top), 1,
                             figsize=(16, 3.5 * len(top)), sharex=False)
    if len(top) == 1:
        axes = [axes]
    for ax, r in zip(axes, top):
        nb    = window_size // bin_size
        pred  = r["pred"].reshape(nb, bin_size).mean(axis=1)
        obs   = r["obs"].reshape(nb, bin_size).mean(axis=1)
        bpos  = r["start"] + np.arange(nb) * bin_size + bin_size // 2
        summit = bpos[int(np.argmax(obs))]
        ax.fill_between(bpos, obs,  color="#e07b39", alpha=0.6,
                        label="Observed", step="mid")
        ax.fill_between(bpos, pred, color="#4273c8", alpha=0.6,
                        label="Predicted", step="mid")
        ax.axvline(summit, color="darkred", lw=1.2, linestyle=":",
                   label=f"Peak summit @ {summit:,}")
        rv = stats.pearsonr(pred, obs)[0]
        rh = stats.spearmanr(pred, obs)[0]
        ax.set_title(f"{r['chrom']}:{r['start']:,}–{r['end']:,}  |  "
                     f"Pearson r={rv:.3f}  Spearman ρ={rh:.3f}  "
                     f"obs_max={obs.max():.2f}", fontsize=9)
        ax.set_ylabel("Signal (128-bp bins)", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(f"Position on {r['chrom']} (bp)", fontsize=9)
    plt.suptitle("Top-N Peak Signal Windows  [Hybrid AlphaGenome · mm10]",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = os.path.join(run_dir, "eval_peak_tracks.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}", flush=True)


def plot_correlation(metrics_df, global_r, global_rho,
                     all_pred_flat, all_obs_flat, run_dir: str) -> None:
    fig, (ax_sc, ax_bar) = plt.subplots(1, 2, figsize=(14, 5))

    rng = np.random.default_rng(42)
    idx = rng.choice(len(all_pred_flat),
                     size=min(40_000, len(all_pred_flat)), replace=False)
    h = ax_sc.hexbin(all_obs_flat[idx], all_pred_flat[idx],
                     gridsize=80, mincnt=1, cmap="YlOrRd", bins="log")
    plt.colorbar(h, ax=ax_sc, label="log₁₀(count)")
    vmax = max(all_obs_flat.max(), all_pred_flat.max()) * 1.05
    ax_sc.plot([0, vmax], [0, vmax], "k--", lw=1.2, label="y = x")
    ax_sc.set_xlabel("Observed signal (128-bp bin)", fontsize=10)
    ax_sc.set_ylabel("Predicted signal (128-bp bin)", fontsize=10)
    ax_sc.set_title(f"Predicted vs Observed\n"
                    f"Pearson r = {global_r:.3f}  |  Spearman ρ = {global_rho:.3f}",
                    fontsize=10)
    ax_sc.legend(fontsize=9)
    ax_sc.grid(True, alpha=0.3)

    bar_colors = [CHROM_COLORS.get(r, "#888") for r in metrics_df["chrom"]]
    ax_bar.bar(range(len(metrics_df)), metrics_df["pearson_r"],
               color=bar_colors, edgecolor="k", linewidth=0.4)
    ax_bar.axhline(global_r, color="red", lw=1.5, linestyle="--",
                   label=f"Global r = {global_r:.3f}")
    ax_bar.set_xticks(range(len(metrics_df)))
    ax_bar.set_xticklabels(
        [f"{row['chrom']}\n{row['start']//1000}k"
         for _, row in metrics_df.iterrows()],
        fontsize=7, rotation=45)
    ax_bar.set_ylabel("Pearson r", fontsize=10)
    ax_bar.set_title("Per-window Pearson r", fontsize=10)
    ax_bar.grid(True, axis="y", alpha=0.3)
    for chrom, c in CHROM_COLORS.items():
        ax_bar.bar([], [], color=c, label=chrom)
    ax_bar.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(run_dir, "eval_correlation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a hybrid fine-tuned AlphaGenome on held-out chromosomes.")
    parser.add_argument("--run-dir",    default=DEFAULT_RUN_DIR,
                        help="Training run directory (contains best_checkpoint/)")
    parser.add_argument("--signal",     default=DEFAULT_SIGNAL,
                        help="Path to CUT&Tag BigWig signal file")
    parser.add_argument("--fasta",      default=DEFAULT_FASTA)
    parser.add_argument("--chrom-sizes",default=DEFAULT_CHROM_SIZES)
    parser.add_argument("--test-chroms",default=",".join(DEFAULT_TEST_CHROMS),
                        help="Comma-separated held-out chromosomes")
    parser.add_argument("--windows-per-chrom", type=int,
                        default=DEFAULT_WINDOWS_PER_CHROM)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--bin-size",    type=int, default=128)
    parser.add_argument("--skip-sanity", action="store_true",
                        help="Skip the chr1 sanity-check inference")
    args = parser.parse_args()

    run_dir     = args.run_dir
    ckpt_dir    = os.path.join(run_dir, "best_checkpoint", "model")
    metrics_csv = os.path.join(run_dir, "metrics.csv")
    test_chroms = [c.strip() for c in args.test_chroms.split(",")]

    _banner("AlphaGenome Fine-tuned Evaluation")
    print(f"  Run dir    : {run_dir}")
    print(f"  Checkpoint : {ckpt_dir}")
    print(f"  Signal     : {args.signal}")
    print(f"  Test chroms: {test_chroms}")
    print(f"  Windows    : {args.windows_per_chrom} per chrom", flush=True)

    # ── Load training config ───────────────────────────────────────────────
    cfg_path = os.path.join(run_dir, "training_config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        print(f"\n  Training config:")
        for k, v in cfg.items():
            print(f"    {k:<22}: {v}")

    # Resolve track_mean: prefer training_config.json, fall back to computing
    # it directly from the BigWig (older runs pre-date this field).
    if "track_mean" in cfg:
        track_mean = float(cfg["track_mean"])
        print(f"\n  track_mean (from config): {track_mean:.6f}", flush=True)
    else:
        print("\n  track_mean not found in config — computing from BigWig...",
              flush=True)
        track_mean = compute_track_mean_from_bigwig(
            args.signal, chrom_sizes_path=args.chrom_sizes
        )
        print(f"  track_mean (computed)   : {track_mean:.6f}", flush=True)

    # ── Load model ─────────────────────────────────────────────────────────
    model, organism = load_model(ckpt_dir, args.signal, args.fasta)

    # ── Sanity check ───────────────────────────────────────────────────────
    if not args.skip_sanity:
        sanity_check(model, organism, args.window_size)

    # ── Training curves ────────────────────────────────────────────────────
    if os.path.exists(metrics_csv):
        _banner("Plotting training curves")
        plot_training_curves(metrics_csv, run_dir)
    else:
        print(f"\n  [skip] metrics.csv not found: {metrics_csv}")

    # ── Build eval windows ─────────────────────────────────────────────────
    chrom_sizes  = _load_chrom_sizes(args.chrom_sizes)
    eval_windows = _build_eval_windows(
        chrom_sizes, test_chroms, args.window_size, args.windows_per_chrom)
    print(f"\n  Evaluation windows ({len(eval_windows)}):")
    for w in eval_windows:
        print(f"    {w[0]}:{w[1]:,}–{w[2]:,}")

    # ── Predictions ────────────────────────────────────────────────────────
    results = run_predictions(model, organism, eval_windows,
                              args.signal, args.window_size,
                              track_mean=track_mean)

    # ── Metrics ────────────────────────────────────────────────────────────
    _banner("Computing metrics")
    metrics_df, global_r, global_rho, all_pred, all_obs = compute_metrics(
        results, args.window_size, args.bin_size)

    print(f"\n  Per-window metrics ({args.bin_size}-bp bins):")
    print(metrics_df.to_string(index=False, float_format="%.4f"))
    print(f"\n  Global Pearson r   : {global_r:.4f}")
    print(f"  Global Spearman ρ  : {global_rho:.4f}")

    out_csv = os.path.join(run_dir, "eval_metrics.csv")
    metrics_df.to_csv(out_csv, index=False)
    print(f"\n  Saved → {out_csv}", flush=True)

    # ── Plots ──────────────────────────────────────────────────────────────
    _banner("Generating plots")
    plot_tracks(results, args.window_size, run_dir)
    plot_peak_tracks(results, args.window_size, run_dir)
    plot_correlation(metrics_df, global_r, global_rho, all_pred, all_obs, run_dir)

    _banner("DONE")
    print(f"  Outputs written to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
