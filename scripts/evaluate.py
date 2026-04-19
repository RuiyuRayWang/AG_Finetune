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
from io import BytesIO
import matplotlib.image as mpimg
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
from alphagenome.data import track_data as track_data_lib
from alphagenome.models import dna_output
from alphagenome.visualization import plot_components
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

def load_model(checkpoint_dir: str, signal_paths: list[str], fasta_path: str):
    """Load fine-tuned AlphaGenome with N-track ATAC metadata."""
    _banner("Loading fine-tuned model")
    print(f"  Checkpoint  : {checkpoint_dir}")
    print(f"  FASTA       : {fasta_path}")
    print(f"  Tracks      : {len(signal_paths)}")
    print(f"  JAX devices : {jax.devices()}", flush=True)

    # Reconstruct the exact N-track metadata used during training
    track_df = pd.DataFrame([
        {
            "output_type": "ATAC",
            "name":        os.path.splitext(os.path.basename(p))[0],
            "strand":      ".",
            "file_path":   p,
        }
        for p in signal_paths
    ])
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
    signal_paths: list[str],
    window_size: int,
    track_means,          # float | list[float] | np.ndarray — one per track
) -> list[dict]:
    """Predict + fetch ground-truth for every eval window.

    Supports N signal tracks; ``results`` contain pred/obs as
    ``[window_size, N]`` arrays. ``track_means`` (from training_config.json)
    is used to apply the inverse AlphaGenome targets_scaling per track.
    """
    track_means = np.asarray(track_means, dtype=np.float32).reshape(-1)
    num_tracks  = len(signal_paths)
    _banner(f"Running predictions on {len(eval_windows)} windows  ({num_tracks} track(s))")
    for i, (sp, tm) in enumerate(zip(signal_paths, track_means)):
        print(f"  track [{i}] track_mean={tm:.6f}  {os.path.basename(sp)}",
              flush=True)

    results = []
    bws = [pyBigWig.open(sp) for sp in signal_paths]

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
        raw = np.array(preds.atac.values, dtype=np.float32)   # [W, T]

        # Apply inverse scaling per track — back to count space
        pred_arr = np.stack([
            np.array(
                alphagenome_predictions_scaling(
                    jnp.array(raw[:, t]), track_mean=float(track_means[t])
                ),
                dtype=np.float32,
            )
            for t in range(num_tracks)
        ], axis=1)  # [W, T]

        obs_arr = np.stack([
            np.nan_to_num(
                bw.values(chrom, start, end, numpy=True), nan=0.0
            ).astype(np.float32)
            for bw in bws
        ], axis=1)  # [W, T]

        assert pred_arr.shape[0] == window_size, f"pred shape={pred_arr.shape}"
        assert obs_arr.shape[0]  == window_size, f"obs  shape={obs_arr.shape}"

        elapsed = time.time() - t0
        print(f"pred_max={pred_arr.max():.4f}  obs_max={obs_arr.max():.4f}"
              f"  ({elapsed:.1f}s)", flush=True)

        results.append({
            "chrom": chrom,
            "start": start,
            "end":   end,
            "pred":  pred_arr,   # [window_size, num_tracks]
            "obs":   obs_arr,    # [window_size, num_tracks]
        })

    for bw in bws:
        bw.close()
    print(f"\n  ✓ Done: {len(results)} windows.", flush=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    results: list[dict],
    window_size: int,
    track_names: list[str],
    bin_size: int = 128,
):
    """Compute per-track, per-window Pearson r / Spearman ρ / MSE / MAE.

    Returns
    -------
    df          : DataFrame with columns [track, chrom, start, end, pearson_r, ...]
    global_r    : dict {track_name -> global Pearson r}
    global_rho  : dict {track_name -> global Spearman rho}
    all_pred    : dict {track_name -> concatenated pred bins (float32)}
    all_obs     : dict {track_name -> concatenated obs bins (float32)}
    """
    num_tracks   = results[0]["pred"].shape[1]
    rows         = []
    track_preds  = [[] for _ in range(num_tracks)]
    track_obs    = [[] for _ in range(num_tracks)]

    for r in results:
        n_bins     = window_size // bin_size
        pred_bins  = r["pred"].reshape(n_bins, bin_size, num_tracks).mean(axis=1)  # [B, T]
        obs_bins   = r["obs"].reshape(n_bins,  bin_size, num_tracks).mean(axis=1)

        for t, name in enumerate(track_names):
            pb = pred_bins[:, t]
            ob = obs_bins[:, t]
            pearson_r,  _ = stats.pearsonr(pb, ob)
            spearman_r, _ = stats.spearmanr(pb, ob)
            rows.append({
                "track":      name,
                "chrom":      r["chrom"],
                "start":      r["start"],
                "end":        r["end"],
                "pearson_r":  pearson_r,
                "spearman_r": spearman_r,
                "mse":        float(np.mean((pb - ob) ** 2)),
                "mae":        float(np.mean(np.abs(pb - ob))),
            })
            track_preds[t].append(pb)
            track_obs[t].append(ob)

    df         = pd.DataFrame(rows)
    global_r   = {}
    global_rho = {}
    all_pred   = {}
    all_obs    = {}
    for t, name in enumerate(track_names):
        pf = np.concatenate(track_preds[t])
        of = np.concatenate(track_obs[t])
        global_r[name],   _ = stats.pearsonr(pf, of)
        global_rho[name], _ = stats.spearmanr(pf, of)
        all_pred[name]       = pf
        all_obs[name]        = of

    return df, global_r, global_rho, all_pred, all_obs


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

PHASE_COLORS  = {"frozen": "#4273c8", "lora": "#e07b39"}
CHROM_COLORS  = {"chr17": "#4273c8", "chr18": "#e07b39", "chr19": "#4caf50"}

# Colors for the paper-style separate-track visualisation
PRED_COLOR = "black"    # predicted signal — matches AlphaGenome colab convention
OBS_COLOR  = "#d63031"  # observed signal  — matches AlphaGenome colab convention


def _build_track_fig(
    r: dict,
    window_size: int,
    bin_size: int,
    track_idx: int = 0,
    track_name: str = "CUT&Tag",
    fig_width: int = 14,
) -> matplotlib.figure.Figure:
    """Return a plot_components figure with pred (top) and obs (bottom) as
    separate filled tracks for one genomic window and one signal track.

    Uses ``alphagenome.visualization.plot_components`` — the same API used in
    the AlphaGenome paper figures — so the layout and styling match the paper.
    """
    nb         = window_size // bin_size
    num_tracks = r["pred"].shape[1]
    pred_b = (r["pred"].reshape(nb, bin_size, num_tracks).mean(axis=1)
              [:, track_idx].astype(np.float32))
    obs_b  = (r["obs"].reshape(nb,  bin_size, num_tracks).mean(axis=1)
              [:, track_idx].astype(np.float32))

    chrom, start, end = r["chrom"], r["start"], r["end"]
    interval = genome_lib.Interval(chrom, start, end)

    # Build TrackData objects; metadata needs at least 'name' and 'strand'.
    pred_tdata = track_data_lib.TrackData(
        values=pred_b.reshape(-1, 1),
        metadata=pd.DataFrame([{"name": f"Predicted [{track_name}]", "strand": "."}]),
        resolution=bin_size,
        interval=interval,
    )
    obs_tdata = track_data_lib.TrackData(
        values=obs_b.reshape(-1, 1),
        metadata=pd.DataFrame([{"name": f"Observed [{track_name}]", "strand": "."}]),
        resolution=bin_size,
        interval=interval,
    )

    rv = stats.pearsonr(pred_b, obs_b)[0]
    rh = stats.spearmanr(pred_b, obs_b)[0]
    title = (
        f"{chrom}:{start:,}–{end:,}  |  "
        f"Pearson r = {rv:.3f}   Spearman ρ = {rh:.3f}"
    )

    fig = plot_components.plot(
        [
            plot_components.Tracks(
                tdata=pred_tdata,
                filled=True,
                track_colors=PRED_COLOR,
                ylabel_template="{name}",
            ),
            plot_components.Tracks(
                tdata=obs_tdata,
                filled=True,
                track_colors=OBS_COLOR,
                ylabel_template="{name}",
            ),
        ],
        interval=interval,
        title=title,
        fig_width=fig_width,
        fig_height_scale=1.8,
    )
    return fig


def _stitch_figs_vertically(figs: list, out_path: str, dpi: int = 120) -> None:
    """Render each matplotlib Figure to a buffer, stitch vertically, save PNG."""
    imgs = []
    for fig in figs:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        imgs.append(mpimg.imread(buf))
        plt.close(fig)

    # Pad canvases to the same width (fill with white)
    max_w = max(im.shape[1] for im in imgs)
    nch   = imgs[0].shape[2]
    padded = []
    for im in imgs:
        if im.shape[1] < max_w:
            pad = np.ones((im.shape[0], max_w - im.shape[1], nch), dtype=np.float32)
            im = np.concatenate([im, pad], axis=1)
        padded.append(im)

    canvas = np.concatenate(padded, axis=0)
    plt.imsave(out_path, canvas)
    print(f"  Saved → {out_path}", flush=True)


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


def plot_tracks(results: list[dict], window_size: int, track_names: list[str],
                run_dir: str, n: int = 4, bin_size: int = 128) -> None:
    """Paper-style track plot: one PNG per signal track with predicted and
    observed as separate stacked panels.
    """
    for t, name in enumerate(track_names):
        figs = [
            _build_track_fig(r, window_size, bin_size, track_idx=t, track_name=name)
            for r in results[:n]
        ]
        tag = name.replace("/", "_").replace(" ", "_")
        _stitch_figs_vertically(
            figs, os.path.join(run_dir, f"pred_vs_obs_tracks_{tag}.png")
        )


def plot_peak_tracks(results: list[dict], window_size: int, track_names: list[str],
                     run_dir: str, n: int = 4, bin_size: int = 128) -> None:
    """Paper-style peak-track plot: top-N windows by per-track observed signal,
    one PNG per signal track.
    """
    for t, name in enumerate(track_names):
        top = sorted(
            results, key=lambda r, _t=t: r["obs"][:, _t].max(), reverse=True
        )[:n]
        figs = [
            _build_track_fig(r, window_size, bin_size, track_idx=t, track_name=name)
            for r in top
        ]
        tag = name.replace("/", "_").replace(" ", "_")
        _stitch_figs_vertically(
            figs, os.path.join(run_dir, f"eval_peak_tracks_{tag}.png")
        )


def plot_correlation(metrics_df, global_r, global_rho,
                     all_pred, all_obs, run_dir: str) -> None:
    """One column of (scatter + bar) subplots per signal track."""
    track_names = list(global_r.keys())
    num_tracks  = len(track_names)

    fig, axes = plt.subplots(2, num_tracks,
                             figsize=(10 * num_tracks, 10),
                             squeeze=False)
    rng = np.random.default_rng(42)

    for col, name in enumerate(track_names):
        ax_sc  = axes[0, col]
        ax_bar = axes[1, col]
        pf   = all_pred[name]
        of   = all_obs[name]
        gr   = global_r[name]
        grho = global_rho[name]

        idx = rng.choice(len(pf), size=min(40_000, len(pf)), replace=False)
        h = ax_sc.hexbin(of[idx], pf[idx],
                     gridsize=80, mincnt=1, cmap="YlOrRd", bins="log")
        plt.colorbar(h, ax=ax_sc, label="log₁₀(count)")
        vmax = max(float(of.max()), float(pf.max())) * 1.05
        ax_sc.plot([0, vmax], [0, vmax], "k--", lw=1.2, label="y = x")
        ax_sc.set_xlabel("Observed signal (128-bp bin)", fontsize=10)
        ax_sc.set_ylabel("Predicted signal (128-bp bin)", fontsize=10)
        ax_sc.set_title(
            f"{name}\nPearson r = {gr:.3f}  |  Spearman ρ = {grho:.3f}",
            fontsize=10)
        ax_sc.legend(fontsize=9)
        ax_sc.grid(True, alpha=0.3)

        sub_df     = metrics_df[metrics_df["track"] == name]
        bar_colors = [CHROM_COLORS.get(r, "#888") for r in sub_df["chrom"]]
        ax_bar.bar(range(len(sub_df)), sub_df["pearson_r"],
                   color=bar_colors, edgecolor="k", linewidth=0.4)
        ax_bar.axhline(gr, color="red", lw=1.5, linestyle="--",
                       label=f"Global r = {gr:.3f}")
        ax_bar.set_xticks(range(len(sub_df)))
        ax_bar.set_xticklabels(
            [f"{row['chrom']}\n{row['start']//1000}k"
             for _, row in sub_df.iterrows()],
            fontsize=7, rotation=45)
        ax_bar.set_ylabel("Pearson r", fontsize=10)
        ax_bar.set_title(f"Per-window Pearson r\n{name}", fontsize=10)
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
                        help="Fallback BigWig path if not in training_config.json")
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

    run_dir     = os.path.abspath(args.run_dir)
    ckpt_dir    = os.path.join(run_dir, "best_checkpoint", "model")
    metrics_csv = os.path.join(run_dir, "metrics.csv")

    _banner("AlphaGenome Fine-tuned Evaluation")
    print(f"  Run dir    : {run_dir}")
    print(f"  Checkpoint : {ckpt_dir}")
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

    # ── Resolve signal paths ───────────────────────────────────────────────
    # Priority: training_config.json `signal_paths` > `signal_path` > CLI --signal
    if "signal_paths" in cfg:
        signal_paths = cfg["signal_paths"]
        print(f"\n  signal_paths (from config, {len(signal_paths)} tracks):", flush=True)
        for p in signal_paths:
            print(f"    {p}")
    elif "signal_path" in cfg:
        signal_paths = [cfg["signal_path"]]
        print(f"\n  signal_path (from config): {signal_paths[0]}", flush=True)
    else:
        signal_paths = [args.signal]
        print(f"\n  signal (from CLI): {signal_paths[0]}", flush=True)

    # ── Resolve window_size from config if not overridden on CLI ──────────
    if args.window_size == DEFAULT_WINDOW_SIZE and "window_size" in cfg:
        window_size = int(cfg["window_size"])
        print(f"\n  window_size (from config): {window_size}", flush=True)
    else:
        window_size = args.window_size

    # ── Resolve test_chroms from config if not overridden on CLI ──────────
    cli_test_chroms = [c.strip() for c in args.test_chroms.split(",")]
    if cli_test_chroms == DEFAULT_TEST_CHROMS and "test_chroms" in cfg:
        test_chroms = list(cfg["test_chroms"])
        print(f"  test_chroms (from config): {test_chroms}", flush=True)
    else:
        test_chroms = cli_test_chroms

    # ── Resolve track_means ────────────────────────────────────────────────
    # Priority: config `track_means` > config `track_mean` > compute from BigWig
    if "track_means" in cfg:
        track_means = list(cfg["track_means"])
        print(f"\n  track_means (from config): {track_means}", flush=True)
    elif "track_mean" in cfg:
        track_means = [float(cfg["track_mean"])]
        print(f"\n  track_mean (from config): {track_means[0]:.6f}", flush=True)
    else:
        print("\n  track_means not in config — computing from BigWig(s)...", flush=True)
        track_means = [
            compute_track_mean_from_bigwig(p, chrom_sizes_path=args.chrom_sizes)
            for p in signal_paths
        ]
        for p, tm in zip(signal_paths, track_means):
            print(f"    {os.path.basename(p)}: {tm:.6f}")

    # Short names for plots (basename without extension)
    track_names = [os.path.splitext(os.path.basename(p))[0] for p in signal_paths]

    # ── Load model ─────────────────────────────────────────────────────────
    model, organism = load_model(ckpt_dir, signal_paths, args.fasta)

    # ── Sanity check ───────────────────────────────────────────────────────
    if not args.skip_sanity:
        sanity_check(model, organism, window_size)

    # ── Training curves ────────────────────────────────────────────────────
    if os.path.exists(metrics_csv):
        _banner("Plotting training curves")
        plot_training_curves(metrics_csv, run_dir)
    else:
        print(f"\n  [skip] metrics.csv not found: {metrics_csv}")

    # ── Build eval windows ─────────────────────────────────────────────────
    chrom_sizes  = _load_chrom_sizes(args.chrom_sizes)
    eval_windows = _build_eval_windows(
        chrom_sizes, test_chroms, window_size, args.windows_per_chrom)
    print(f"\n  Evaluation windows ({len(eval_windows)}):")
    for w in eval_windows:
        print(f"    {w[0]}:{w[1]:,}–{w[2]:,}")

    # ── Predictions ────────────────────────────────────────────────────────
    results = run_predictions(model, organism, eval_windows,
                              signal_paths, window_size,
                              track_means=track_means)

    # ── Metrics ────────────────────────────────────────────────────────────
    _banner("Computing metrics")
    metrics_df, global_r, global_rho, all_pred, all_obs = compute_metrics(
        results, window_size, track_names, args.bin_size)

    print(f"\n  Per-window metrics ({args.bin_size}-bp bins):")
    print(metrics_df.to_string(index=False, float_format="%.4f"))
    for name in track_names:
        print(f"\n  [{name}]  Global Pearson r = {global_r[name]:.4f}"
              f"   Global Spearman rho = {global_rho[name]:.4f}")

    out_csv = os.path.join(run_dir, "eval_metrics.csv")
    metrics_df.to_csv(out_csv, index=False)
    print(f"\n  Saved → {out_csv}", flush=True)

    # ── Plots ──────────────────────────────────────────────────────────────
    _banner("Generating plots")
    plot_tracks(results, window_size, track_names, run_dir)
    plot_peak_tracks(results, window_size, track_names, run_dir)
    plot_correlation(metrics_df, global_r, global_rho, all_pred, all_obs, run_dir)

    _banner("DONE")
    print(f"  Outputs written to: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
