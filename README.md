# AG_Finetune

Fine-tuning [AlphaGenome](https://github.com/google-deepmind/alphagenome_research) (450 M parameter JAX/Haiku model) on custom CUT&Tag data.

The recommended strategy is a **two-phase hybrid Head+LoRA curriculum** implemented in `scripts/tf_ft_hybrid.py`.

---

## Strategy: Hybrid Head+LoRA Curriculum

Training proceeds in two phases controlled by the `freeze_epochs` config key:

| Phase | What trains | Params | Per-step time |
|---|---|---|---|
| Phase 1 — frozen trunk | New ATAC head only | ~9 K | ~0.15 s |
| Phase 2 — head + LoRA | Head + LoRA A/B on all Q/K/V projections | ~437 K (0.10 %) | intermediate |

**Why this beats full fine-tuning for histone marks:**
CUT&Tag H3K27ac/H3K27me3 signals differ substantially from the ATAC-seq signals AlphaGenome was pretrained on. LoRA adapts the attention layers for the new biochemistry while preserving the general genomic representations, reducing catastrophic forgetting.

---

## Hardware

8 × NVIDIA RTX 6000 Ada (49 GB each). Parameters are **replicated** across all devices; only the batch dimension is sharded. JAX auto-averages gradients — no `pmap` or manual `pmean` required.

---

## Data Preparation

CUT&Tag uses Tn5 tagmentation, making it biochemically equivalent to ATAC-seq. The ATAC-seq head at 1 bp resolution is used.

### 1 — Alignment and base-resolution signal

```bash
# Align with bowtie2
bowtie2 --end-to-end --very-sensitive --no-mixed --no-discordant \
    -x /path/to/mm10_index -1 R1.fastq.gz -2 R2.fastq.gz \
    | samtools sort -o aligned.bam && samtools index aligned.bam

# Base-resolution insertion counts (ChromBPNet helper)
# +4/-4 Tn5 shift, matching AlphaGenome pretraining
python chrombpnet/helpers/preprocessing/reads_to_bigwig.py \
    --bam aligned.bam --output signal.bw \
    --chrom-sizes mm10.chrom.sizes \
    --plus-shift 4 --minus-shift -4
```

(Preprocessing ran on luolab@luolab-X11DAi-N. Copy bigwig's with `chrombpnet_reads_to_bw_unstranded` tag to server.)

### 2 — Average replicates and normalise to 100 M insertions

```bash
python scripts/average_normalize_bigwigs.py \
    --config samples.tsv \
    --output-dir data/CUTTag/processed \
    --chrom-sizes data/resources/mm10.chrom.sizes \
    --normalize-to 100000000
```

`samples.tsv` columns: `sample_id`, `cell_type`, `epitope`, `bigwig_path`.

**Signal requirements:**
- Base-pair resolution, non-negative values
- Normalised to 100 M total insertions per track
- **Do not** apply log, z-score, or any further scaling — the loss function handles all transforms internally via `alphagenome_targets_scaling`

---

## Training

### Quick start

```bash
LD_PRELOAD=/home/ray/miniforge3/envs/alphagenome/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2 \
    python scripts/tf_ft_hybrid.py --config configs/config_bigwig_hybrid.yaml
```

> **`LD_PRELOAD` is required** for multi-GPU runs. It forces a single NCCL instance, resolving the jaxlib 0.9.0 dual-NCCL conflict (see `docs/NCCL_RESOLUTION.md`).

```bash
LD_PRELOAD=/home/ray/miniforge3/envs/alphagenome/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2 \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python scripts/tf_ft_hybrid.py --config configs/config_bigwig_hybrid_quick.yaml
```

### Configuration

Edit `configs/config_bigwig_hybrid.yaml`. Key parameters:

```yaml
# Paths
checkpoint: /path/to/alphagenome-all-folds
fasta:      /path/to/mm10.fa
signal:     /path/to/signal.bw        # BigWig (recommended) or BedGraph
output:     /path/to/checkpoints/fine_tuned

# Optional: BED file of specific windows. If omitted, whole-genome
# non-overlapping windows are generated automatically from chrom_sizes.
windows_bed:
chrom_sizes: /path/to/mm10.chrom.sizes

# Precision
precision_policy: params=float32,compute=float32,output=float32

# Training
epochs: 10
freeze_epochs: 2        # Phase 1 length (head-only, trunk frozen)
batch_size: 4
learning_rate: 5e-4     # Peak LR (AdamW, warmup + cosine decay to 0)
weight_decay: 0.04      # AdamW weight decay

# LoRA (Phase 2)
lora_rank: 8            # Increase to 32/64 if Phase 2 plateaus
lora_alpha: 32.0        # Effective scaling = alpha/rank

# Data
window_size: 8192       # 2^13 bp — fits AAV genome (~4.7 kb) with ~1.75 kb flanking
organism_id: 1          # 0=human, 1=mouse
augment: true
max_shift: 1024

# Chromosome splits (train excludes val + test chroms)
val_chroms:  [chr7]             # Used for best checkpoint selection
test_chroms: [chr8, chr9]       # Never seen during training
```

### Optimizer

**AdamW** with default β parameters (β₁=0.9, β₂=0.999, ε=1e-8), matching the AlphaGenome paper.
- Weight decay: 0.04 (paper distillation value; pre-training used 0.4)
- LR schedule: linear warmup (5 % of phase steps) → **cosine decay to 0**
- Gradient clipping: global norm 0.5
- Fresh optimizer at phase transition — no momentum bleed from Phase 1 to Phase 2

### Train/val/test splits

Splits are chromosome-based (set `val_chroms` / `test_chroms` in the YAML):
- **Training:** all canonical chromosomes except val + test chroms, shuffled continuously across epochs
- **Validation:** evaluated after every epoch; used to select the best checkpoint
- **Test:** held out entirely — use `scripts/evaluate.py` after training is complete

### Training logs

Each run creates `<output>_hybrid_freeze{N}_{timestamp}/`:

| File | Contents |
|---|---|
| `train.log` | Full stdout (line-buffered, survives crashes) |
| `stderr.log` | JAX/NCCL warnings (redirected to keep terminal clean) |
| `metrics.csv` | Per-batch: `timestamp, epoch, batch, loss, phase, nan_event, skipped` |
| `training_config.json` | Full config snapshot including `track_mean`, `val_chroms`, `test_chroms` |
| `epoch_NNN/` | Per-epoch orbax checkpoint |
| `best_checkpoint/` | Epoch with lowest **validation** loss (or train loss if no val chroms set) |

---

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n alphagenome \
    python scripts/evaluate.py \
    --run-dir checkpoints/fine_tuned_hybrid_freeze2_<timestamp>
```

Outputs written to the run directory:

| File | Contents |
|---|---|
| `training_curve.png` | Phase 1 + 2 step-level loss scatter + epoch bars |
| `eval_metrics.csv` | Pearson r / Spearman ρ / MSE / MAE per eval window |
| `pred_vs_obs_tracks.png` | Predicted vs observed signal overlay |
| `eval_correlation.png` | Hex scatter + per-window correlation bar chart |

---

## Checkpoints

Format: [orbax](https://github.com/google/orbax) `StandardCheckpointer`.

Each checkpoint directory contains:
- `model/` — trunk params with LoRA deltas merged in; compatible with `dna_model.create()` API
- `lora_raw/` — raw A/B matrices for training resumption
- `lora_metadata.json` — rank, alpha, scaling, target layers, current phase

**Loading a fine-tuned checkpoint:**
```python
from alphagenome_research.model import dna_model
model = dna_model.create(
    checkpoint_path='checkpoints/fine_tuned.../best_checkpoint',
    organism_settings=None,
    device=None,
)
```

> Optimizer state is **not** saved — it is reinitialized if training is resumed.

---

## Project structure

```
AG_Finetune/
├── scripts/
│   ├── tf_ft_hybrid.py              # ★ Recommended: hybrid Head+LoRA curriculum
│   ├── tf_ft_freeze.py              # Trunk-freeze → full-unfreeze curriculum
│   ├── tf_ft.py                     # Full-parameter training from epoch 1
│   ├── evaluate.py                  # Post-training evaluation (single GPU)
│   └── average_normalize_bigwigs.py    # Replicate averaging + normalisation to 100 M
├── configs/
│   ├── config_bigwig_hybrid.yaml    # ★ Config for tf_ft_hybrid.py
│   ├── config_bigwig_freeze.yaml    # Config for tf_ft_freeze.py
│   └── config_bigwig.yaml           # Config for tf_ft.py
├── models/
│   └── model.py                     # HeadReplaceAlphaGenomeModel, LoRAAlphaGenomeModel
├── utils/
│   ├── borzoi_loss.py               # AlphaGenome Poisson + Multinomial NLL loss
│   └── lora_utils.py                # LoRA param init and layer targeting
├── dataset/
│   └── tf_dataset.py                # TensorFlow / BigWig data pipeline
└── docs/
    └── NCCL_RESOLUTION.md           # NCCL dual-instance fix documentation
```

---

## Troubleshooting

### NCCL errors on multi-GPU startup
Prepend `LD_PRELOAD=.../libnccl.so.2` to all training commands. See `docs/NCCL_RESOLUTION.md`.

### ZFS pool `/celestial` not mounted
```bash
sudo zpool import celestial
ls /celestial/ray/model_checkpoints/AlphaGenome/
```

### Rendezvous warnings during training
`"This thread has been waiting for acquire clique..."` messages are false positives — each is followed by `"Thread is unstuck!"` within seconds. These are redirected to `stderr.log`.
