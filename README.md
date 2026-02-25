# AG_Finetune

Fine-tuning [AlphaGenome](https://github.com/google-deepmind/alphagenome_research) (450 M parameter JAX/Haiku model) on custom CUT&Tag data, using two strategies selectable at runtime.

## Strategies

| Strategy | Description | Trainable params |
|---|---|---|
| `head_replace` (default) | Replaces AlphaGenome's ATAC output heads with a new single-track head and fine-tunes the full model. Mirrors the official DeepMind fine-tuning colab. | ~404 M |
| `lora` | Injects LoRA adapters (rank 8) into Q/K/V projections in all 9 transformer blocks; base weights are frozen. | ~428 K (0.10 %) |

Both strategies share the same training loop, loss function, checkpoint format, and multi-GPU data-parallel setup.

---

## Data Preparation

CUT&Tag uses Tn5 tagmentation, making it biochemically equivalent to ATAC-seq. Use the ATAC-seq head at base-pair (1 bp) resolution.

### 1 — Alignment and signal generation

```bash
# Align
bowtie2 --end-to-end --very-sensitive --no-mixed --no-discordant \
    -x /path/to/mm10_index -1 R1.fastq.gz -2 R2.fastq.gz \
    | samtools sort -o aligned.bam
samtools index aligned.bam

# Base-resolution insertion counts (ChromBPNet helper)
# Apply +4/-4 Tn5 shift, same as AlphaGenome pretraining
python chrombpnet/helpers/preprocessing/reads_to_bigwig.py \
    --bam aligned.bam --output signal.bw \
    --chrom-sizes mm10.chrom.sizes \
    --plus-shift 4 --minus-shift -4

# Convert to bedGraph for downstream processing
bigWigToBedGraph signal.bw signal.bedGraph
```

### 2 — Average replicates and normalise

```bash
python scripts/process_cutandtag_bedgraphs.py \
    --config samples.tsv \
    --output-dir data/CUTTag/processed_bedgraph \
    --chrom-sizes data/resources/mm10.chrom.sizes \
    --normalize-to 100000000
```

`samples.tsv` columns: `sample_id`, `cell_type`, `epitope`, `bedgraph_path`.

Output: one `{cell_type}_{epitope}_averaged_normalized.bedgraph` per group, normalised to 100 M insertions.

**Signal requirements:**
- Base-pair resolution, non-negative values
- Normalised to 100 M total insertions per track
- **Do not** apply log, z-score, or any further scaling — the loss function handles transforms internally

---

## Training

Two training scripts available:

1. **`scripts/pytorch_ft.py`** (PyTorch data loading)
   - Dynamic window generation from genome
   - Built-in shift + reverse-complement augmentation
   - **Known issue:** Data corruption after ~1700 batches in multi-GPU setups
     - Occurs with both BedGraph and BigWig formats
     - Signal values spike to 10-4700x expected ranges
     - Related to PyTorch→JAX sharding pipeline, not signal format
   
2. **`scripts/tf_ft.py`** (TensorFlow data loading)
   - Follows DeepMind's official finetuning approach
   - Supports both BigWig and BedGraph (auto-detected)
   - Requires pre-defined BED file with training windows
   - **Known issue:** NCCL AllReduce errors during first batch compilation
     - Related to mesh setup or sharding specifications in multi-GPU training
     - DeepMind confirmed multi-GPU training works (tested on 8 H100s)

**⚠️ Current Status:** Multi-GPU training blocked by NCCL errors. Debugging with `jax.eval_shape` to identify sharding issues.

**Recommended command (TensorFlow pipeline):**
```bash
python scripts/tf_ft.py --strategy head_replace \
    --checkpoint checkpoints/pre_trained/alphagenome-all-folds \
    --fasta data/resources/mm10.fa \
    --signal data/CUTTag/processed_bedgraph/D1MSN_H3K27ac_averaged_normalized.bedgraph \
    --windows-bed exports/peak_safe_windows/D1MSN_H3K27ac_merged_peak_safe_windows.bed \
    --output ./checkpoints/fine_tuned \
    --batch-size 8 --lr 5e-4 --window-size 131072
```

**Debug multi-GPU with eval_shape (DeepMind recommendation):**
```bash
python scripts/debug_with_eval_shape.py --config configs/config_bigwig.yaml
```

**Alternative (PyTorch pipeline):**
```bash
python scripts/pytorch_ft.py --strategy head_replace \
    --checkpoint checkpoints/pre_trained/alphagenome-all-folds \
    --fasta data/resources/mm10.fa \
    --signal data/CUTTag/processed_bedgraph/D1MSN_H3K27ac_averaged_normalized.bedgraph \
    --output ./checkpoints/fine_tuned \
    --batch-size 8 --lr 5e-4 --window-size 131072

# LoRA (parameter-efficient, works with both pipelines)
python scripts/tf_ft.py --strategy lora \
    --checkpoint checkpoints/pre_trained/alphagenome-all-folds \
    --fasta data/resources/mm10.fa \
    --signal data/CUTTag/processed_bedgraph/D1MSN_H3K27ac_averaged_normalized.bedgraph \
    --windows-bed exports/peak_safe_windows/D1MSN_H3K27ac_merged_peak_safe_windows.bed \
    --output ./checkpoints/fine_tuned_lora \
    --batch-size 4 --lr 1e-4 --lora-rank 8
```
    --batch-size 8 --lr 1e-4 --lora-rank 8
```

### Key arguments

| Argument | Default | Notes |
|---|---|---|
| `--strategy` | `head_replace` | `head_replace` or `lora` |
| `--window-size` | 131072 | Must be divisible by 128. AlphaGenome was pretrained on 1 M bp; smaller windows reduce memory |
| `--batch-size` | 8 | Total across all GPUs |
| `--lr` | 5e-4 | Peak LR; 5 % linear warmup then cosine decay |
| `--epochs` | 10 | |
| `--lora-rank` | 8 | LoRA strategy only |
| `--organism` | 1 | 0 = human, 1 = mouse |
| `--no-augment` | — | Disable shift (±1024 bp) + reverse-complement augmentation |

### Training logs

Each run writes to `output/`:
- `train_YYYYMMDD_HHMMSS.log` — full stdout mirror (line-buffered, survives crashes)
- `metrics.csv` — per-batch structured log: `timestamp, epoch, batch, loss, grad_norm, nan_event, skipped`

---

## Multi-GPU

Tested on 8 × NVIDIA RTX 6000 Ada (49 GB each). Parameters are **replicated** across all devices; only the batch dimension is sharded. JAX auto-averages gradients across devices — no `pmap` or manual `pmean` required.

**Configuration:**
- Model parameters (replicated): ~37 GB per GPU
- With `batch-size 8` across 8 GPUs: 1 sample/GPU (1 × 131,072 × 4 float32 ≈ 2 MB input)
- Default settings: `--batch-size 8 --window-size 131072`

**Multi-GPU stability fixes (Feb 2026):**
- Explicit `in_shardings` and `out_shardings` in `jax.jit()` prevent NCCL errors from implicit resharding
- Validation statistics use `jax.device_get()` to avoid NCCL AllReduce operations on sharded arrays
- Both fixes implemented in `models/model.py` and training scripts

**Debugging NCCL errors:**
- Use `scripts/debug_with_eval_shape.py` to validate shapes without GPU execution
- Compare mesh setup with DeepMind's official "Set Up Device Mesh for Data Parallelism" cell
- DeepMind confirmed multi-GPU training works (tested on 8 H100 GPUs)

---

## Checkpoints

Format: [orbax](https://github.com/google/orbax) `StandardCheckpointer`.

- **During training:** `output/epoch_NNN/` (every epoch) and `output/best_checkpoint/` (lowest average epoch loss)
- **head_replace:** saves `params + state` (compatible with `dna_model.create()` API)
  - Optimizer state is NOT saved (will be reinitialized when resuming training)
  - This allows fine-tuned checkpoints to be loaded directly with AlphaGenome's standard API
- **lora:** saves `lora_A, lora_B, scaling, rank, alpha` only

---

## Project structure

```
AG_Finetune/
├── scripts/
│   ├── pytorch_ft.py                # PyTorch data pipeline (dynamic windows, may have corruption)
│   ├── tf_ft.py                     # TensorFlow data pipeline (recommended, requires BED file)
│   ├── process_cutandtag_bedgraphs.py  # Replicate averaging + normalisation
│   ├── inspect_and_export.py        # Inspect AlphaGenome layers; LoRA targeting
│   ├── analyze_batch17.py           # Find first NaN-gradient batch
│   ├── debug_gradnorm_detailed.py   # Per-batch gradient norm diagnostics
│   ├── debug_forward_pass.py        # Forward-pass NaN isolation
│   ├── investigate_nan_gradients.py # Scan for NaN gradients with/without shuffle
│   ├── trace_parameter_drift.py     # Monitor LoRA param statistics per step
│   └── manul_debug_grad_norm.py     # Debug entry point (manual gradient norm)
├── models/
│   └── model.py                     # HeadReplaceAlphaGenomeModel + LoRAAlphaGenomeModel
├── utils/
│   ├── borzoi_loss.py               # Baskerville Poisson + Multinomial NLL loss
│   └── lora_utils.py                # LoRA param init and layer targeting
└── dataset/
    ├── dataset.py                   # CutAndTagDataset (PyTorch, used by pytorch_ft.py)
    └── tf_dataset.py                # TensorFlow data pipeline (used by tf_ft.py)
```

---

## Loss function

`cut_and_tag_loss_baskerville` in `utils/borzoi_loss.py` implements the Baskerville-exact Poisson + Multinomial NLL loss:

1. **Target transform** — `baskerville_target_transform`: scale → sqrt → soft clip at 64 → hard clip at 384
2. **Poisson term** — Poisson NLL on per-track sequence totals
3. **Multinomial term** — Multinomial NLL on the positional distribution within the sequence

Pass **raw signal counts** — do not pre-transform. The loss applies all transforms internally and uses `scaled_predictions_1bp` from the ATAC head.

---

## Numerical stability

Three layered defences against NaN/corrupt-data events:

1. **Dataset clip** (`dataset/dataset.py`) — signal values are hard-capped at 50 000 counts/bp at load time. Physically impossible for 100 M-normalised CUT&Tag data; catches integer-overflow artifacts before they propagate.
2. **Pre-`train_step` guard** (`scripts/train.py`) — if `max(targets) > 50 000` after batching, the batch is logged with `nan_event=True` and skipped before `train_step` is called.
3. **Parameter-commit gate** (`models/model.py`) — `train_step` only writes `new_params` / `new_opt_state` back to `self` when `grad_norm >= 0`. Negative sentinels (`-1` = grad NaN, `-2` = grad Inf, `-3` = param NaN, `-4` = param Inf) leave the stored parameters untouched, preventing a single bad batch from permanently corrupting weights.

All training uses **full float32** (`params=float32,compute=float32,output=float32`). This resolves the bfloat16 forward-NaN instability confirmed against the official DeepMind fine-tuning colab.

---

## Hardware

8 × NVIDIA RTX 6000 Ada Generation (49 140 MiB each), Linux, CUDA 12.
