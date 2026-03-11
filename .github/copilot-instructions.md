# AG_Finetune — Copilot Instructions

Fine-tuning AlphaGenome (450M param JAX/Haiku genomics model) on CUT&Tag data (two strategies: head-replacement and LoRA), on 8-GPU Linux server.

**⚠️ CRITICAL: Always reference the [AlphaGenome official documentation](https://www.alphagenomedocs.com/index.html) when working with the model API. Do NOT make assumptions about checkpoint loading, model initialization, or prediction interfaces - verify against the docs first.**

**Key reference patterns:**
- Checkpoint loading: `params, state = ocp.StandardCheckpointer().restore(checkpoint_path)` (see `alphagenome_research/evals/track_prediction.py:74`)
- Model creation: Use `hk.transform_with_state` with `model_lib.AlphaGenome(metadata)` constructor (see `alphagenome_research/evals/track_prediction.py:79-83`)
- Metadata loading: `metadata_lib.load(organism)` for each organism (see `alphagenome_research/evals/track_prediction.py:76-78`)

---

## Architecture Overview

```
CutAndTagDataPipeline (TF/BigWig) → tf.data → JAX arrays → { HybridAlphaGenomeModel             (NEW best, tf_ft_hybrid.py)
                                                            { FreezeHeadReplaceAlphaGenomeModel  (tf_ft_freeze.py)
                                                            { HeadReplaceAlphaGenomeModel          (full training, tf_ft.py)
                                                            { LoRAAlphaGenomeModel                 (parameter-efficient, tf_ft.py)
                                                                        ↕
                                               models/model.py → cut_and_tag_loss_baskerville → orbax checkpoint
```

**Key components:**
- `models/model.py` — `HeadReplaceAlphaGenomeModel`, `LoRAAlphaGenomeModel`. Both share `.train_step(seq, org, targets, loss_fn)` and orbax `.save()`/`.load()`. Both use fp32 and `optax.clip_by_global_norm(0.5)` + adam.
- `scripts/tf_ft_hybrid.py` — **Recommended training entry.** Two-phase curriculum: Phase 1 head-only (~0.15 s/step), Phase 2 head + LoRA on QKV layers (~428 K extra params). Uses `HybridAlphaGenomeModel` (defined inline). Config: `configs/config_bigwig_hybrid.yaml`.
- `scripts/tf_ft_freeze.py` — Trunk-freeze → full-unfreeze curriculum. Phase 2 trains all 450M params (~28 s/step). Config: `configs/config_bigwig_freeze.yaml`.
- `scripts/tf_ft.py` — Full-parameter training from epoch 1 (no freeze). Slower but simpler. Config: `configs/config_bigwig.yaml`.
- `utils/lora_utils.py` — `initialize_lora_params`, `apply_lora_to_params`: LoRA uses `ΔW = B @ A * (alpha/rank)`. Haiku param keys are flat strings like `'alphagenome/transformer_tower/mha_block_0/q_layer'`.
- `utils/borzoi_loss.py` — `cut_and_tag_loss_baskerville`: Borzoi-style Poisson + multinomial NLL. Applies `baskerville_target_transform` internally — **pass raw counts, do not pre-transform**.
- `scripts/evaluate.py` — Standalone evaluation script. Loads a fine-tuned checkpoint on a single GPU, runs sanity-check inference, generates training curves, computes Pearson r / Spearman ρ / MSE / MAE, and writes PNGs + `eval_metrics.csv` to the run directory. Launch via CLI or as a subprocess from a notebook.
- `scripts/pytorch_ft.py` — PyTorch pipeline (**not recommended for multi-GPU** due to data corruption; see Troubleshooting).

---

## Multi-GPU Data Parallelism

Parameters are **replicated** (not sharded); only the batch dimension is sharded. The `Mesh` must be created **before** the model is loaded.

```python
mesh = Mesh(jax.devices(), ('batch',))
batch_sharding    = NamedSharding(mesh, P('batch', None, None))  # [batch, seq_len, 4]
organism_sharding = NamedSharding(mesh, P('batch'))              # [batch]
signal_sharding   = NamedSharding(mesh, P('batch', None))        # [batch, seq_len]
param_sharding    = NamedSharding(mesh, P())                     # replicated on all devices
```

JAX auto-averages gradients across devices on the backward pass when inputs are sharded. No explicit `pmap` or `pmean` needed.

---

## Precision Policy

All training now uses **full float32** (`params=float32,compute=float32,output=float32`) — hardwired for `HeadReplaceAlphaGenomeModel`, configurable via `--precision-policy` for `lora`. This resolves the bfloat16 forward-NaN issue (confirmed by comparison with the official DeepMind fine-tuning colab, which also uses fp32 explicitly).

Gradient clipping is re-enabled at `0.5` (matching DeepMind colab) in both model classes. Gradient sanitization (`NaN/Inf → 0`) remains as a safety net.

When collecting scalar outputs from jitted+sharded functions, always use `jax.device_get(val)` (not bare `float()` or `np.asarray()`) to avoid `AttributeError: 'UnspecifiedValue' object has no attribute 'is_fully_replicated'`.

---

## Dataset & Signal Format

- Signal: BigWig (primary) or BedGraph, base-pair resolution, normalized to **100M total insertions per track**. The TensorFlow pipeline reads BigWig directly via `pyBigWig`.
- Raw counts are passed as `float32` targets; the loss function transforms them internally — **do not pre-transform targets**.
- **AlphaGenome target transform:** `utils/borzoi_loss.py` implements `alphagenome_targets_scaling` matching the paper exactly:
  ```
  targets = targets / track_mean           # normalize by genome-wide non-zero mean
  targets = where(targets > 10, 2*sqrt(targets*10) - 10, targets)  # smooth clip
  ```
  `track_mean` is computed at training startup by `compute_track_mean_from_bigwig()` (genome-wide weighted mean of non-zero BigWig values) and frozen into `loss_fn` via `functools.partial`. The inverse `alphagenome_predictions_scaling(preds, track_mean)` converts model outputs back to count space for evaluation. Both are in `utils/borzoi_loss.py`. The training script uses `cut_and_tag_loss_alphagenome`; the old `cut_and_tag_loss_baskerville` (Baskerville sqrt transform, no `track_mean`) is kept for backward compatibility only.
- `organism_id`: `0` = human, `1` = mouse.
- `window_size` must be divisible by 128. AlphaGenome was pretrained on 1,048,576 bp (2^20^); fine-tuning uses 131,072 bp (2^17^).
- CUT&Tag is conceptually identical to ATAC-seq: use the `atac` prediction head at 1 bp resolution.
- `windows_bed` is **optional**. If omitted, the pipeline generates non-overlapping `window_size` bp windows across all canonical chromosomes from `chrom_sizes` (requires `chrom_sizes` to be set). This allows the model to learn histone-mark signal patterns genome-wide, not just near peaks.

---

## Training Commands

⚠️ **Always prepend `LD_PRELOAD`** to all multi-GPU commands — resolves the jaxlib 0.9.0 NCCL dual-instance conflict (see Troubleshooting).

```bash
# RECOMMENDED: Hybrid curriculum (head-only Phase 1, then head+LoRA Phase 2)
LD_PRELOAD=/home/ray/miniforge3/envs/alphagenome/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2 \
    python scripts/tf_ft_hybrid.py --config configs/config_bigwig_hybrid.yaml

# Alternative: Trunk-freeze → full-unfreeze (head-only Phase 1, then all-param Phase 2)
LD_PRELOAD=/home/ray/miniforge3/envs/alphagenome/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2 \
    python scripts/tf_ft_freeze.py --config configs/config_bigwig_freeze.yaml

# Alternative: Full-parameter training from epoch 1 (no freeze)
LD_PRELOAD=/home/ray/miniforge3/envs/alphagenome/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2 \
    python scripts/tf_ft.py --config configs/config_bigwig.yaml

# Config files:
# configs/config_bigwig_hybrid.yaml — Hybrid curriculum (recommended)
# configs/config_bigwig_freeze.yaml — Trunk-freeze → full training
# configs/config_bigwig.yaml        — Full training from epoch 1
```

Baseline validation on human ENCODE data (official DeepMind pipeline):
```bash
LD_PRELOAD=/home/ray/miniforge3/envs/alphagenome/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2 \
    python scripts/ag-colab_ft.py --num-steps 5 --batch-size 1
```

---

## Completed Training Runs

| Run directory | Strategy | Epochs | Best loss | Notes |
|---|---|---|---|---|
| `fine_tuned_hybrid_freeze1_2026-02-28_14-00-57` | Hybrid | 1 frozen + 2 LoRA | 0.0372 | Quick test run |
| `fine_tuned_hybrid_freeze2_2026-02-28_16-26-59` | Hybrid | 2 frozen + 8 LoRA | 0.000138 | Full run |
| `fine_tuned_hybrid_freeze1_2026-03-01_13-35-58` | Hybrid | 1 frozen + 2 LoRA | 0.000288 | First run with corrected AlphaGenome `targets_scaling`; track_mean=1.171128 |
| `fine_tuned_hybrid_freeze2_2026-03-01_14-50-31` | Hybrid | 2 frozen + 8 LoRA | 0.064421 | Full 10-epoch run; track_mean=1.171128 |

All runs are under `checkpoints/` (symlink → `/celestial/ray/model_checkpoints/AlphaGenome`). Each run contains `metrics.csv`, `train.log`, `stderr.log`, `training_config.json`, per-epoch `epoch_NNN/` checkpoints, and `best_checkpoint/`.

**Evaluation notebooks:** `notebooks/evaluate_freeze1_2026-02-28.ipynb`, `notebooks/evaluate_freeze2_2026-02-28.ipynb`, `notebooks/evaluate_freeze1_2026-03-01.ipynb`, and `notebooks/evaluate_freeze2_2026-03-01.ipynb` — cover training curves (sections 1–5), model loading (section 6), and results display (section 8). Inference is delegated to `scripts/evaluate.py` launched as a subprocess (section 7).

---

## Checkpointing

All checkpoints use **orbax** (`orbax.checkpoint.StandardCheckpointer`). Pickle is fully replaced.

- `LoRAAlphaGenomeModel.save_lora(dirpath)` / `.load_lora(dirpath)` — saves `lora_A`, `lora_B`, `scaling`, `rank`, `alpha`.
- `HeadReplaceAlphaGenomeModel.save(dirpath)` / `.load(dirpath)` — saves `params + state` (compatible with `dna_model.create()` API).
  - **IMPORTANT:** Optimizer state is NOT saved. This makes checkpoints compatible with AlphaGenome's standard loading API.
  - When resuming training via `.load()`, optimizer state is reinitialized from scratch.

Checkpoints are written per epoch to `output_dir/epoch_NNN/` and best checkpoint to `output_dir/best_checkpoint/`.

**Loading fine-tuned checkpoints:**
```python
# Direct loading with AlphaGenome API (works with head_replace checkpoints)
model = dna_model.create(
    checkpoint_path='checkpoints/fine_tuned/best_checkpoint',
    organism_settings=None,
    device=None
)
```

---

## Evaluation

`scripts/evaluate.py` — self-contained CLI script for post-training evaluation on a single GPU.

```bash
# Run directly (single GPU, no LD_PRELOAD needed)
CUDA_VISIBLE_DEVICES=0 conda run -n alphagenome \
    python scripts/evaluate.py \
    --run-dir checkpoints/fine_tuned_hybrid_freeze2_2026-02-28_16-26-59
```

Key design decisions:
- Sets `CUDA_VISIBLE_DEVICES=0` automatically if not already set — avoids NCCL deadlock without `LD_PRELOAD`.
- Uses `print(..., flush=True)` and `matplotlib.use('Agg')` — safe to run outside a display session.
- Checkpoint is loaded using the same 1-track `AlphaGenomeOutputMetadata(atac=track_df)` used during training.
- **TrackData API:** `preds.atac` is already a `TrackData` object (unwrapped). Access predictions as `preds.atac.values` (shape `[window_size, num_tracks]`). Index the single track as `preds.atac.values[:, 0]`. **`scaled_predictions_1bp` does NOT exist on `TrackData`** — that key exists only in the raw predictions dict before wrapping.

Outputs written to `<run-dir>/`:

| File | Contents |
|---|---|
| `training_curve.png` | Phase 1 + 2 step-level scatter + epoch bars |
| `per_epoch_curves.png` | Per-epoch zoom plots |
| `eval_metrics.csv` | Pearson r / Spearman ρ / MSE / MAE per eval window |
| `pred_vs_obs_tracks.png` | Predicted vs observed signal overlay |
| `eval_correlation.png` | Hex scatter + per-window correlation bar chart |
| `eval_peak_tracks.png` | Top-N high-signal windows |

---

## Hybrid Head+LoRA Curriculum (recommended)

Implemented in `scripts/tf_ft_hybrid.py` using `HybridAlphaGenomeModel`.

**Phase 1 — trunk frozen (`freeze_epochs` epochs):**
- Identical to tf_ft_freeze.py Phase 1. Only the 9,224-param ATAC head receives gradients.
- Per-step time: **~0.15 s**. Loss converges: 1.2 → ~0.07 within 500 steps.

**Phase 2 — head + LoRA (remaining epochs):**
- `model.activate_lora()` initialises LoRA A/B matrices for the Q/K/V projections of all 9 transformer blocks.
- Trunk base weights remain frozen (`trunk_params` in Python closure — JAX never differentiates through them).
- Trainable params: head (~9 K) + LoRA A/B (~428 K at rank 8) = **~437 K total (0.10% of 450M)**.
- Per-step time: intermediate (estimated ~1–5 s, much less than full ~28 s).
- Fresh combined optimizer over `{'head', 'lora_A', 'lora_B'}` — no momentum bleed from Phase 1.

**Why hybrid beats full-unfreeze for histone marks:**
H3K27ac/H3K27me3 patterns differ substantially from ATAC-seq. LoRA adapts the attention layers for the new biochemistry while the convolutional trunk stays in the general genomic representation space, reducing catastrophic forgetting.

**Key config knobs** in `configs/config_bigwig_hybrid.yaml`:
- `freeze_epochs` — Phase 1 length
- `lora_rank` — increase to 32/64 if Phase 2 plateaus early
- `lora_alpha` — effective scaling = `alpha/rank`

**Checkpoint format:** `save()` writes two sub-directories:
- `model/` — LoRA deltas merged into trunk params, compatible with `dna_model.create()` API
- `lora_raw/` — raw A/B matrices for training resumption
- `lora_metadata.json` — rank, alpha, target layers, current phase

---

## Trunk-Freeze Curriculum (recommended for head_replace)

Implemented in `scripts/tf_ft_freeze.py` using `FreezeHeadReplaceAlphaGenomeModel`.

**Phase 1 — trunk frozen (`freeze_epochs` epochs):**
- Only the 9,224-param ATAC head receives gradients. JAX never differentiates through the trunk (captured in Python closure, not a `value_and_grad` argument).
- Per-step time: **~0.15 s** (vs ~28 s full training) — roughly 180× faster.
- Compilation time: ~108 s (vs ~876 s full training) — 8× faster.
- Loss descends rapidly: 1.2 → 0.07 within 500 steps (~75 s wall time).
- Use `freeze_epochs: 2` to give the head more time to converge before trunk gradients are introduced.

**Phase 2 — trunk unfrozen (remaining epochs):**
- `model.unfreeze()` merges trunk+head params and initialises a **fresh full-parameter optimizer** (no momentum bleed from Phase 1).
- JAX recompiles the full training step on the first batch of this phase.
- Per-step time returns to ~28 s.
- Expected loss trajectory: brief jump upward at phase transition (larger param space + fresh optimizer), then descent below the Phase 1 plateau.

**Key config knob:** `freeze_epochs` in `configs/config_bigwig_freeze.yaml`. Set to `0` to disable freeze (equivalent to `tf_ft.py`).

**Checkpoint compatibility:** `FreezeHeadReplaceAlphaGenomeModel.save()` merges trunk+head before saving in frozen phase — output format is identical to `HeadReplaceAlphaGenomeModel.save()` and compatible with `dna_model.create()` loading API.

---

## LoRA Configuration (lora strategy only)

- **Target layers**: Q, K, V projections in all 9 transformer blocks (`get_attention_qkv_layers()` in `utils/lora_utils.py`).
- **Rank 8, alpha 16** → 428,544 trainable params (0.10% of 450M).
- `lora_A` → scaled Kaiming normal init; `lora_B` → zero init (identity at start).
- `apply_lora_to_params` merges LoRA into a copy of base params at forward time; base params are never mutated.
- `loss_fn` is passed as `static_argnames` to `jax.jit` because it is a Python callable, not a JAX array.

---

## Debug Scripts Map

| Script | Purpose |
|---|---|
| `debug_with_eval_shape.py` | **[PRIMARY]** Validate training logic with `jax.eval_shape` — identifies shape mismatches without running GPU kernels (DeepMind recommendation) |
| `debug_signal_values.py` | Inspect PyTorch DataLoader signal values for corruption (isolates data pipeline from JAX training) |
| `inspect_and_export.py` | Export AlphaGenome layer names suitable for LoRA targeting; output in `exports/linear_layers_analysis.json` |

Archived scripts (no longer needed after NCCL resolution) are in `/celestial/ray/ARCHIVED/AG_Finetune/scripts/`.

---

# Supplementary Methods

This section documents additional (but crucial) details for finetuning the [AlphaGenome](https://github.com/google-deepmind/alphagenome_research) model.

The blockquoted (>) contents are excerptions from the supplementary methods of the paper. Fenced codes (```) are pseudocodes from the paper. Between the excerptions I give my remarks in italicized texts.

> ## Data acquisition and processing

> ### Training Data
> To generate training targets for AlphaGenome, RNA-seq and epigenomic datasets were sourced from the ENCODE7, FANTOM57, and GTEx consortia. Specifically, this included RNA-seq, PRO-cap, DNase-seq, ATAC-seq, transcription factor (TF) and histone ChIP-seq data from ENCODE; RNA-seq data from GTEx; CAGE data from Fantom portals; and genomic contact maps from the 4D Nucleome portal.

*We're trying to finetune the AlphaGenome model on CUT&Tag dataset using LoRA. Biochemically, the CUT&Tag is conceptually similar to ATAC-seq. Both use Tn5 enzyme to profile the genome, so we should use the ATAC-seq head at base-pair resolution.*

> #### DNase and ATAC Data
> DNase-seq and ATAC-seq datasets were obtained from the ENCODE portal. Strict Fraction of Reads in Peaks (FRiP) score thresholds were applied (>10% for both assays, determined via manual inspection to balance coverage versus quality), along with different minimum read length requirements (≥36 nt for DNase-seq, ≥45 nt for ATAC-seq, following ENCODE guidelines).
> Instead of using ENCODE pre-processed p-value or fold-change signal bigWigs, which can obscure base-resolution information, we ingested the raw alignment (BAM) files for individual replicates. These BAM files were converted to base-resolution count bigWig files using the reads_to_bigwig.py script (from the ChromBPNet repository29, https://github.com/kundajelab/chrombpnet/blob/master/ chrombpnet/helpers/preprocessing/reads_to_bigwig.py), with default parameters. These files record counts of Tn5 transposase insertions (for ATAC-seq) or DNase-I cleavage sites (for DNaseseq) at each base. This approach preserves the base-resolution nature of the data and facilitates decoupling enzyme cut bias from true signal by applying appropriate read shifts. Specifically, shifts of +4/-4 for ATAC-seq and 0/+1 for DNase-seq were applied to correct for the misalignment of strand-specific Tn5 and DNase-I position weight matrices (PWMs), respectively.
> The assays were grouped by biosample ontology CURIE. The resulting base-resolution count BigWigs within each ontology group were then averaged. Following this averaging step, these consolidated tracks underwent post-aggregation normalization, rescaling the signal such that the total counts summed to 100 million insertions per track.

*We have implemented the following preprocessing steps to the CUT&Tag bigwigs with the chrombpnet `reads_to_bigwig.py` script:*
*- base-pair resolution*
*- normalize to 100M*
*- +4/-4 shift*
*The TensorFlow training pipeline reads these BigWig files directly via `pyBigWig`.*

> ### Final data representations
> Processed genomic tracks, serving as model prediction targets, were converted to brain floating point (bfloat16) format for numerical efficiency and stored in z-standard compressed sharded matrices. Most data tracks were maintained at base-pair resolution, except for ChIPseq (TF and Histone) tracks, which represent fold-change values, and were stored as base-resolution cumulative sums. This strategy allows for efficient querying of their average signal at 128 bp resolution.
> The model directly uses the track values resulting from the upstream processing steps, which typically represent normalized read or insertion counts (often scaled to a total of 100 million signals per track for sequencing assays) or fold-change enrichments (for ChIP-seq). No additional scaling transformations (e.g., log-scaling or z-score normalization across tracks) are applied before model input. Predicted splice junction counts are an exception; while their values are additionally scaled (as described in the training methods), this does not affect their primary use in calculating ratios for relative splice site assessment.

*We implemented the normalization and aggregation in a stand alone script (`scripts/process_cutandtag_bedgraphs.py`)*
*Note that we should NOT do scaling transformation to the final data.*

> ## Model

> AlphaGenome is a deep learning based model that processes a 1 Mb DNA sequence along with a species identifier (human or mouse) to predict a diverse array of genomic features. These outputs include 1D tracks representing signals such as chromatin accessibility, TF binding, and gene expression at various resolutions; 2D chromosomal contact maps; and specific splicing-related predictions like splice site probabilities, splice site usage (SSU), and splice junction counts.
> AlphaGenome employs an encoder-decoder framework that integrates convolutional and transformer components, drawing inspiration from models such as Enformer and Borzoi. It introduces several key architectural features, including a U-Net-style decoder incorporating residual (skip) connections from the encoder. The model processes information through five core components: (1) a sequence encoder, (2) a transformer tower, (3) pairwise interaction blocks, (4) a sequence decoder, and (5) task-specific output heads. AlphaGenome has approximately 450 million trainable parameters (20% in the encoder, 28% in the sequence transformer, 15% in the pairwise blocks, 25% in the decoder, and 12% in the output embedding and prediction heads). The model was implemented using JAX and Haiku.

*The `inspect_and_export.py` script can be used to inspect the internal architecture of the AlphaGenome and give recommendations on best trainable layers. Analysis are exported to `exports/linear_layers_analysis.json`.*

> ### Sequence Encoder
> The sequence encoder progressively downsamples the input DNA sequence from 1 bp resolution (length 1 Mb or 2^20^ bp) to 128 bp resolution embeddings (length 8192 or 2^13^) over 7 stages. The number of feature channels increases from 768 up to 1536 to enhance representational capacity. This downsampling is achieved through repeated convolutional blocks, each followed by max-pooling with a stride of 2. The corresponding pseudocode is as follows:

```
def conv_block(x: Array, num_channels: int, width: int = 5) -> Array:
    x = RMSBatchNorm(x)
    x = GeLU(x)
    if width == 1:
        x = Linear(num_channels=num_channels)(x)
    else:
        x = StandardizedConv1D(num_channels=num_channels, width=width)(x)
    return x

def dna_embedder(x: Array):
    out = Conv1D(num_channels=768, width=15)(x)
    return out + conv_block(out, num_channels=768)
  
def downres_block(x: Array) -> Array:
    # Increase number of channels by 128 and apply skip connection by padding.
    out = conv_block(x, num_channels=x.shape[-1] + 128)
    out = out + Pad(x, [(0, 0), (0, 0), (0, 128)])
    return out + conv_block(out, num_channels=out.shape[-1])

def sequence_encoder(x: Array) -> tuple[Array, map[str, Array]]:
    intermediates = {}
    for bin_size in [1, 2, 4, 8, 16, 32, 64]:
        block = dna_embedder if bin_size == 1 else downres_block
        x = block(x)
        intermediates[f'bin_size_{bin_size}'] = x
        x = MaxPool(pool_size=2)(x)
    return x, intermediates
```

> The core component of the above code is a conv_block, which applies root mean square batch normalization (RMSBatchNorm—same as BatchNorm with learned scale and offset, but without shifting by the sample mean), a GeLU activation, and then a standardized 1D convolution (StandardizedConv1D) with a kernel width of 5. Following Brock et al., these convolutions employ scaled weight standardization – re-parameterizing by standardizing and scaling the weights – which we found effective for stabilizing activation magnitudes. Each RMSBatchNorm layer maintains an exponential moving average (EMA, decay 0.9) of per-channel variance during training; this EMA variance is used for normalization during inference. To facilitate training and gradient flow, residual skip connections are added around each convolutional block. When channel dimensions differ across a residual connection (as in downres_block), the input is padded with zeros before addition. All convolution layers used “same” padding.
> The initial stage uses the dna_embedder block, which applies a wider 1D convolution (kernel width 15, embedding to 768 channels), similar to Enformer, followed by a residual conv_block. The subsequent six downres_block stages each apply two conv_block layers (the first increasing the channel count by 128) with skip connections. Intermediate representations from each of the 7 stages, corresponding to resolutions of 1 bp, 2 bp, 4 bp, . . . , 64 bp before the max-pooling step, are stored for use as U-net skip connections in the sequence decoder. The final output of the encoder has a resolution of 128 bp, sequence axis length 8192 (1 Mb / 128 bp) and 1536 channels (768 + 6 * 128).

> ### Transformer Tower

> Following the encoder, a transformer tower processes the 128 bp resolution sequence embeddings to model long-range interactions across the full 1 Mb input context. The tower consists of 9 stacked transformer blocks. Each block contains a multi-head attention (MHA) layer followed by a two-layer perceptron (MLP) layer, both wrapped in residual connections. We use standard array dimension conventions where ‘B’ denotes batch size, ‘S’ sequence length (S = 1M b/128 bp = 8192), ‘C’ channels, ‘P’ pairwise sequence length, and ‘F’ pairwise channels. In pseudocode, these components can be written as:

```
def mha_block(x: Array[B, S, C], attention_bias: Array[B, 8, S, S]) -> Array[B, S, C]:
    x = RMSBatchNorm(x)
    # Multi-query attention: 8 query heads, 1 shared key/value head. 
    # Each query and key head has 128 channels. The value head has 192 channels.
    q = LayerNorm(Linear((8, 128), with_bias=False)(x), axis=-1)
    k = LayerNorm(Linear((1, 128), with_bias=False)(x), axis=-1)
    v = LayerNorm(Linear((1, 192), with_bias=False)(x), axis=-1)
    # Apply RoPE with max_position = S = 8192 (1 Mb / 128 bp).
    q = apply_rope(q, max_position=8192)
    k = apply_rope(k, max_position=8192)
    attention_logits = Einsum('bshc,bS1c->bhsS', q, k) / sqrt(128)
    # Add bias from pairwise activations and soft-clip logits in [-5, 5].
    attention_logits = Tanh((attention_logits + attention_bias) / 5.0) * 5.0
    attention_weights = Softmax(attention_logits, axis=-1)
    y = Einsum('bhsS,bS1c->bshc', attention_weights, v)
    # Reshape and project back to input channels C = 1536.
    y = Linear(x.shape[-1])(y.reshape(y.shape[:2] + (-1,)))
    return Dropout(RMSBatchNorm(y))

def mlp_block(x: Array[B, S, C]) -> Array[B, S, C]:
    y = RMSBatchNorm(x)
    y = Linear(2 * x.shape[-1])(y)
    y = Dropout(ReLU(x))
    y = Linear(x.shape[-1])(y)
    return Dropout(RMSBatchNorm(y))

def attention_bias_block(x: Array[B, P, P, F]) -> Array[B, 8, S, S]:
    x = GeLU(RMSBatchNorm(x))
    x = Linear(8, with_bias=False)(x)
    # Repeat attention bias to S = 8192 length (factor S/P = 8192/512 = 16).
    return Moveaxis(Repeat(x, 16, axis=(1, 2)), 3, 1)

def transformer_tower(x: Array[B, S, C]) -> tuple[Array[B, S, C], Array[B, P, P, F]]:
    pair_x = None
    for i in range(9):
        if i % 2 == 0:
            pair_x = pair_update_block(x, pair_x)
        x = x + mha_block(x, attention_bias=attention_bias_block(pair_x))
        x = x + mlp_block(x)
    return x, pair_x
```

> Before every second MHA block, pairwise representations at 2048 bp resolution (P = 1M b/2048 bp = 512) are initialized or updated based on the sequence embeddings (detailed in Pairwise blocks section). They are primarily used for contact map prediction but also provide a bias term to the MHA layers. The attention_bias_block function projects the pairwise channels (F) to match the number of attention heads (8) and then repeats the result 16 times (2048 bp / 128 bp) along the two sequence axes to match the sequence length dimension. This bias is added directly to the attention logits before the softmax, analogous to techniques used in AlphaFold2.
> Our MHA implementation (mha_block) uses multi-query attention, where key (k) and value (v) projections are shared across all 8 attention heads, reducing computational cost. We apply smooth clipping to the attention logits, constraining them to the range [-5.0, 5.0] using a scaled hyperbolic tangent function before the Softmax. Relative positional information is incorporated using Rotary Position Embeddings (RoPE) applied to query and key vectors. We use a modified calculation for the inverse frequencies compared to the standard RoPE formulation (apply_rope). This modification reduces the density of rotational frequencies corresponding to short relative distances and is applied consistently throughout the model. The core of this implementation is detailed in pseudocode:

```
def apply_rope(x: Array, max_position: int, positions: Array | None = None):
    positions = positions or Arange(x.shape[1])
    num_freq = x.shape[-1] // 2
    freq = 1.0 / (Arange(num_freq) + Geomspace(1, max_position - num_freq + 1, num_freq))
    theta = Repeat(Einsum('...s,f->...sf', positions, freq), 2, axis=-1)
    x_rotated = Stack((-x[..., 1::2], x[..., ::2]), axis=-1).reshape(x.shape)
    return x * Cos(theta) + x_rotated * Sin(theta)
```

> The MLP block (mlp_block) follows standard transformer design, using ReLU activation and expanding the number of channels by a factor of 2 in the hidden layer. Dropout is applied within the MHA and MLP blocks during training only, with rates of 0.3 for pre-training and 0.1 for distillation.

> ### Pairwise Blocks

*Skipping since our task doesn't involve contact maps.*
 
> ### Sequence Decoder

> The sequence decoder mirrors the encoder’s structure but operates in reverse: it progressively upsamples the 128 bp resolution embeddings (output from the transformer tower, with 1536 channels) back to the original 1 bp resolution, while simultaneously decreasing the number of feature channels down to 768. This is achieved through 7 iterative stages, each applying an upres_block module. As is standard in UNet architectures, the decoder incorporates skip connections, adding embeddings (intermediates) stored at corresponding resolutions during the encoder pass. The corresponding pseudocode is as follows:

```
def upres_block(x: Array, unet_skip: Array) -> Array:
    num_channels = unet_skip.shape[2]
    out = conv_block(x, num_channels) + x[:, :, :num_channels]
    out = Repeat(out, 2, axis=1) * GetParameter('residual_scale', (1,), init=0.9)
    out += conv_block(unet_skip, num_channels, width=1)
    return out + conv_block(out, num_channels)

def sequence_decoder(x: Array, intermediates: map[str, Array]) -> Array:
    for bin_size in [64, 32, 16, 8, 4, 2, 1]:
        x = upres_block(x, intermediates[f'bin_size_{bin_size}'])
    return x
```

> Each upres_block takes the output from the previous stage and the corresponding resolution skip connection from the encoder. First, a conv_block reduces the channel dimension of the embedding to match that of the skip connection. As usual, the convolution is paired with a residual connection which handles the reduction in channel dimension by cropping. The result is then upsampled by a factor of two along the sequence length axis via repetition and scaled by a learnable coefficient specific to that resolution level. Concurrently, the incoming skip connection is processed through its own conv_block (using a kernel width of 1). This processed skip connection is then added to the scaled, upsampled representation. Finally, this combined tensor passes through one more conv_block, again including an internal residual connection, before being passed to the next stage or returned as the final 1 bp resolution sequence embedding.

> ### Output Heads

> The AlphaGenome architecture, after processing the input DNA through its encoder, transformer, and decoder stages, produces key internal representations at multiple resolutions. These learned embeddings (specifically the 1D embeddings at 1 bp and 128 bp resolution along the sequence, and the 2D pairwise embedding at 2048 bp resolution) form the basis for predicting a diverse array of genomic features. Predictions are adapted to the specific organism (human or mouse) by incorporating learned, organism-specific embeddings within these functions. The following pseudocode illustrates the core data flow that generates these embeddings via the main model components (sequence_encoder, transformer_tower, sequence_decoder) and output embedding functions (output_embedder, output_pair).

```
def output_embedder(x: Array, organism_index: int, skip_x: Array | None = None) -> Array:
    x = Linear(2 * x.shape[2])(x)
    if skip_x is not None:
        skip_x = Linear(x.shape[2], with_bias=False)(skip_x)
        x += Repeat(skip_x, x.shape[1] // skip_x.shape[1], axis=1)
    return GeLU(RMSBatchNorm(x) + Embedding(x.shape[2])(organism_index))

def output_pair(x: Array, organism_index: int) -> Array:
    x = (x + Swapaxes(x, 1, 2)) / 2.0 # Symmetrize.
    return GeLU(RMSNorm(x) + Embedding(128)(organism_index))

def model_embeddings(input_dna: Array, organism_index: int) -> tuple[Array, Array, Array]:
    trunk, intermediates = sequence_encoder(input_dna)
    trunk += Embedding(1536)(organism_index)
    trunk, pair_activations = transformer_tower(trunk)
    x = sequence_decoder(trunk, intermediates)
    embeddings_128bp = output_embedder(trunk, organism_index)
    embeddings_1bp = output_embedder(x, organism_index, embeddings_128bp)
    embeddings_pair = output_pair(pair_activations, organism_index)
    return embeddings_1bp, embeddings_128bp, embeddings_pair
```

> Beyond the organism-specific embeddings applied during the generation of these shared representations (embeddings_1bp, embeddings_128bp, embeddings_pair), each output head itself learns a distinct set of parameters for human and mouse predictions. The subsequent sections detail the specific parameterization and loss function defined for each prediction modality. The total training loss for AlphaGenome is the sum of these losses defined by each head with no additional weighting coefficients.

> #### RNA-seq, CAGE, ATAC, DNase and PRO-Cap Output Heads

*This section is critical to our task.*

> Predictions for these assays are generated from both the 1 bp and 128 bp resolution embeddings. Each resolution uses a dedicated head consisting of a linear layer mapping the input channel dimension to the number of target tracks for that assay type. A softplus activation is applied to the linear layer’s output, which is then multiplied by a learnable, per-track positive scaling factor, ensuring non-negative outputs. Strand-specific readouts are treated as distinct output tracks. Each head predicts tracks as in the pseudocode:

```
def tracks_scaled_predictions(x: Array[S, C], num_tracks: int) -> Array[S, num_tracks]:
    x = Linear(num_tracks)(x)
    scale = GetParameter('scale', (num_tracks,), init=0.0)
    return Softplus(x) * Softplus(scale)
```

> To improve numerical stability, experimental target tracks undergo a scaling procedure before loss calculation (targets_scaling). Tracks are first normalized by dividing by the mean of their non-zero values (pre-calculated per track across the entire genome during dataset generation). High values are then dampened using square-root-based smooth clipping. For RNA-Seq tracks only, an additional power transformation is applied for further compression. These scaling and clipping operations are reversed (predictions_scaling) when evaluating model predictions against the original experimental data. The scaling and inverse scaling functions are:

```
def targets_scaling(
    targets: Array[S, C], track_means: Array[C], apply_squashing: bool
) -> Array[S, C]:
    targets = targets / track_means
    if apply_squashing: # Applied RNA-seq tracks only.
        targets = targets ** 0.75
    return Where(targets > 10.0, 2 * Sqrt(x * 10.0) - 10.0, targets)

def predictions_scaling(
    x: Array[S, C], track_means: Array[C], apply_squashing: bool
) -> Array[S, C]:
    x = Where(x > 10.0, (x + 10.0) ** 2 / (4 * 10.0), x)
    if apply_squashing:
        x = x ** (1.0 / 0.75)
    return x * track_means
```

> The training loss follows the approach of Borzoi, calculated as a weighted sum of Poisson and Multinomial negative log-likelihoods (NLL). For each track, the relevant sequence axis (e.g., 2^20^ bins for 1 bp, 8192 bins for 128 bp resolution outputs) is divided into 8 equal segments. Within each segment, the loss combines two components:

> 1. A Multinomial NLL term comparing the predicted distribution of counts across bins to the target distribution, which ensures the model learns the relative values and overall shape of the predictions within each segment. This multinomial component is weighted by a factor of 5.0, as in Borzoi.
> 2. A Poisson NLL term comparing the sum of predicted and target counts for each segment, which encourages the model to learn the accurate overall count (regardless of shape). This term is scaled inversely by the segment length (multinomial_resolution = 217).

> Calculating the loss over 8 segments, rather than the full sequence, mitigates numerical instability (using smaller segments was found to empirically degrade performance). The loss function is implemented as shown in the pseudocode:

```
def multinomial_loss(
    x: Array[S, C], targets: Array[S, C], multinomial_resolution: int
) -> Array[]:
    x = x.reshape((-1, multinomial_resolution, x.shape[-1]))
    targets = targets.reshape((-1, multinomial_resolution, targets.shape[-1]))
    sum_pred = Sum(x, axis=1, keepdims=True)
    sum_target = Sum(targets, axis=1, keepdims=True)
    poisson_loss = Sum(sum_pred - sum_target * Log(sum_pred + 1e-7))
    multinomial_prob = x / (sum_pred + 1e-7)
    positional_loss = Sum(-targets * Log(multinomial_prob + 1e-7))
    return (poisson_loss / multinomial_resolution + 5.0 * positional_loss)
```

> For RNA-seq tracks, an additional loss component, inspired by Decima, promotes tissue-specific expression patterns. First, predicted and scaled target counts are aggregated within annotated gene boundaries found in the input interval and subsequently normalized by gene length to yield per-gene counts. These per-gene counts are then used to compute a loss across the tissue/cell type dimension for each gene. This loss calculation mirrors the structure of the main track loss described above: it combines a Poisson NLL term on the total normalized expression per gene across all tissues and a Multinomial NLL term on the distribution across tissues. The Multinomial term within this calculation is weighted by a factor of 5.0. This auxiliary gene-level loss contributes to the model’s total training loss with an overall weight of 0.1.
> After all model training runs had finished, we identified an implementation error affecting the RNAseq output head’s tissue-specific gene expression loss. This error resulted in incorrectly flipped gene boundary strands for half of the training intervals. We have since verified that, due to the low relative weight and magnitude of this auxiliary loss term, the error’s impact on final model performance was within the range of expected noise, as measured by the Pearson correlation of log-fold change in gene expression.

## Borzoi loss

*Neither [Borzoi repo](https://github.com/calico/borzoi) nor [Borzoi-pytorch repo](https://github.com/johahi/borzoi-pytorch) explains how loss function is implemented. I can only find the following paragraph from the Borzoi paper:*
> We trained the model in a multi-task setting to predict coverage for all assays from one species, with a species-specific head attached to the shared model trunk. During training, we alternated human and mouse training batches by dynamically swapping in the corresponding species-specific head. To avoid less accurate predictions on the sequence boundaries (owing to asymmetric visibility), we cropped from each side to focus the loss computation on the center 196,608 bp. We used a Poisson loss function but decomposed the loss analogous to BPnet to separate magnitude and shape terms. Having independent Poisson distributions at each sequence position is mathematically equivalent to a single Poisson distribution representing their sum, followed by allocating the counts to sequence positions using a multinomial distribution. Thus, we apply a Poisson loss on the sum of the observed and predicted coverage and a multinomial loss on the normalized observed and predicted coverage across the sequence length. This decomposition allows us to weight the multinomial shape loss by a greater amount (five times), which we found boosts performance.

## Local Hardware Settings

1. The GPUs we use are 8 * NVIDIA RTX 6000 Ada, with 49140MiB memory each.
2. Loading one AlphaGenome model takes ~36931MiB GPU memory. Always make sure one model is loaded to one GPU.

# Troubleshooting

Full NCCL debug investigation (Issues 0–5, step-by-step trial log) is archived in [`miscs/NCCL_debug_archive.md`](../miscs/NCCL_debug_archive.md).

---

### NCCL Dual-Instance Conflict in jaxlib 0.9.0 ✅ RESOLVED

**Root cause:** jaxlib 0.9.0 compiles in its own NCCL (`external/nccl_archive`) **and** also dlopen's the pip `nvidia-nccl-cu12` library. The two instances create incompatible communicator handles → `ncclAllReduce: invalid argument` / `corrupted comm object detected`.

**Fix — force a single NCCL instance via `LD_PRELOAD`:**

```bash
LD_PRELOAD=/home/ray/miniforge3/envs/alphagenome/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2 \
    python scripts/tf_ft.py --config configs/config_bigwig.yaml
```

`LD_PRELOAD` overrides jaxlib's internal NCCL for all `dlopen` calls, leaving only the pip library active.

**Environment (Feb 2026):** Driver `580.126.09` (CUDA 13.0), `nvidia-nccl-cu12==2.27.5`, jaxlib 0.9.0.

**Confirmed working:** 8-GPU training on RTX 6000 Ada — `tf_ft.py` runs full epochs, loss decreases correctly.

---

### PyTorch→JAX Data Pipeline: Signal Corruption ⚠️ UNRESOLVED

`scripts/pytorch_ft.py` exhibits systematic signal corruption (~4,700× amplification) after ~1,700 batches during multi-GPU training. Root cause is in the PyTorch→JAX bridge with `NamedSharding`, independent of signal file format (BedGraph or BigWig).

**Workaround:** Use `scripts/tf_ft.py` (TensorFlow pipeline) for all multi-GPU training.

---

### ZFS Pool `/celestial` Not Mounted After Reboot

The `checkpoints/` symlink points to `/celestial/ray/model_checkpoints/AlphaGenome`. If the pool is unmounted, checkpoint writes silently fail with `FileNotFoundError`.

```bash
ls /celestial/ray/model_checkpoints/AlphaGenome/   # empty → not mounted
sudo zpool import celestial
```

---

### Rendezvous Warnings in Training Output

False-positive `"This thread has been waiting for acquire clique for rank 0... for 10 seconds"` messages appear every ~28 s. These are harmless — each is followed by `"Thread is unstuck!"` within seconds. `tf_ft.py` redirects stderr to `<checkpoint_dir>/stderr.log` to keep the terminal clean.

---

### Official AlphaGenome Finetuning API: Mouse Not Supported

`alphagenome_research.finetuning` raises `NotImplementedError` for non-human organisms. Use `scripts/tf_ft.py` with the custom TensorFlow + BigWig pipeline for mouse data — it sets `organism_id=1` (`MUS_MUSCULUS`) directly.
