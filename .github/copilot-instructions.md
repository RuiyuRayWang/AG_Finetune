# AG_Finetune — Copilot Instructions

Fine-tuning AlphaGenome (450M param JAX/Haiku genomics model) on CUT&Tag H3K27ac data (two strategies: head-replacement and LoRA), on 8-GPU Linux server.

**⚠️ CRITICAL: Always reference the [AlphaGenome official documentation](https://www.alphagenomedocs.com/index.html) when working with the model API. Do NOT make assumptions about checkpoint loading, model initialization, or prediction interfaces - verify against the docs first.**

**Key reference patterns:**
- Checkpoint loading: `params, state = ocp.StandardCheckpointer().restore(checkpoint_path)` (see `alphagenome_research/evals/track_prediction.py:74`)
- Model creation: Use `hk.transform_with_state` with `model_lib.AlphaGenome(metadata)` constructor (see `alphagenome_research/evals/track_prediction.py:79-83`)
- Metadata loading: `metadata_lib.load(organism)` for each organism (see `alphagenome_research/evals/track_prediction.py:76-78`)

---

## Architecture Overview

```
CutAndTagDataset (PyTorch) → DataLoader → collate_to_jax → { HeadReplaceAlphaGenomeModel  (default, --strategy head_replace)
          ↕                                                  { LoRAAlphaGenomeModel          (alternative, --strategy lora)
   dataset/dataset.py                                                   ↕
   (bedgraph + FASTA)                                      models/model.py → cut_and_tag_loss_baskerville → orbax checkpoint
```

**Key components:**
- `models/model.py` — `HeadReplaceAlphaGenomeModel` and `LoRAAlphaGenomeModel`. Both share the same `.train_step(seq, org, targets, loss_fn)` API and orbax `.save()`/`.load()` interface. Both use fp32 and `optax.clip_by_global_norm(0.5)` + adam.
- `utils/lora_utils.py` — `initialize_lora_params`, `apply_lora_to_params`: LoRA uses `ΔW = B @ A * (alpha/rank)`. Haiku param keys are flat strings like `'alphagenome/transformer_tower/mha_block_0/q_layer'`.
- `dataset/dataset.py` — `CutAndTagDataset`: loads bedgraph signal + FASTA; generates 131,072 bp windows; applies shift (±1024 bp) and reverse-complement augmentation.
- `utils/borzoi_loss.py` — `cut_and_tag_loss_baskerville`: Borzoi-style Poisson + multinomial NLL. Applies `baskerville_target_transform` internally — **pass raw counts, do not pre-transform**. Use `preds.atac.scaled_predictions_1bp`.
- `scripts/pytorch_ft.py` — Main training entry (both strategies).

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

> **TODO (remaining debugging):**
> - Remove sentinel grad-norm flags (−3/−4) from `lora_model.py` once `head_replace` training is confirmed stable.
> - Update `manul_debug_grad_norm.py` to use the new strategy-aware model API.

---

## Dataset & Signal Format

- Signal: bedgraph, base-pair resolution, normalized to **100M total insertions per track**.
- Raw counts are passed as `float32` targets; the loss function transforms them internally — **do not pre-transform targets**.
- `organism_id`: `0` = human, `1` = mouse.
- `window_size` must be divisible by 128. AlphaGenome was pretrained on 1,048,576 bp (2^20^); fine-tuning uses 131,072 bp (2^17^).
- CUT&Tag is conceptually identical to ATAC-seq: use the `atac` prediction head at 1 bp resolution.

---

## Training Commands

```bash
# Head-replacement (default, recommended):
python scripts/pytorch_ft.py --strategy head_replace \\
    --checkpoint /home/ray/AG_Finetune/checkpoints/pre_trained/alphagenome-all-folds \\
    --fasta /home/ray/AG_Finetune/data/resources/mm10.fa \\
    --signal data/CUTTag/processed_bedgraph/D1MSN_H3K27ac_averaged_normalized.bedgraph \\
    --output ./checkpoints/fine_tuned \\
    --batch-size 8 --lr 5e-4 --window-size 131072

# LoRA (parameter-efficient alternative):
python scripts/pytorch_ft.py --strategy lora \\
    --checkpoint /home/ray/AG_Finetune/checkpoints/pre_trained/alphagenome-all-folds \\
    --fasta /home/ray/AG_Finetune/data/resources/mm10.fa \\
    --signal data/CUTTag/processed_bedgraph/D1MSN_H3K27ac_averaged_normalized.bedgraph \\
    --lora-rank 8

# Using config file (recommended):
python scripts/pytorch_ft.py --config configs/config_bigwig.yaml
python scripts/pytorch_ft.py --config configs/config_bedgraph.yaml
```

~~# Debug entry point:
python scripts/manul_debug_grad_norm.py~~

~~# NaN diagnostic (bf16 vs fp32 ablation):
python scripts/analyze_batch17.py --max-batches 120
python scripts/analyze_batch17.py --max-batches 120 \\
    --precision-policy "params=float32,compute=float32,output=float32"~~

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
| `debug_with_eval_shape.py` | **[PRIMARY]** Validate training logic with jax.eval_shape (DeepMind recommendation) - identifies shape mismatches before NCCL calls |
| `analyze_batch17.py` | Find first NaN-grad batch; inspect all prediction heads + grad stats |
| `trace_parameter_drift.py` | Monitor LoRA param statistics per training step |
| `debug_gradnorm_detailed.py` | Per-batch grad norm with NaN/Inf sentinel codes (−1=grad NaN, −2=grad Inf, −3=param NaN, −4=param Inf) |
| `investigate_nan_gradients.py` | Scan batches for NaN gradients with/without shuffle |
| `debug_training_pipeline.py` | Validate data pipeline stages (FASTA → bedgraph → collation → device sharding) |
| `inspect_and_export.py` | Export AlphaGenome layer names suitable for LoRA targeting |

---

# Supplementary Methods

This document contains detailed information for finetuning the [AlphaGenome](https://github.com/google-deepmind/alphagenome_research) model.

The blockquoted (>) contents are excerptions from the paper relevant to the task. Fenced codes (```) are pseudocodes from the paper. Between the excerptions I give my remarks in italicized texts.

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
*Previously, pyBigWig failed when writing 1bp-resolution averaged data because it contains millions of 1bp intervals, which is likely exceeding pyBigWig's internal limits. Converted bigwig to bedgraph to proceed.*
*In a recently released finetuning tutorial notebook from DeepMind, pyBigWig was used. Consider switch back to bigwig format for consistency.*

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

## Troubleshooting

### Multi-GPU Training Issues (February 2026)

**⚠️ IMPORTANT UPDATE:** After consultation with the DeepMind AlphaGenome team, we confirmed that **multi-GPU training IS fully supported**. DeepMind has tested finetuning on 8 H100 GPUs successfully. The NCCL errors we encountered are due to **implementation bugs in our wrapper code**, not fundamental model limitations.

This section documents our debugging journey and the issues we encountered. The root cause is likely incorrect mesh setup, sharding specifications, or data pipeline bugs in our custom implementation.

---

#### Issue 1: Data Corruption in PyTorch→JAX Pipeline

**Problem:** Training exhibited systematic signal corruption with values far exceeding expected ranges (7.234e+04 or 3.269e+07 vs expected ~7,000 from normalized data).

**Investigation Timeline:**
1. **Preprocessed data verified clean** — direct inspection shows max values ~7,000, properly normalized to 100M total insertions
2. **PyTorch DataLoader tested in isolation** — `debug_signal_values.py` found NO corrupt batches in first 300 batches when loading data without JAX training
3. **Corruption only occurs during JAX training** — appears at batch ~1700-1825, not immediately
4. **Safe windows strategy ineffective** — pre-validated genomic regions (excluding hypersensitive peaks) did NOT prevent corruption
5. **Signal format independent (Feb 25, 2026)** — corruption occurs with BOTH BedGraph AND BigWig inputs:
   - BedGraph run: max=3.269e+07 at batch ~1800 (4,700x amplification)
   - BigWig run: max=7.234e+04 at batch ~1723 (10x amplification)
   - Identical corrupt values across consecutive batches (100+ batches affected)
6. **Corruption pattern suggests sharding bug:**
   - Only occurs with 8-GPU data parallelism via `NamedSharding(mesh, P('batch', ...))`
   - Consistent timing (after 1500-1800 batches, ~40% through epoch)
   - Affects all subsequent batches until caught by 50,000/bp threshold guard

**Root Cause Hypothesis:** The PyTorch→JAX bridge (`jax.device_put` with sharding spec) has race conditions or incorrect array views when transferring large arrays across 8 devices. Memory corruption likely occurs during `collate_fn` → `jax.device_put` → device sharding sequence. The issue is independent of signal file format (BedGraph vs BigWig) but dependent on multi-GPU sharding.

**Status:** Unresolved. PyTorch-based pipeline (`scripts/pytorch_ft.py`) NOT recommended for multi-GPU training due to data corruption.

---

#### Issue 2: NCCL AllReduce Errors in Validation Code

**Problem:** TensorFlow+pyBigWig pipeline (`scripts/finetune_bak.py`) hit `NCCL operation ncclAllReduce... failed: invalid argument` when computing global statistics for validation.

**Trigger:** Calling `jnp.sum()`, `jnp.max()`, `jnp.mean()` on sharded arrays in Python (non-jitted) code:
```python
target_sum = float(jnp.sum(targets['signal']))  # NCCL error!
target_max = float(jnp.max(targets['signal']))  # NCCL error!
```

**Solution:** Use `jax.device_get()` to move data to host before computing statistics:
```python
target_sum = float(jnp.sum(jax.device_get(targets['signal'])))  # Works
target_max = float(jnp.max(jax.device_get(targets['signal'])))  # Works
```

**Why:** Global reductions on sharded arrays trigger NCCL AllReduce across devices. `device_get()` transfers data to CPU first, bypassing NCCL communication. Element-wise operations and loss functions don't need this workaround.

**Status:** Partially resolved. Validation code fixed, but training step still fails.

---

#### Issue 3: NCCL AllReduce Errors Inside Jitted Training Step (TensorFlow Pipeline)

**Problem:** Even after fixing validation code, NCCL errors persist inside the jitted `train_step()` function at the first batch.

**Attempted Fixes:**
1. ✅ Fixed validation statistics (jnp.sum/max) using `device_get()` → still fails
2. ✅ Removed all NaN/Inf checking tree reductions from jitted function → still fails  
3. ✅ Replicated optimizer state with `jax.device_put(opt_state, param_sharding)` → still fails
4. ✅ Removed gradient norm computation (tree_reduce) entirely → **still fails**
5. ✅ Simplified training loop to bare minimum (forward + optimizer.update) → **still fails**

**Minimal Failing Code:**
```python
def _train_step(dna_sequence, organism_index, targets, params, state, opt_state, loss_fn):
    def compute_loss(p):
        predictions = apply_fn(p, state, dna_sequence, organism_index)
        return loss_fn(predictions, targets['signal']), predictions
    
    (loss_value, predictions), grads = jax.value_and_grad(compute_loss, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)  # ← NCCL fails here
    new_params = optax.apply_updates(params, updates)
    return loss_value, predictions, new_params, new_opt_state, jnp.float32(0.0)
```

**Error Location:** The NCCL error occurs during `optimizer.update()` which internally performs AllReduce for gradient averaging across devices. Should investigate if optimizer state sharding or gradient sharding is misconfigured.

**Hypothesis:** The issue is deeper than our code - likely:
- NCCL library version incompatibility with JAX
- GPU memory fragmentation from large model (450M params)
- Incorrect sharding specs causing NCCL to receive malformed buffers
- Underlying JAX/XLA bug with replicated parameters + sharded inputs

**Status:** Unresolved. Custom multi-GPU training approach abandoned.

**Update (Feb 24, 2026 - evening):** **POTENTIAL FIX IDENTIFIED**
- Root cause: `jax.jit()` was not specifying `in_shardings` and `out_shardings` for the training step
- Without explicit sharding specs, JAX tries to infer and reshard data, triggering NCCL errors
- **Solution:** Add explicit sharding specifications to `jax.jit()` call:
  ```python
  self._train_step_jit = jax.jit(
      self._train_step,
      static_argnames=['loss_fn'],
      in_shardings=(
          batch_sharding,      # inputs sharded on batch dimension
          organism_sharding,
          {'signal': signal_sharding},
          replicated,          # params/state/opt_state replicated
          replicated,
          replicated,
      ),
      out_shardings=(
          replicated,          # outputs all replicated
          None,                # predictions (let JAX infer)
          replicated,
          replicated,
          replicated,
      )
  )
  ```
- This tells JAX exactly what sharding to expect, avoiding implicit resharding
- **Status:** Fix implemented in `models/model.py`, testing required

---

#### Issue 4: Official AlphaGenome Finetuning API Doesn't Support Mouse

**Problem:** The official `alphagenome_research.finetuning` module only supports human data:
```python
# From alphagenome_research/finetuning/dataset.py:171
if organism != dna_model.Organism.HOMO_SAPIENS:
    raise NotImplementedError('Only HOMO_SAPIENS is currently supported.')
```

**Impact:** Cannot use DeepMind's official finetuning notebook/API for mouse CUT&Tag data. The `scripts/finetune.py` script refactored from the colab notebook immediately fails at dataset initialization.

**Workarounds:**
1. Modify `alphagenome_research` source code to add mouse support (non-trivial - requires implementing mouse genome handling, intervals, etc.)
2. Use single-GPU training with custom pipeline (slow but may avoid NCCL issues)
3. Convert mouse data to appear as "human" (hack, likely breaks internal assumptions)

**Status:** Blocked. Need to either extend official API or debug multi-GPU NCCL issues.

---

#### Issue 5: Multi-GPU Training NCCL Errors (Implementation Bug, Not Model Limitation) ⚠️ **DEBUGGING IN PROGRESS**

**Problem:** Our multi-GPU training attempts fail with NCCL "corrupted comm object" errors during gradient computation, despite NCCL working correctly for basic operations.

**Investigation Timeline:**
1. ✅ Basic NCCL tests pass (`test_nccl_simple.py`)
2. ✅ AlphaGenome forward passes work with multi-GPU (Tests 1-3)
3. ❌ AlphaGenome gradient computation fails with NCCL errors (Test 4)
4. ✅ PyTorch training never had NCCL errors (only data corruption)

**DeepMind Confirmation (February 2026):**
> "Yes, the finetuning script supports multiple devices on the same host. You can see this in 'Set Up Device Mesh for Data Parallelism' cell. We've also tested this internally on a machine with 8 H100 GPUs."
> 
> "NCCL errors can be tricky to diagnose. A good first step is using jax.eval_shape to validate your training logic without actually running the GPU kernels."
>
> — Vincent (DeepMind AlphaGenome Team)

**Root Cause:** This is an **implementation bug in our code**, NOT a fundamental AlphaGenome limitation. DeepMind has confirmed multi-GPU training works correctly on their 8-GPU setup.

**Likely Issues:**
- Incorrect mesh setup or sharding specifications
- Bug in our custom model wrapper classes (`HeadReplaceAlphaGenomeModel`, `LoRAAlphaGenomeModel`)
- Data pipeline issue specific to PyTorch→JAX bridge
- Missing initialization step from official notebook

**Debugging Strategy (per DeepMind suggestion):**
1. Use `jax.eval_shape` to validate training logic without running GPU kernels
2. Compare our mesh setup line-by-line with official "Set Up Device Mesh for Data Parallelism" cell
3. Test with official `alphagenome_research.finetuning` module first (human data) to verify multi-GPU works
4. Then adapt working multi-GPU code for mouse data

**Temporary Workaround:** Single-GPU training with gradient accumulation (`scripts/train_singlegpu.py`) while debugging multi-GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_singlegpu.py \\
    --config configs/config_bigwig.yaml \\
    --accumulation-steps 8 \\
    --epochs 10
```

**Status:** ⚠️ Multi-GPU is POSSIBLE (DeepMind confirmed), but our implementation has bugs. Need to debug following DeepMind's suggestions.

---

### Summary of Lessons Learned

1. **PyTorch→JAX data loading is fragile** with multi-GPU sharding:
   - Mysterious corruption appears after ~1700 batches (not immediately)
   - **Signal format independent** - occurs with BOTH BedGraph AND BigWig inputs
   - BedGraph: max=3.269e+07 (4,700x amplification)
   - BigWig: max=7.234e+04 (10x amplification)
   - Always affects consecutive batches (100+) once started
   - Related to PyTorch→JAX bridge with 8-GPU sharding, not data quality
2. **NCCL requires careful handling** in JAX:
   - Use `device_get()` for validation statistics outside jitted functions
   - Tree reductions inside jitted functions can trigger NCCL errors
   - Even minimal training steps fail with 8-GPU sharding in our setup
3. **JAX JIT requires explicit sharding specifications for multi-GPU** ⭐ **KEY FINDING**:
   - Without `in_shardings` and `out_shardings` in `jax.jit()`, JAX tries to infer shardings
   - This causes implicit resharding operations that trigger NCCL AllReduce errors
   - **Always specify sharding for inputs and outputs when using data parallelism**
4. **AlphaGenome's official finetuning is human-only** - mouse organism not supported in `alphagenome_research.finetuning` module
5. **Safe windows don't prevent corruption** - the issue is in PyTorch pipeline, not genomic data quality
6. **Gradient computation simplifications don't help** - removing gradient norm calculation, NaN checks, etc. still fails at optimizer.update() (until sharding specs are added)
7. **✅ jax.eval_shape validation PASSED** ⭐ **MAJOR BREAKTHROUGH** (Feb 25, 2026):
   - Created `scripts/debug_with_eval_shape.py` to validate training logic without GPU execution
   - Forward pass, gradient computation, and optimizer updates all work correctly
   - All shapes are compatible through the pipeline (589 param arrays, 81 BatchNorm layers)
   - **Confirmed: Issue is NOT shape mismatch** - must be sharding/mesh configuration
   - Base model outputs 256 ATAC tracks; fine-tuning replaces with 1 track via head replacement
   - Next step: Compare mesh setup with official DeepMind notebook line-by-line
8. **✅ CORRECTED: AlphaGenome DOES support multi-GPU training** (our implementation had bugs):
   - **DeepMind confirmation (Feb 2026)**: "The finetuning script supports multiple devices on the same host. We've tested this internally on a machine with 8 H100 GPUs."
   - Official notebook has "Set Up Device Mesh for Data Parallelism" cell showing proper multi-GPU setup
   - Our NCCL errors are due to **implementation bugs in our code**, not model limitations
   - Need to debug using `jax.eval_shape` and compare with official mesh setup
   - Forward passes work fine with multi-GPU sharding (Tests 1-3 passed)
   - Gradient computation fails in our implementation (Test 4) - likely mesh/sharding bug
   - **PyTorch training never had NCCL errors** - suggests issue is in our JAX wrapper code

### Critical Training Requirements

**✅ UPDATE (February 2026): Multi-GPU training IS supported by AlphaGenome**

DeepMind has confirmed that AlphaGenome supports multi-GPU finetuning and they've tested it on 8 H100 GPUs. Our NCCL errors are due to **implementation bugs in our code**, not model limitations.

**Multi-GPU Training is MANDATORY:**
- Training on single GPU is **8x slower** (~200 hours vs ~25 hours)
- Single-GPU wastes 7 out of 8 available GPUs
- **We MUST fix the multi-GPU implementation**, not work around it

**Debugging Strategy (per DeepMind):**
1. **Use `jax.eval_shape`** to validate training logic without GPU execution (primary recommendation)
2. **Compare our mesh setup** with official "Set Up Device Mesh for Data Parallelism" cell line-by-line
3. **Test official finetuning module** with human data first to verify multi-GPU works on our hardware
4. **Fix bugs** in `HeadReplaceAlphaGenomeModel`/`LoRAAlphaGenomeModel` wrapper classes

**Single-GPU Script Status:**
- `scripts/train_singlegpu.py` exists **for testing only** (data pipeline validation, quick loss checks)
- **NOT for production training** - performance is unacceptable
- Use only to verify data loads correctly while debugging multi-GPU
   ```python
   # Accumulate gradients over N micro-batches to simulate batch size 8
   accumulated_grads = None
   for micro_batch in range(8):
       grads = compute_grads(params, data[micro_batch])
       if accumulated_grads is None:
           accumulated_grads = grads
       else:
           accumulated_grads = jax.tree.map(lambda a, g: a + g, accumulated_grads, grads)
   
   # Average and apply
   accumulated_grads = jax.tree.map(lambda g: g / 8, accumulated_grads)
   updates, new_opt_state = optimizer.update(accumulated_grads, opt_state, params)
   ```
3. **Reduce window size** if memory limited (131072 → 65536 bp)
4. **Use LoRA** for parameter-efficient training (<1% trainable params)

**Performance Impact (Single-GPU - NOT VIABLE):**
- 8x slower than multi-GPU approach
- ~200 hours for full training vs ~25 hours (unacceptable)
- Wastes 7 out of 8 available GPUs

**Conclusion: Multi-GPU implementation MUST be fixed for production use.**

### Recommended Next Steps

**⚠️ CURRENT STATUS: Multi-GPU training blocked by NCCL errors (Feb 25, 2026)**
- **PyTorch pipeline:** Data corruption at batch ~1700
- **TensorFlow pipeline:** NCCL AllReduce errors during first batch compilation

**Primary Debugging Approach (DeepMind recommendation):**

1. **✅ Use `jax.eval_shape` for validation** (DeepMind's primary suggestion):
   ```bash
   python scripts/debug_with_eval_shape.py --config configs/config_bigwig.yaml
   ```
   - **STATUS: PASSED ✓** (Feb 25, 2026)
   - Validates shapes without GPU execution
   - Confirms forward pass, gradient computation, and optimizer updates all work
   - **All shapes compatible** - issue is NOT shape mismatch
   - ATAC predictions: (8, 131072, 256) from base model → will be (8, 131072, 1) after head replacement
   - 81 BatchNorm layers in state, 589 parameter arrays
   - **Conclusion:** Training logic is sound; NCCL errors caused by sharding/mesh configuration

2. **Line-by-line comparison with official notebook** (NEXT STEP):
   - Review "Set Up Device Mesh for Data Parallelism" cell as reference implementation
   - Compare mesh creation in `models/model.py` with official notebook
   - Check for missing initialization steps or incorrect sharding specs
   - **Action needed:** Extract mesh setup code from official notebook and compare

3. **Test official module first** (validate multi-GPU works on our hardware):
   - Run official `alphagenome_research.finetuning` with human data
   - Confirms 8-GPU training works with our drivers/hardware
   - Provides working baseline for comparison

4. **Systematic debugging of wrapper classes** (after confirming official code works):
   - Audit `HeadReplaceAlphaGenomeModel._train_step()` in `models/model.py`
   - Audit `LoRAAlphaGenomeModel._train_step()` in `models/model.py`
   - Verify all `jax.jit` calls have correct `in_shardings` and `out_shardings`
   - Validate PyTorch→JAX data pipeline doesn't corrupt sharded arrays
   - Test with `scripts/test_alphagenome_model.py` after each fix

**Testing Only (Not for Production):**
- `scripts/train_singlegpu.py` can be used to quickly test data pipeline
- Use only for: verifying data loads, checking loss computation works
- **DO NOT use for full training** - 8x performance penalty is unacceptable

**Resources:**
- Official finetuning notebook: \"Set Up Device Mesh for Data Parallelism\" cell
- DeepMind contact: Vincent (AlphaGenome team) - responsive to questions
- Use `jax.eval_shape` for debugging training logic without GPU execution
 