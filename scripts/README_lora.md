# LoRA Fine-Tuning for AlphaGenome

This directory contains code for parameter-efficient fine-tuning of AlphaGenome using LoRA (Low-Rank Adaptation) on custom CUT&Tag data.

## Overview

**LoRA** allows you to fine-tune large models by training only small low-rank matrices, dramatically reducing:
- Trainable parameters: 450M → ~428K (99.9% reduction with rank=8)
- Memory requirements
- Training time

## Files

- `lora_utils.py` - LoRA parameter initialization and utilities
- `lora_model.py` - LoRA-wrapped AlphaGenome model
- `data_loader.py` - CUT&Tag data loading
- `train_lora.py` - Main training script
- `inference_lora.py` - Inference with fine-tuned model

## Quick Start

### Step 1: Prepare Your CUT&Tag Data

Process your FASTQ files to generate signal tracks:

```bash
# 1. Align reads
bowtie2 -x /path/to/genome_index \
    -1 your_sample_R1.fastq.gz \
    -2 your_sample_R2.fastq.gz \
    | samtools sort -o aligned.bam

# 2. Index BAM
samtools index aligned.bam

# 3. Generate signal track (BigWig)
bamCoverage -b aligned.bam \
    -o signal.bw \
    --binSize 10 \
    --normalizeUsing CPM

# 4. Convert to BedGraph (optional)
bigWigToBedGraph signal.bw signal.bedGraph
```

### Step 2: Train LoRA Model

```bash
python train_lora.py \
    --checkpoint /home/ray/AlphaGenome/models/pre_trained/alphagenome-fold-0 \
    --fasta /home/ray/RESOURCES/reference/hg38/GRCh38.p13.genome.fa \
    --signal ./signal.bedGraph \
    --output ./lora_checkpoints \
    --epochs 10 \
    --batch-size 4 \
    --lr 1e-4 \
    --lora-rank 8
```

### Step 3: Use Fine-Tuned Model for Inference

```python
from alphagenome_research.model import dna_model
from lora_model import LoRAAlphaGenomeModel
from lora_utils import load_lora_checkpoint, get_attention_qkv_layers

# Load base model
base_model = dna_model.create(
    checkpoint_path="/path/to/alphagenome-fold-0"
)

# Load LoRA fine-tuned weights
target_layers = get_attention_qkv_layers()
lora_model = LoRAAlphaGenomeModel(
    base_model=base_model,
    target_layers=target_layers
)
lora_model.load_lora("./lora_checkpoints/lora_best.pkl")

# Make predictions
predictions = lora_model.forward(dna_sequence, organism_index)
```

## Training Parameters

### LoRA Hyperparameters

- **`--lora-rank`** (default: 8)
  - Higher rank = more expressiveness but more parameters
  - Recommended: 4, 8, 16, or 32
  - rank=8 gives ~428K trainable params (99.9% reduction)

- **`--lora-alpha`** (default: 16.0)
  - Scaling factor: final_delta = (B @ A) * (alpha / rank)
  - Typically set to 2 * rank

### Training Hyperparameters

- **`--lr`** (default: 1e-4)
  - Learning rate for Adam optimizer
  - Start conservative; increase if training is slow

- **`--batch-size`** (default: 4)
  - Depends on GPU memory
  - 4-8 works well on Ada 6000 (48GB VRAM)

- **`--epochs`** (default: 10)
  - Start with 5-10 epochs
  - Monitor loss curve for convergence

### Data Parameters

- **`--window-size`** (default: 2048)
  - Sequence length for training
  - AlphaGenome supports up to 2^20 (1M bp)

- **`--windows-bed`** (optional)
  - BED file specifying training regions
  - If not provided, samples from whole genome

## Advanced Usage

### Custom Target Layers

By default, LoRA is applied to all Q/K/V attention layers. To customize:

```python
# Apply LoRA to specific layers
custom_layers = [
    "alphagenome/transformer_tower/mha_block/q_layer/w",
    "alphagenome/transformer_tower/mha_block/k_layer/w",
    # Add more layers...
]

lora_model = LoRAAlphaGenomeModel(
    base_model=base_model,
    target_layers=custom_layers,
    lora_rank=8
)
```

### Include MLP Layers

For more capacity, also fine-tune MLP layers:

```python
from lora_utils import get_attention_qkv_layers

# Get Q/K/V layers
qkv_layers = get_attention_qkv_layers()

# Add MLP layers
mlp_layers = [
    f"alphagenome/transformer_tower/mlp_block_{i}/linear/w"
    for i in range(9)
]

all_layers = qkv_layers + mlp_layers
```

### Custom Loss Functions

Modify `train_lora.py` to use different loss functions:

```python
def custom_loss(predictions, targets):
    # Poisson loss for count data
    pred = predictions['atac']['predictions_1bp']
    target = targets['signal']
    
    loss = jnp.mean(pred - target * jnp.log(pred + 1e-8))
    return loss
```

## Expected Results

### Parameter Efficiency

| LoRA Rank | Trainable Params | % of Original |
|-----------|------------------|---------------|
| 4         | 214K            | 0.05%         |
| 8         | 428K            | 0.09%         |
| 16        | 857K            | 0.19%         |
| 32        | 1.7M            | 0.38%         |

### Training Time

On 8x Ada 6000 GPUs:
- ~1-2 seconds per batch (batch_size=4, window=2048)
- ~1000 batches per epoch (depends on data)
- ~30 minutes per epoch

### Memory Usage

- Base model: ~2 GB
- LoRA params: <10 MB
- Training: ~10-15 GB per GPU (with batch_size=4)

## Troubleshooting

### Out of Memory

1. Reduce `--batch-size`
2. Reduce `--window-size`
3. Use fewer target layers

### Loss Not Decreasing

1. Check data quality (signal distribution)
2. Increase `--lr` (try 3e-4)
3. Increase `--lora-rank`
4. Verify loss function matches your data

### Slow Training

1. Increase `--batch-size` if memory allows
2. Reduce number of target layers
3. Use multiple GPUs (requires code modification)

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [AlphaGenome](https://github.com/google-deepmind/alphagenome_research)
- [CUT&Tag Protocol](https://www.nature.com/articles/s41596-020-0373-x)

## Citation

If you use this code, please cite:

```bibtex
@article{alphagenome,
  title={Advancing regulatory variant effect prediction with {AlphaGenome}},
  author={Avsec, {\v Z}iga and Latysheva, Natasha and Cheng, Jun and ...},
  journal={Nature},
  year={2026}
}
```
