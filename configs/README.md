# Configuration Files

This directory contains YAML configuration files for fine-tuning AlphaGenome on CUT&Tag data.

## Quick Start

1. **Copy the template**:
   ```bash
   cp config.yaml my_experiment.yaml
   ```

2. **Edit your config**:
   Modify `my_experiment.yaml` with your specific paths and hyperparameters.

3. **Run training**:
   ```bash
   python scripts/train.py --config configs/my_experiment.yaml
   ```

## Configuration Structure

The config file contains the following sections:

### Strategy
- `strategy`: Fine-tuning approach (`head_replace` or `lora`)
- `precision_policy`: Numerical precision (default: full float32)

### Paths (Required)
- `checkpoint`: Pre-trained AlphaGenome checkpoint directory
- `fasta`: Reference genome FASTA file
- `signal`: CUT&Tag bedGraph file (base-pair resolution, 100M normalized)
- `output`: Output directory for checkpoints and logs
- `windows_bed`: Optional BED file with specific genomic windows

### Training Hyperparameters
- `epochs`: Number of training epochs
- `batch_size`: Total batch size across all GPUs
- `learning_rate`: Peak learning rate (Adam optimizer)

### LoRA Configuration
- `lora_rank`: LoRA rank (only used when `strategy: lora`)
- `lora_alpha`: LoRA scaling factor

### Data & Augmentation
- `window_size`: Sequence window size in bp (must be divisible by 128)
- `organism_id`: 0=human, 1=mouse
- `augment`: Enable/disable data augmentation
- `max_shift`: Random shift range for augmentation (±bp)

### System
- `num_workers`: Number of DataLoader workers

## CLI Override

Command-line arguments always override config file values:

```bash
# Use config but override batch size and learning rate
python scripts/train.py --config configs/my_experiment.yaml \
    --batch-size 16 \
    --lr 1e-4
```

## Config Backup

Each training run automatically:
1. Backs up the config file to the output directory
2. Adds a timestamp to the backup filename (e.g., `config_20260223_083657.yaml`)
3. This ensures reproducibility and tracks exact settings for each run

Example backup location:
```
checkpoints/fine_tuned_2026-02-23/
├── config_20260223_083657.yaml  ← Backup with timestamp
├── train_20260223_083657.log
├── metrics.csv
└── epoch_001/
```

## Example Configs

### Head Replacement (Full Fine-tuning)

```yaml
strategy: head_replace
checkpoint: /path/to/alphagenome-all-folds
fasta: /path/to/mm10.fa
signal: /path/to/D1MSN_H3K27ac_averaged_normalized.bedgraph
output: ./checkpoints/fine_tuned_head_replace
learning_rate: 5e-4
batch_size: 8
epochs: 10
window_size: 131072
organism_id: 1
augment: true
```

### LoRA (Parameter-efficient)

```yaml
strategy: lora
checkpoint: /path/to/alphagenome-all-folds
fasta: /path/to/mm10.fa
signal: /path/to/D1MSN_H3K27ac_averaged_normalized.bedgraph
output: ./checkpoints/fine_tuned_lora
learning_rate: 1e-5  # Lower LR for LoRA
lora_rank: 8
lora_alpha: 16.0
batch_size: 8
epochs: 20
window_size: 131072
organism_id: 1
```

## Tips

1. **Start with the template**: Always copy `config.yaml` as a starting point
2. **Use descriptive names**: Name configs after your experiments (e.g., `d1msn_h3k27ac_lora_rank8.yaml`)
3. **Version control**: Commit your config files to track experiment history
4. **Document changes**: Add comments in YAML files to note modifications
5. **Test first**: Run a short test with `epochs: 1` before full training

## Validation

The script validates:
- Required fields are present (checkpoint, fasta, signal)
- Window size is divisible by 128
- Paths exist (where applicable)

Missing or invalid configurations will produce clear error messages.
