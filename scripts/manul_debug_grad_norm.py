import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from functools import partial

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__name__))))

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from torch.utils.data import DataLoader
from alphagenome_research.model import dna_model
from models.lora_model import LoRAAlphaGenomeModel
from dataset.dataset import CutAndTagDataset
from utils.lora_utils import get_attention_qkv_layers
from utils.borzoi_loss_v2 import cut_and_tag_loss_baskerville
import json

checkpoint_path = os.path.abspath('/home/ray/AG_Finetune/checkpoints/pre_trained/alphagenome-all-folds')
fasta_path = os.path.abspath('/home/ray/AG_Finetune/data/resources/mm10.fa')
signal_path = os.path.abspath('/home/ray/AG_Finetune/data/CUTTag/processed_bedgraph/D1MSN_H3K27ac_averaged_normalized.bedgraph')
output_dir = os.path.abspath('/home/ray/AG_Finetune/checkpoints/debug_gradnorm_nan')
epochs: int = 10
batch_size: int = 8
learning_rate: float = 1e-5
lora_rank: int = 8
lora_alpha: float = 16.0
window_size: int = 131072
organism_id: int = 1
genome_windows: str = None
augment: bool = True
max_shift: int = 1024
num_workers: int = 0

def collate_to_jax(batch: List[Tuple[np.ndarray, int, Dict]]) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    """
    Custom collate function to convert batch of numpy arrays to JAX arrays.
    
    Following AlphaGenome's mixed precision policy:
    - Input sequences: float32 (standard input precision)
    - Targets: float32 (raw counts, will be transformed in loss function)
    - Model outputs: bfloat16 (from mixed precision policy)
    
    Args:
        batch: List of (sequence, organism_id, targets) tuples from Dataset
    
    Returns:
        (sequences, organism_ids, targets) as JAX arrays:
            sequences: [batch, window_size, 4] - float32
            organism_ids: [batch] - int32
            targets: dict with 'signal' [batch, window_size] - float32 (raw counts)
    """
    sequences = [item[0] for item in batch]
    organism_ids = [item[1] for item in batch]
    signals = [item[2]['signal'] for item in batch]
    
    return (
        jnp.array(sequences, dtype=jnp.float32),
        jnp.array(organism_ids, dtype=jnp.int32),
        {'signal': jnp.array(signals, dtype=jnp.float32)}
    )

def pearson_correlation(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    pred_mean = jnp.mean(predictions)
    target_mean = jnp.mean(targets)
    
    numerator = jnp.sum((predictions - pred_mean) * (targets - target_mean))
    denominator = jnp.sqrt(
        jnp.sum((predictions - pred_mean) ** 2) * 
        jnp.sum((targets - target_mean) ** 2)
    )
    
    return numerator / (denominator + 1e-8)

# Setup device mesh for multi-GPU training (MUST BE FIRST)
num_devices = jax.device_count()

if num_devices > 1:
    print("="*60)
    print(f"MULTI-GPU DETECTED: {num_devices} devices")
    print("="*60)
    print(f"Devices: {jax.devices()}")
    print(f"Using JAX data parallelism with parameter replication")
    print()
    
    # Create device mesh for data parallelism
    mesh = Mesh(jax.devices(), ('batch',))
    # Shard batch dimension across devices, replicate other dimensions
    batch_sharding = NamedSharding(mesh, P('batch', None, None))  # for sequences: [batch, seq_len, 4]
    organism_sharding = NamedSharding(mesh, P('batch'))  # for organism_ids: [batch]
    signal_sharding = NamedSharding(mesh, P('batch', None))  # for signals: [batch, seq_len]
else:
    print("Single GPU mode")
    mesh = None
    batch_sharding = None
    organism_sharding = None
    signal_sharding = None

# Validate window size
if window_size % 128 != 0:
    raise ValueError(f"window_size must be divisible by 128, got {window_size}")

print("="*60)
print("LORA FINE-TUNING SETUP")
print("="*60)
print(f"Window size: {window_size:,} bp (2^{int(np.log2(window_size))})")
if window_size < 1048576:
    print(f"  Note: Using smaller windows than pretraining (1Mb)")
    print(f"        This reduces memory but may limit long-range context")
    
print(f"Augmentation: {'enabled' if augment else 'disabled'}")
if augment:
    print(f"  - Shift augmentation: ±{max_shift} bp")
    print(f"  - Reverse complement: 50% probability")

# Convert paths to absolute
checkpoint_path = os.path.abspath(checkpoint_path)
fasta_path = os.path.abspath(fasta_path)
signal_path = os.path.abspath(signal_path)
output_dir = os.path.abspath(output_dir)
if genome_windows:
    genome_windows = os.path.abspath(genome_windows)

# Load base model
print(f"\n1. Loading base model from {checkpoint_path}...")
base_model = dna_model.create(
    checkpoint_path=checkpoint_path,
    organism_settings=None,
    device=None
)
print(f"   ✓ Model loaded on {base_model._device_context._device}")

# Get target layers
print(f"\n2. Identifying LoRA target layers...")
target_layers = get_attention_qkv_layers()
print(f"   ✓ Targeting {len(target_layers)} Q/K/V layers")

# Setup data loader BEFORE model (need to know total steps for LR schedule)
print(f"\n3. Setting up data loader...")
dataset = CutAndTagDataset(
    fasta_path=fasta_path,
    signal_bedgraph=signal_path,
    genome_windows=genome_windows,
    window_size=window_size,
    organism_id=organism_id,
    augment=augment,
    max_shift=max_shift,
)

data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=collate_to_jax,
    drop_last=True,  # Drop incomplete batches for multi-GPU training
)

# Calculate LR schedule parameters
steps_per_epoch = len(data_loader)
total_steps = steps_per_epoch * epochs
warmup_steps = int(0.05 * total_steps)  # 5% warmup
print(f"   LR schedule: {warmup_steps} warmup steps, {total_steps} total steps")

# Initialize LoRA model with LR schedule
print(f"\n4. Initializing LoRA model (rank={lora_rank}, alpha={lora_alpha})...")
lora_model = LoRAAlphaGenomeModel(
    base_model=base_model,
    lora_rank=lora_rank,
    lora_alpha=lora_alpha,
    target_layers=target_layers,
    learning_rate=learning_rate,
    mesh=mesh,  # Pass mesh for multi-GPU parameter replication
    warmup_steps=warmup_steps,
    total_steps=total_steps,
)

# Training loop
print(f"\n5. Starting training for {epochs} epochs...")
if num_devices > 1:
    print(f"\n   Multi-GPU Training Notes:")
    print(f"   - Each GPU will process {batch_size // num_devices} examples per batch")
    print(f"   - Parameters are replicated across all {num_devices} devices")
    print(f"   - Check nvidia-smi: all GPUs should show similar memory (~{37000 // num_devices + 500}MB)")
    print(f"   - Gradients are automatically averaged across devices")
    print()
    
print("="*60)

# Create loss function following Instructions.md requirements
# Uses Borzoi-style Poisson + Multinomial NLL loss with proper target scaling
# This matches the AlphaGenome paper's ATAC/DNase loss specification
#
# Numerical stability improvements:
# - Predictions are sanitized for NaN/Inf before loss computation
# - Gradients are sanitized after computation (NaN/Inf -> 0)
# - Batches with zero/near-zero gradients are skipped
# - Loss function uses stop_gradient for extreme edge cases
loss_fn = cut_and_tag_loss_baskerville
print(f"\n   Using Borzoi-style Poisson+Multinomial NLL loss")
print(f"   (Matches AlphaGenome paper specification for ATAC-seq tracks)")
print(f"   With numerical stability safeguards against NaN gradients")

best_loss = float('inf')

# # Initialize gradient norm tracking
# if not hasattr(train, '_grad_norms'):
#     train._grad_norms = []
grad_norms = []  # Track gradient norms for all batches

# for epoch in range(epochs):
epoch = 0
    
epoch_losses = []
# train._grad_norms = []  # Reset for each epoch

print(f"\nEpoch {epoch + 1}/{epochs}")
print("-" * 60)

# Corrected training loop
for batch_idx, (sequences, organism_ids, targets) in enumerate(data_loader):
    if batch_idx >= 30:  # Test first 30 batches
        break
    
    # Progress indicator for first batch
    if batch_idx == 0:
        print(f"  Compiling JAX functions...")
        import time
        start_time = time.time()
    
    # Shard batch across devices
    if batch_sharding is not None:
        sequences = jax.device_put(sequences, batch_sharding)
        organism_ids = jax.device_put(organism_ids, organism_sharding)
        targets = {'signal': jax.device_put(targets['signal'], signal_sharding)}
    
    # Skip batches with extremely low signal
    target_sum = float(jnp.sum(targets['signal']))
    if target_sum < 1e-6:
        continue
    
    # Training step
    loss_value, predictions, grad_norm = lora_model.train_step(
        sequences, organism_ids, targets, loss_fn
    )
    
    # Check for NaN/Inf
    if jnp.isnan(loss_value) or jnp.isinf(loss_value):
        print(f"\n  ⚠️  WARNING: NaN/Inf loss at batch {batch_idx + 1}")
        break
    
    if grad_norm < 1e-10:
        print(f"\n  ⚠️  WARNING: Zero gradient at batch {batch_idx + 1}")
        continue
    
    # Store values
    epoch_losses.append(float(loss_value))
    grad_norms.append(float(grad_norm))  # DON'T reset before appending
    
    # Log progress
    if batch_idx == 0:
        compile_time = time.time() - start_time
        print(f"  ✓ Compilation complete! Took {compile_time:.1f}s")
        print(f"  Loss: {float(loss_value):.6f}")
        print(f"  Grad Norm: {float(grad_norm):.6e}")
        print(f"  Training on {len(data_loader)} batches...")
    
    # Print every 10 batches
    if (batch_idx + 1) % 10 == 0:
        avg_loss = sum(epoch_losses[-10:]) / min(10, len(epoch_losses))
        avg_grad_norm = sum(grad_norms[-10:]) / min(10, len(grad_norms))
        print(f"  Batch {batch_idx + 1}/{len(data_loader)}: Loss = {avg_loss:.6f}, Grad Norm = {avg_grad_norm:.6e}")

print(f"\n✓ Completed {len(epoch_losses)} batches")
print(f"Final avg loss: {sum(epoch_losses) / len(epoch_losses):.6f}")
print(f"Final avg grad norm: {sum(grad_norms) / len(grad_norms):.6e}")
print(f"Grad norm stats: min={min(grad_norms):.6e}, max={max(grad_norms):.6e}")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Best loss: {best_loss:.6f}")
print(f"Checkpoints saved to: {output_dir}")

# Save training config
config = {
    'checkpoint_path': checkpoint_path,
    'lora_rank': lora_rank,
    'lora_alpha': lora_alpha,
    'learning_rate': learning_rate,
    'epochs': epochs,
    'batch_size': batch_size,
    'window_size': window_size,
    'augment': augment,
    'max_shift': max_shift,
    'best_loss': best_loss,
    'target_layers': target_layers,
}

with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
    json.dump(config, f, indent=2)