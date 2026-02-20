import os
import sys
import numpy as np
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from torch.utils.data import DataLoader
from alphagenome_research.model import dna_model
from models.lora_model import LoRAAlphaGenomeModel
from dataset.dataset import CutAndTagDataset
from utils.lora_utils import get_attention_qkv_layers
from utils.borzoi_loss_v2 import cut_and_tag_loss_baskerville

print("="*80)
print("DETAILED GRADIENT NORM DEBUGGING")
print("="*80)

# Configuration
checkpoint_path = '/home/ray/AG_Finetune/checkpoints/pre_trained/alphagenome-all-folds'
fasta_path = '/home/ray/AG_Finetune/data/resources/mm10.fa'
signal_path = '/home/ray/AG_Finetune/data/CUTTag/processed_bedgraph/D1MSN_H3K27ac_averaged_normalized.bedgraph'
batch_size = 8
learning_rate = 1e-5
lora_rank = 8
lora_alpha = 16.0
window_size = 131072
organism_id = 1
genome_windows = None
augment = True
max_shift = 1024

def collate_to_jax(batch: List[Tuple[np.ndarray, int, Dict]]) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    sequences = [item[0] for item in batch]
    organism_ids = [item[1] for item in batch]
    signals = [item[2]['signal'] for item in batch]
    return (
        jnp.array(sequences, dtype=jnp.float32),
        jnp.array(organism_ids, dtype=jnp.int32),
        {'signal': jnp.array(signals, dtype=jnp.float32)}
    )

# Setup multi-GPU
num_devices = jax.device_count()
print(f"\nDevices: {num_devices} GPUs")
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

# Load model
print("\nLoading base model...")
base_model = dna_model.create(
    checkpoint_path=checkpoint_path,
    organism_settings=None,
    device=None
)

print("Getting target layers...")
target_layers = get_attention_qkv_layers()

# Setup data
print("Setting up dataset...")
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
    num_workers=0,
    collate_fn=collate_to_jax,
    drop_last=True,
)

# Calculate LR schedule
steps_per_epoch = len(data_loader)
total_steps = steps_per_epoch * 10
warmup_steps = int(0.05 * total_steps)

# Initialize LoRA model
print("Initializing LoRA model...")
lora_model = LoRAAlphaGenomeModel(
    base_model=base_model,
    lora_rank=lora_rank,
    lora_alpha=lora_alpha,
    target_layers=target_layers,
    learning_rate=learning_rate,
    mesh=mesh,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
)

loss_fn = cut_and_tag_loss_baskerville

print("\n" + "="*80)
print("STARTING DETAILED GRADIENT DEBUGGING (50 batches)")
print("="*80)
print("Legend:")
print("  grad_norm = -1.0  → Gradients contain NaN")
print("  grad_norm = -2.0  → Gradients contain Inf")
print("  grad_norm = -3.0  → Parameters contain NaN")
print("  grad_norm = -4.0  → Parameters contain Inf")
print("  grad_norm > 0     → Normal gradient norm")
print("="*80 + "\n")

import time

epoch_losses = []
grad_norms = []
target_sums = []

for batch_idx, (sequences, organism_ids, targets) in enumerate(data_loader):
    if batch_idx >= 50:  # Test 50 batches
        break
    
    if batch_idx == 0:
        print("Compiling JAX functions...")
        start_time = time.time()
    
    # Shard batch
    if batch_sharding is not None:
        sequences = jax.device_put(sequences, batch_sharding)
        organism_ids = jax.device_put(organism_ids, organism_sharding)
        targets = {'signal': jax.device_put(targets['signal'], signal_sharding)}
    
    target_sum = float(jnp.sum(targets['signal']))
    target_sums.append(target_sum)
    
    # Training step
    loss_value, predictions, grad_norm = lora_model.train_step(
        sequences, organism_ids, targets, loss_fn
    )
    
    loss_val = float(loss_value)
    grad_val = float(grad_norm)
    
    epoch_losses.append(loss_val)
    grad_norms.append(grad_val)
    
    if batch_idx == 0:
        compile_time = time.time() - start_time
        print(f"✓ Compilation took {compile_time:.1f}s\n")
    
    # Print INDIVIDUAL batch info (not averaged)
    status = ""
    if grad_val < 0:
        if grad_val == -1.0:
            status = "← GRADIENTS CONTAIN NaN"
        elif grad_val == -2.0:
            status = "← GRADIENTS CONTAIN Inf"
        elif grad_val == -3.0:
            status = "← PARAMETERS CONTAIN NaN ⚠️⚠️⚠️"
        elif grad_val == -4.0:
            status = "← PARAMETERS CONTAIN Inf ⚠️⚠️⚠️"
    elif grad_val < 1e-10:
        status = "← Zero gradient"
    
    print(f"Batch {batch_idx+1:3d}: Loss={loss_val:9.6f}  Grad={grad_val:12.6e}  Target_sum={target_sum:12.1f}  {status}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total batches: {len(grad_norms)}")
print(f"Normal grad_norms (>0): {sum(1 for g in grad_norms if g > 0)}")
print(f"NaN in gradients (=-1): {sum(1 for g in grad_norms if g == -1.0)}")
print(f"Inf in gradients (=-2): {sum(1 for g in grad_norms if g == -2.0)}")
print(f"NaN in params (=-3): {sum(1 for g in grad_norms if g == -3.0)}")
print(f"Inf in params (=-4): {sum(1 for g in grad_norms if g == -4.0)}")
print(f"Zero gradients (<1e-10): {sum(1 for g in grad_norms if 0 <= g < 1e-10)}")

# Find when NaN first appears
nan_batches = [i+1 for i, g in enumerate(grad_norms) if g < 0]
if nan_batches:
    print(f"\nBatches with NaN/Inf gradients: {nan_batches[:10]}...")
    print(f"First NaN/Inf at batch: {nan_batches[0]}")
else:
    print("\n✅ No NaN/Inf gradients detected!")

# Statistics
valid_norms = [g for g in grad_norms if g > 0]
if valid_norms:
    print(f"\nValid gradient norm stats:")
    print(f"  Mean: {np.mean(valid_norms):.6e}")
    print(f"  Min:  {np.min(valid_norms):.6e}")
    print(f"  Max:  {np.max(valid_norms):.6e}")
