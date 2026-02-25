#!/usr/bin/env python3
"""
Test with ACTUAL training data from TensorFlow dataset.

This is the final piece - if this fails, we know it's the data.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import tensorflow as tf
import pyBigWig
import sys
sys.path.insert(0, '/home/ray/AG_Finetune')

from models.model import HeadReplaceAlphaGenomeModel
from utils.borzoi_loss import cut_and_tag_loss_baskerville

print("=" * 70)
print("ACTUAL TRAINING DATA DIAGNOSTIC")
print("=" * 70)

# Load config
import yaml
with open('/home/ray/AG_Finetune/configs/config_bigwig.yaml') as f:
    config = yaml.safe_load(f)

# Setup
mesh = Mesh(jax.devices(), ('batch',))
batch_sharding = NamedSharding(mesh, P('batch'))
organism_sharding = NamedSharding(mesh, P('batch'))
signal_sharding = NamedSharding(mesh, P('batch', None))
param_sharding = NamedSharding(mesh, P())

print("\n[Step 1] Loading safe windows...")
windows = []
with open(config['safe_windows']) as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            windows.append((parts[0], int(parts[1]), int(parts[2])))
print(f"✓ Loaded {len(windows)} safe windows")

print("\n[Step 2] Creating TensorFlow dataset (8 batches only)...")
def data_generator():
    """Generate training examples from safe windows."""
    fasta_path = config['fasta']
    bigwig_path = config['signal']
    window_size = config['window_size']
    organism_id = config['organism_id']
    
    import pyfaidx
    genome = pyfaidx.Fasta(fasta_path)
    bigwig = pyBigWig.open(bigwig_path)
    
    for chrom, start, end in windows[:64]:  # Only 64 windows = 8 batches
        if end - start != window_size:
            continue
        
        # Get sequence
        seq_str = str(genome[chrom][start:end])
        if len(seq_str) != window_size:
            continue
        
        # One-hot encode
        seq_1hot = jnp.zeros((window_size, 4), dtype=jnp.float32)
        for i, nt in enumerate(seq_str.upper()):
            if nt == 'A':
                seq_1hot = seq_1hot.at[i, 0].set(1.0)
            elif nt == 'C':
                seq_1hot = seq_1hot.at[i, 1].set(1.0)
            elif nt == 'G':
                seq_1hot = seq_1hot.at[i, 2].set(1.0)
            elif nt == 'T':
                seq_1hot = seq_1hot.at[i, 3].set(1.0)
        
        # Get signal
        signal = jnp.array(bigwig.values(chrom, start, end), dtype=jnp.float32)
        signal = jnp.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        
        yield {
            'sequence': seq_1hot,
            'organism': organism_id,
            'signal': signal,
        }

# Create dataset
dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature={
        'sequence': tf.TensorSpec(shape=(config['window_size'], 4), dtype=tf.float32),
        'organism': tf.TensorSpec(shape=(), dtype=tf.int32),
        'signal': tf.TensorSpec(shape=(config['window_size'],), dtype=tf.float32),
    }
)
dataset = dataset.batch(8, drop_remainder=True)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

print(f"✓ Dataset created")

print("\n[Step 3] Initializing model...")
model = HeadReplaceAlphaGenomeModel(
    checkpoint_dir=config['checkpoint'],
    learning_rate=config['learning_rate'],
    warmup_steps=1000,
    total_steps=50000,
    mesh=mesh,
    batch_sharding=batch_sharding,
    organism_sharding=organism_sharding,
    signal_sharding=signal_sharding,
    param_sharding=param_sharding,
)
print(f"✓ Model initialized")

print("\n[Step 4] Testing with actual training data...")
for batch_idx, batch in enumerate(dataset.take(3)):  # Test first 3 batches
    print(f"\n  Batch {batch_idx + 1}:")
    
    # Convert to JAX and shard
    sequences = jnp.array(batch['sequence'].numpy())
    organisms = jnp.array(batch['organism'].numpy())
    signals = jnp.array(batch['signal'].numpy())
    
    print(f"    Shapes: seq={sequences.shape}, org={organisms.shape}, sig={signals.shape}")
    print(f"    Signal range: [{jnp.min(signals):.2f}, {jnp.max(signals):.2f}]")
    print(f"    Signal mean: {jnp.mean(signals):.2f}")
    
    # Check for data issues
    if jnp.any(jnp.isnan(signals)):
        print(f"    ⚠ WARNING: NaN in signal!")
    if jnp.any(jnp.isinf(signals)):
        print(f"    ⚠ WARNING: Inf in signal!")
    if jnp.max(signals) > 100000:
        print(f"    ⚠ WARNING: Extremely large signal values!")
    
    # Shard
    sequences = jax.device_put(sequences, batch_sharding)
    organisms = jax.device_put(organisms, organism_sharding)
    signals = jax.device_put(signals, signal_sharding)
    
    try:
        # Run training step
        loss, predictions, grad_norm = model.train_step(
            sequences, organisms, signals, cut_and_tag_loss_baskerville
        )
        
        print(f"    ✓ Training step succeeded")
        print(f"      Loss: {loss:.4f}")
        print(f"      Grad norm: {grad_norm:.4f}")
        
    except Exception as e:
        print(f"    ✗ Training step FAILED: {e}")
        print(f"\n    → This batch triggered the NCCL error!")
        print(f"    → Batch index: {batch_idx}")
        print(f"    → Signal stats: min={jnp.min(signals):.2f}, max={jnp.max(signals):.2f}, mean={jnp.mean(signals):.2f}")
        
        import traceback
        traceback.print_exc()
        
        # Save the problematic batch for inspection
        print("\n    Saving problematic batch to /tmp/bad_batch.npz...")
        jnp.savez('/tmp/bad_batch.npz',
                  sequences=sequences,
                  organisms=organisms,
                  signals=signals)
        print("    → Saved to /tmp/bad_batch.npz")
        
        exit(1)

print("\n" + "=" * 70)
print("✓ ALL DATA TESTS PASSED")
print("=" * 70)
print("""
If this passes, then:
  - The model wrapper works
  - The data loading works
  - Everything should work!

The issue must be in the full training script's setup or a rare edge case.
""")
