#!/usr/bin/env python3
"""
Single-GPU AlphaGenome fine-tuning with gradient accumulation.

Based on official alphagenome_research/finetuning/finetune.py pattern.
Uses gradient accumulation to simulate larger batch sizes.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_singlegpu.py \\
        --config configs/config_bigwig.yaml \\
        --accumulation-steps 8 \\
        --epochs 10
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force single GPU

import argparse
import yaml
import jax
import jax.numpy as jnp
import haiku as hk
import jmp
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
import pyBigWig
import pyfaidx
from pathlib import Path
from datetime import datetime

from alphagenome_research.model import model as model_lib, dna_model
from alphagenome_research.model.metadata import metadata as metadata_lib

print(f"JAX devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")
assert jax.device_count() == 1, "This script requires exactly 1 GPU (use CUDA_VISIBLE_DEVICES=0)"


def create_dataset(config):
    """Create TensorFlow dataset from BigWig + FASTA."""
    import random
    
    # Open genome and bigwig to determine valid chromosomes
    genome = pyfaidx.Fasta(config['fasta'])
    bigwig = pyBigWig.open(config['signal'])
    window_size = config['window_size']
    
    # Get valid chromosomes (present in both genome and bigwig)
    valid_chroms = []
    for chrom in genome.keys():
        if chrom in bigwig.chroms():
            chrom_len = len(genome[chrom])
            if chrom_len >= window_size:
                valid_chroms.append((chrom, chrom_len))
    
    print(f"Found {len(valid_chroms)} valid chromosomes")
    print(f"Generating random {window_size} bp windows...")
    
    # Generate random windows (1000 windows for now)
    num_windows = 1000
    windows = []
    for _ in range(num_windows):
        chrom, chrom_len = random.choice(valid_chroms)
        max_start = chrom_len - window_size
        start = random.randint(0, max_start)
        end = start + window_size
        windows.append((chrom, start, end))
    
    print(f"Generated {len(windows)} random windows")
    
    def data_generator():
        """Generate training examples."""
        genome = pyfaidx.Fasta(config['fasta'])
        bigwig = pyBigWig.open(config['signal'])
        window_size = config['window_size']
        organism_id = config['organism_id']
        
        for chrom, start, end in windows:
            
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
    dataset = dataset.batch(1, drop_remainder=True)  # Batch size 1 for gradient accumulation
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_model(checkpoint_dir):
    """Create AlphaGenome model following official pattern."""
    # Load checkpoint
    params, state = ocp.StandardCheckpointer().restore(checkpoint_dir)
    print(f"Loaded checkpoint: {len(jax.tree.leaves(params))} param leaves")
    
    # Load metadata
    metadata = {
        dna_model.Organism.MUS_MUSCULUS: metadata_lib.load(dna_model.Organism.MUS_MUSCULUS)
    }
    
    # Create Haiku transform (official pattern)
    @hk.transform_with_state
    def forward(dna_sequence, organism_index):
        policy = jmp.get_policy('params=float32,compute=float32,output=float32')
        with hk.mixed_precision.push_policy(model_lib.AlphaGenome, policy):
            return model_lib.AlphaGenome(metadata)(dna_sequence, organism_index)
    
    return forward, params, state


def create_loss_fn(forward_fn):
    """Create loss function for single-track ATAC prediction."""
    def loss_fn(params, state, sequence, organism, target):
        """Compute MSE loss on first ATAC track."""
        (predictions, _), new_state = forward_fn.apply(params, state, None, sequence, organism)
        predictions = dna_model.extract_predictions(predictions)
        atac_1bp = predictions[dna_model.OutputType.ATAC]  # Shape: (1, seq_len, 256)
        
        # Use first track (index 0) for fine-tuning
        pred_track = atac_1bp[0, :, 0]  # Shape: (seq_len,)
        loss = jnp.mean((pred_track - target[0]) ** 2)
        
        return loss, new_state
    
    return loss_fn


def create_train_step(forward_fn, optimizer):
    """Create training step with gradient accumulation support."""
    loss_fn = create_loss_fn(forward_fn)
    
    @jax.jit
    def compute_loss_and_grads(params, state, sequence, organism, target):
        """Compute loss and gradients for one micro-batch."""
        (loss, new_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, state, sequence, organism, target
        )
        return loss, grads, new_state
    
    return compute_loss_and_grads


def train(config, args):
    """Main training loop with gradient accumulation."""
    # Setup output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_base = config.get('output') or config.get('output_dir', './checkpoints/fine_tuned')
    output_dir = Path(output_base) / f"singlegpu_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Create model
    print("\nLoading model...")
    forward_fn, params, state = create_model(config['checkpoint'])
    
    # Create optimizer
    learning_rate = config.get('learning_rate', 5e-4)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate)
    )
    opt_state = optimizer.init(params)
    
    print(f"Optimizer initialized with LR={learning_rate}")
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = create_dataset(config)
    
    # Create train step
    compute_loss_and_grads = create_train_step(forward_fn, optimizer)
    
    accumulation_steps = args.accumulation_steps
    print(f"\nGradient accumulation: {accumulation_steps} micro-batches")
    print(f"Effective batch size: {accumulation_steps}")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)
    
    global_step = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)
        
        epoch_loss = 0
        num_steps = 0
        
        data_iter = iter(dataset)
        accumulated_grads = None
        accumulated_loss = 0
        
        for micro_step in range(accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            
            # Convert to JAX arrays
            sequence = jnp.array(batch['sequence'].numpy())
            organism = jnp.array(batch['organism'].numpy())
            target = jnp.array(batch['signal'].numpy())
            
            if micro_step == 0 and global_step == 0:
                print(f"  First batch shapes: seq={sequence.shape}, org={organism.shape}, target={target.shape}")
            
            # Compute loss and gradients
            loss, grads, state = compute_loss_and_grads(
                params, state, sequence, organism, target
            )
            
            accumulated_loss += jax.device_get(loss) / accumulation_steps
            
            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = jax.tree.map(
                    lambda a, g: a + g, accumulated_grads, grads
                )
        
        if accumulated_grads is None:
            print("  No more data in epoch")
            break
        
        # Average gradients (use float division)
        accumulated_grads = jax.tree.map(
            lambda g: g / float(accumulation_steps), accumulated_grads
        )
        
        # Apply updates
        updates, opt_state = optimizer.update(accumulated_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        # Logging
        global_step += 1
        epoch_loss += accumulated_loss
        num_steps += 1
        
        if global_step % 10 == 0:
            print(f"  Step {global_step}: loss={accumulated_loss:.6f}")
        
        # Reset for next accumulation
        accumulated_grads = None
        accumulated_loss = 0
        
        print(f"  Epoch {epoch + 1} complete: avg_loss={epoch_loss / num_steps:.6f}")
        
        # Save checkpoint
        ckpt_dir = output_dir / f"epoch_{epoch + 1:03d}"
        ckpt_dir.mkdir(exist_ok=True)
        ocp.StandardCheckpointer().save(ckpt_dir, {'params': params, 'state': state})
        print(f"  Saved checkpoint: {ckpt_dir}")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Final model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Single-GPU AlphaGenome fine-tuning")
    parser.add_argument('--config', type=str, default='configs/config_bigwig.yaml',
                        help='Path to config file')
    parser.add_argument('--accumulation-steps', type=int, default=8,
                        help='Number of gradient accumulation steps (effective batch size)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    print("=" * 70)
    print("SINGLE-GPU FIN-TUNING WITH GRADIENT ACCUMULATION")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Accumulation steps: {args.accumulation_steps}")
    print(f"Epochs: {args.epochs}")
    
    train(config, args)


if __name__ == '__main__':
    main()
