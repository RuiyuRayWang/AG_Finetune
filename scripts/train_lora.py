"""
Training script for LoRA fine-tuning of AlphaGenome on CUT&Tag data.

CUT&Tag uses Tn5 tagmentation (like ATAC-seq), so this script uses the ATAC
prediction head which operates at base-pair resolution.

Note: While AlphaGenome was pretrained on 1Mb (2^20) windows, you can fine-tune
with smaller windows (e.g., 2^18 = 262,144 bp) to reduce memory usage.

Data augmentation (following AlphaGenome pretraining):
  - Shift augmentation: randomly shift windows by ±1024 bp
  - Reverse complementation: 50% probability (both sequence and signal)

Usage:
    python train_lora.py \
        --checkpoint /path/to/alphagenome-fold-0 \
        --fasta /path/to/genome.fa \
        --signal /path/to/signal.bedGraph \
        --output ./lora_checkpoints \
        --epochs 10 \
        --batch-size 4 \
        --lr 1e-4 \
        --lora-rank 8 \
        --window-size 262144
    
    # Disable augmentation for validation/testing
    python train_lora.py ... --no-augment
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
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
import json


def collate_to_jax(batch: List[Tuple[np.ndarray, int, Dict]]) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    """
    Custom collate function to convert batch of numpy arrays to JAX arrays.
    
    Args:
        batch: List of (sequence, organism_id, targets) tuples from Dataset
    
    Returns:
        (sequences, organism_ids, targets) as JAX arrays:
            sequences: [batch, window_size, 4]
            organism_ids: [batch]
            targets: dict with 'signal' [batch, window_size]
    """
    sequences = [item[0] for item in batch]
    organism_ids = [item[1] for item in batch]
    signals = [item[2]['signal'] for item in batch]
    
    return (
        jnp.array(sequences, dtype=jnp.float32),
        jnp.array(organism_ids, dtype=jnp.int32),
        {'signal': jnp.array(signals, dtype=jnp.float32)}
    )


def cut_and_tag_loss(predictions: Dict, targets: Dict) -> float:
    """
    Loss function for CUT&Tag signal prediction.
    
    CUT&Tag uses Tn5 tagmentation (like ATAC-seq), so we use the ATAC head
    which predicts at base-pair resolution. Follows the AlphaGenome/Borzoi
    multinomial + Poisson loss over segments.
    
    Args:
        predictions: Model predictions dict from AlphaGenome
        targets: Ground truth dict with 'signal' key [batch, seq_len]
        
    Returns:
        Multinomial + Poisson loss value
    """
    # Extract ATAC predictions at base-pair resolution
    # CUT&Tag is biochemically similar to ATAC (Tn5 tagmentation)
    # Use scaled_predictions_1bp which already has softplus + learnable scaling applied
    pred_dict = predictions.get('atac', {})
    
    # Use scaled predictions (these already have softplus and per-track scaling)
    pred_signal = pred_dict['scaled_predictions_1bp']  # Shape: [batch, seq_len, num_tracks]
    
    # For CUT&Tag, average across tracks dimension
    if len(pred_signal.shape) == 3:
        pred_signal = jnp.mean(pred_signal, axis=-1)  # [batch, seq_len]
    
    # DEBUG: Print prediction statistics on first call
    if not hasattr(cut_and_tag_loss, '_pred_debug_printed'):
        print(f"\n  [DEBUG] Prediction stats (before clipping):")
        print(f"    Shape: {pred_signal.shape}")
        print(f"    Min: {jnp.min(pred_signal):.6f}")
        print(f"    Max: {jnp.max(pred_signal):.6f}")
        print(f"    Mean: {jnp.mean(pred_signal):.6f}")
        cut_and_tag_loss._pred_debug_printed = True
    
    # Clip predictions to prevent numerical instability
    pred_signal = jnp.clip(pred_signal, 1e-8, 1e6)
    
    # Target signal at base-pair resolution
    target_signal = targets['signal']  # [batch, seq_len]
    
    # DEBUG: Print signal statistics on first call
    import jax
    if not hasattr(cut_and_tag_loss, '_debug_printed'):
        print(f"\n  [DEBUG] Target signal stats:")
        print(f"    Shape: {target_signal.shape}")
        print(f"    Min: {jnp.min(target_signal):.6f}")
        print(f"    Max: {jnp.max(target_signal):.6f}")
        print(f"    Mean: {jnp.mean(target_signal):.6f}")
        print(f"    Non-zero count: {jnp.sum(target_signal > 0)}")
        cut_and_tag_loss._debug_printed = True
    
    # Ensure shapes match
    min_len = min(pred_signal.shape[1], target_signal.shape[1])
    pred_signal = pred_signal[:, :min_len]
    target_signal = target_signal[:, :min_len]
    
    # Apply AlphaGenome target scaling (as per the paper)
    # 1. Normalize by mean of non-zero values
    nonzero_mask = target_signal > 0
    target_mean = jnp.sum(target_signal * nonzero_mask) / (jnp.sum(nonzero_mask) + 1e-7)
    target_signal = target_signal / (target_mean + 1e-7)
    
    # 2. Smooth clipping for values > 10.0: 2 * sqrt(x * 10) - 10
    target_signal = jnp.where(
        target_signal > 10.0,
        2.0 * jnp.sqrt(target_signal * 10.0) - 10.0,
        target_signal
    )
    
    # Add channel dimension for multinomial loss (expects [batch, seq_len, channels])
    pred_signal = pred_signal[..., None]  # [batch, seq_len, 1]
    target_signal = target_signal[..., None]  # [batch, seq_len, 1]
    
    # AlphaGenome/Borzoi multinomial loss over 8 segments
    # Divide sequence into 8 equal segments for numerical stability
    num_segments = 8
    seq_len = pred_signal.shape[1]
    segment_len = seq_len // num_segments
    
    # Reshape into segments: [batch, num_segments, segment_len, channels]
    pred_segments = pred_signal[:, :num_segments * segment_len, :].reshape(
        pred_signal.shape[0], num_segments, segment_len, pred_signal.shape[2]
    )
    target_segments = target_signal[:, :num_segments * segment_len, :].reshape(
        target_signal.shape[0], num_segments, segment_len, target_signal.shape[2]
    )
    
    # Sum over segment length: [batch, num_segments, channels]
    sum_pred = jnp.sum(pred_segments, axis=2, keepdims=True)  # [batch, num_segments, 1, channels]
    sum_target = jnp.sum(target_segments, axis=2, keepdims=True)  # [batch, num_segments, 1, channels]
    
    # Poisson NLL on segment totals (normalized by segment length)
    poisson_loss = jnp.sum(sum_pred - sum_target * jnp.log(sum_pred + 1e-7)) / segment_len
    
    # Multinomial NLL on distribution within segments (weighted by 5.0)
    multinomial_prob = pred_segments / (sum_pred + 1e-7)  # Normalize to probabilities
    multinomial_loss = jnp.sum(-target_segments * jnp.log(multinomial_prob + 1e-7))
    
    # DEBUG: Print loss components on first call
    if not hasattr(cut_and_tag_loss, '_loss_debug_printed'):
        print(f"\n  [DEBUG] Loss components:")
        print(f"    Poisson: {poisson_loss:.6f}")
        print(f"    Multinomial (raw): {multinomial_loss:.6f}")
        print(f"    sum_pred stats - min: {jnp.min(sum_pred):.6f}, max: {jnp.max(sum_pred):.6f}, mean: {jnp.mean(sum_pred):.6f}")
        print(f"    sum_target stats - min: {jnp.min(sum_target):.6f}, max: {jnp.max(sum_target):.6f}, mean: {jnp.mean(sum_target):.6f}")
        cut_and_tag_loss._loss_debug_printed = True
    
    # Combined loss: Poisson + 5.0 * Multinomial (as per Borzoi/AlphaGenome)
    total_loss = poisson_loss + 5.0 * multinomial_loss
    
    # Normalize by batch size and number of channels to prevent explosion
    batch_size = pred_signal.shape[0]
    num_channels = pred_signal.shape[-1]
    total_loss = total_loss / (batch_size * num_channels)
    
    return total_loss


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


def train(
    checkpoint_path: str,
    fasta_path: str,
    signal_path: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    window_size: int = 262144,  # 2^18 bp, can use smaller than 1Mb for fine-tuning
    organism_id: int = 0,
    genome_windows: str = None,
    augment: bool = True,  # Enable data augmentation
    max_shift: int = 1024,  # Maximum shift for augmentation
    num_workers: int = 0,  # Number of data loading workers
):
    """
    Train LoRA model on CUT&Tag data.
    
    Args:
        checkpoint_path: Path to pre-trained AlphaGenome checkpoint
        fasta_path: Path to reference genome FASTA
        signal_path: Path to CUT&Tag signal BedGraph (normalized, base-resolution)
        output_dir: Directory to save LoRA checkpoints
        epochs: Number of training epochs
        batch_size: Batch size (reduce if OOM)
        learning_rate: Learning rate
        lora_rank: LoRA rank (lower = fewer parameters, less overfitting)
        lora_alpha: LoRA alpha scaling
        window_size: Sequence window size in bp. AlphaGenome was pretrained on
                    2^20 (1,048,576 bp = 1Mb), but you can use smaller windows
                    for fine-tuning to reduce memory:
                    - 2^18 (262,144 bp) = ~262kb (recommended for fine-tuning)
                    - 2^17 (131,072 bp) = ~131kb (lower memory)
                    - 2^16 (65,536 bp) = ~66kb (minimal memory)
                    Must be divisible by 128 for compatibility with model heads.
        organism_id: 0=human, 1=mouse
        genome_windows: Optional BED file with training regions
        augment: Enable data augmentation (shift + reverse complement)
        max_shift: Maximum shift in bp for shift augmentation (default 1024)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup device mesh for multi-GPU training
    num_devices = jax.device_count()
    
    if num_devices > 1:
        print("="*60)
        print(f"MULTI-GPU DETECTED: {num_devices} devices")
        print("="*60)
        print(f"Devices: {jax.devices()}")
        print(f"Using JAX automatic parallelization with NamedSharding")
        print()
        
        # Create device mesh for sharding batches across devices
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
    
    # Initialize LoRA model
    print(f"\n3. Initializing LoRA model (rank={lora_rank}, alpha={lora_alpha})...")
    lora_model = LoRAAlphaGenomeModel(
        base_model=base_model,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        target_layers=target_layers,
        learning_rate=learning_rate,
    )
    
    # Setup data loader
    print(f"\n4. Setting up data loader...")
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
    )
    print(f"   ✓ Loaded {len(dataset.regions)} training regions")
    
    # Training loop
    print(f"\n5. Starting training for {epochs} epochs...")
    print("="*60)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_losses = []
        
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 60)
        
        for batch_idx, (sequences, organism_ids, targets) in enumerate(data_loader):
            # Progress indicator for first batch (JAX compilation happens here)
            if batch_idx == 0:
                print(f"  Compiling JAX functions (this may take 2-5 minutes on first batch)...")
                import time
                start_time = time.time()
            
            # Shard batch across devices using JAX automatic parallelization
            if batch_sharding is not None:
                # Use jax.device_put to physically shard arrays across devices
                # JAX will automatically parallelize computation based on this sharding
                sequences = jax.device_put(sequences, batch_sharding)
                organism_ids = jax.device_put(organism_ids, organism_sharding)
                targets = {
                    'signal': jax.device_put(targets['signal'], signal_sharding)
                }
            
            # Training step (automatically parallelized based on input sharding)
            loss_value, predictions = lora_model.train_step(
                sequences, organism_ids, targets, cut_and_tag_loss
            )
            
            # Check for NaN loss (indicates training instability)
            if jnp.isnan(loss_value) or jnp.isinf(loss_value):
                print(f"\n  ⚠️  WARNING: NaN/Inf loss detected at batch {batch_idx + 1}")
                print(f"  This usually indicates:")
                print(f"    - Learning rate too high")
                print(f"    - Gradient explosion")
                print(f"    - Numerical instability in loss")
                print(f"  Consider:")
                print(f"    - Reducing learning rate (try --lr 1e-5 or 1e-6)")
                print(f"    - Using smaller batch size")
                print(f"    - Checking data for extreme values")
                print(f"\n  Stopping training to prevent further issues.")
                break
            
            epoch_losses.append(float(loss_value))
            
            # Log progress
            if batch_idx == 0:
                compile_time = time.time() - start_time
                print(f"  ✓ Compilation complete! Took {compile_time:.1f}s")
                
                # Visualize array sharding for debugging (multi-GPU only)
                if batch_sharding is not None:
                    print(f"\n  Array Sharding Visualization:")
                    print(f"  Sequences shape: {sequences.shape}")
                    print(f"  Sequences sharding: {sequences.sharding}")
                    # Visualize a 2D slice since visualize_array_sharding only supports 1D/2D
                    try:
                        print(f"  Sequences[:, 0, 0] (batch dimension):")
                        jax.debug.visualize_array_sharding(sequences[:, 0, 0])
                    except Exception as e:
                        print(f"  (Visualization skipped: {e})")
                    print(f"  Loss value: {loss_value}")
                    print()
                
                print(f"  Training on {len(data_loader)} batches...")
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = sum(epoch_losses[-10:]) / min(10, len(epoch_losses))
                print(f"  Batch {batch_idx + 1}/{len(data_loader)}: Loss = {avg_loss:.6f}")
        
        # Epoch summary
        if len(epoch_losses) > 0:
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Loss: {avg_epoch_loss:.6f}")
            print(f"  Batches processed: {len(epoch_losses)}")
            
            # Save checkpoint if best
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                checkpoint_file = os.path.join(output_dir, f"lora_best.pkl")
                lora_model.save_lora(checkpoint_file)
                print(f"  ✓ Saved best checkpoint (loss={best_loss:.6f})")
        else:
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  ⚠️  No batches completed (training stopped early due to NaN/Inf)")
            avg_epoch_loss = float('inf')
        
        # Save epoch checkpoint
        checkpoint_file = os.path.join(output_dir, f"lora_epoch_{epoch+1}.pkl")
        lora_model.save_lora(checkpoint_file)
    
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


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune AlphaGenome with LoRA on CUT&Tag data'
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', required=True,
                       help='Path to pre-trained AlphaGenome checkpoint')
    parser.add_argument('--fasta', required=True,
                       help='Path to reference genome FASTA')
    parser.add_argument('--signal', required=True,
                       help='Path to CUT&Tag signal BedGraph file')
    parser.add_argument('--output', required=True,
                       help='Output directory for LoRA checkpoints')
    
    # Optional arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (default: 8, must be divisible by number of GPUs)')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate (default: 1e-5, lower for stability)')
    parser.add_argument('--lora-rank', type=int, default=8,
                       help='LoRA rank (default: 8)')
    parser.add_argument('--lora-alpha', type=float, default=16.0,
                       help='LoRA alpha scaling (default: 16.0)')
    parser.add_argument('--window-size', type=int, default=131072,
                       help='Sequence window size in bp (default: 131072 = 2^17)')
    parser.add_argument('--organism', type=int, default=1,
                       help='Organism ID: 0=human, 1=mouse (default: 1)')
    parser.add_argument('--windows-bed', type=str, default=None,
                       help='BED file with training regions (optional)')
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable data augmentation (default: enabled)')
    parser.add_argument('--max-shift', type=int, default=1024,
                       help='Maximum shift for shift augmentation in bp (default: 1024)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of DataLoader worker processes (default: 4)')
    
    args = parser.parse_args()
    
    train(
        checkpoint_path=args.checkpoint,
        fasta_path=args.fasta,
        signal_path=args.signal,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        window_size=args.window_size,
        organism_id=args.organism,
        genome_windows=args.windows_bed,
        augment=not args.no_augment,
        max_shift=args.max_shift,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    main()
