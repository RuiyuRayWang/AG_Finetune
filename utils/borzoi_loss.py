"""
Borzoi-style loss functions and scaling utilities for genomic signal prediction.

Implements the target scaling, prediction scaling, and multinomial + Poisson NLL loss
as described in the AlphaGenome/Borzoi papers.

Reference: AlphaGenome Extended Data Section on Output Heads and Loss Functions
"""

import jax
import jax.numpy as jnp
from typing import Dict


def targets_scaling(
    targets: jnp.ndarray,
    track_means: jnp.ndarray,
    apply_squashing: bool = False
) -> jnp.ndarray:
    """
    Scale targets before loss calculation for numerical stability.
    
    Follows AlphaGenome/Borzoi approach:
    1. Normalize by dividing by mean of non-zero values
    2. Smooth clipping for high values: Where(x > 10, 2*sqrt(x*10) - 10, x)
    3. Optional power transform for RNA-seq: x^0.75
    
    Args:
        targets: Array of shape [batch, seq_len, num_tracks] or [batch, seq_len]
        track_means: Pre-calculated mean of non-zero values per track [num_tracks] or scalar
        apply_squashing: Apply power transform (True for RNA-seq, False for others)
        
    Returns:
        Scaled targets with same shape as input
    """
    # Normalize by track means
    targets = targets / (track_means + 1e-7)
    
    # Apply squashing for RNA-seq only
    if apply_squashing:
        targets = targets ** 0.75
    
    # Smooth clipping for high values
    # Where(x > 10, 2*sqrt(x*10) - 10, x)
    targets = jnp.where(
        targets > 10.0,
        2.0 * jnp.sqrt(targets * 10.0) - 10.0,
        targets
    )
    
    return targets


def predictions_scaling(
    predictions: jnp.ndarray,
    track_means: jnp.ndarray,
    apply_squashing: bool = False
) -> jnp.ndarray:
    """
    Inverse scaling for predictions (reverse of targets_scaling).
    
    Used when evaluating model predictions against original experimental data.
    
    Args:
        predictions: Array of shape [batch, seq_len, num_tracks] or [batch, seq_len]
        track_means: Pre-calculated mean of non-zero values per track [num_tracks] or scalar
        apply_squashing: Reverse power transform (True for RNA-seq, False for others)
        
    Returns:
        Unscaled predictions in original data space
    """
    # Reverse smooth clipping
    # Where(x > 10, (x + 10)^2 / 40, x)
    predictions = jnp.where(
        predictions > 10.0,
        (predictions + 10.0) ** 2 / 40.0,
        predictions
    )
    
    # Reverse power transform for RNA-seq
    if apply_squashing:
        predictions = predictions ** (1.0 / 0.75)
    
    # Reverse normalization
    predictions = predictions * track_means
    
    return predictions


def multinomial_nll_loss(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    num_segments: int = 8,
    multinomial_weight: float = 5.0,
) -> float:
    """
    Multinomial + Poisson negative log-likelihood loss (Borzoi-style).
    
    The loss is calculated over segments of the sequence axis:
    1. Divide sequence into num_segments equal parts
    2. For each segment, compute:
       - Poisson NLL on total counts (scaled by 1/segment_length)
       - Multinomial NLL on distribution across positions (weighted by multinomial_weight)
    
    Args:
        predictions: Predicted signal [batch, seq_len] or [batch, seq_len, num_tracks]
        targets: Target signal [batch, seq_len] or [batch, seq_len, num_tracks]
        num_segments: Number of segments to split sequence into (default 8)
        multinomial_weight: Weight for multinomial component (default 5.0)
        
    Returns:
        Scalar loss value
    """
    # Handle both single-track [batch, seq_len] and multi-track [batch, seq_len, num_tracks]
    if len(predictions.shape) == 2:
        # Add track dimension for consistency
        predictions = predictions[..., jnp.newaxis]  # [batch, seq_len, 1]
        targets = targets[..., jnp.newaxis]  # [batch, seq_len, 1]
    
    batch_size, seq_len, num_tracks = predictions.shape
    
    # Calculate segment resolution (bins per segment)
    multinomial_resolution = seq_len // num_segments
    
    # Reshape to [batch, num_segments, multinomial_resolution, num_tracks]
    pred_reshaped = predictions.reshape(
        batch_size, num_segments, multinomial_resolution, num_tracks
    )
    target_reshaped = targets.reshape(
        batch_size, num_segments, multinomial_resolution, num_tracks
    )
    
    # Sum across positions within each segment: [batch, num_segments, 1, num_tracks]
    sum_pred = jnp.sum(pred_reshaped, axis=2, keepdims=True)
    sum_target = jnp.sum(target_reshaped, axis=2, keepdims=True)
    
    # Poisson NLL on segment totals
    # NLL = sum_pred - sum_target * log(sum_pred + eps)
    poisson_loss = jnp.sum(
        sum_pred - sum_target * jnp.log(sum_pred + 1e-7)
    )
    
    # Scale Poisson by segment length (as in Borzoi)
    poisson_loss = poisson_loss / multinomial_resolution
    
    # Multinomial NLL on distribution within segments
    # Get probability distribution: p_i = x_i / sum(x)
    multinomial_prob = pred_reshaped / (sum_pred + 1e-7)
    
    # Multinomial NLL: -sum(target * log(prob))
    positional_loss = jnp.sum(
        -target_reshaped * jnp.log(multinomial_prob + 1e-7)
    )
    
    # Combine losses with weighting
    total_loss = poisson_loss + multinomial_weight * positional_loss
    
    # Average over batch
    total_loss = total_loss / batch_size
    
    return total_loss


def compute_track_means(signal_data: Dict[str, any], percentile: float = 50.0) -> float:
    """
    Compute the mean of non-zero signal values for normalization.
    
    This should be pre-calculated across the entire genome during dataset generation,
    but for single-track fine-tuning we can estimate from a sample.
    
    Args:
        signal_data: Dictionary of chromosome -> pandas DataFrame with signal
        percentile: Use percentile instead of mean for robustness to outliers
        
    Returns:
        Mean (or percentile) of non-zero values
    """
    import numpy as np
    
    all_values = []
    for chrom, df in signal_data.items():
        values = df['score'].values
        # Only use non-zero values
        nonzero = values[values > 0]
        if len(nonzero) > 0:
            all_values.extend(nonzero)
    
    if len(all_values) == 0:
        return 1.0  # Fallback
    
    # Use percentile for robustness
    return float(np.percentile(all_values, percentile))


def cut_and_tag_loss(
    predictions: Dict,
    targets: Dict,
    track_mean: float = None,
    num_segments: int = 8,
    multinomial_weight: float = 5.0,
) -> float:
    """
    Loss function for CUT&Tag signal prediction using Borzoi-style loss.
    
    CUT&Tag uses Tn5 tagmentation (like ATAC-seq), so we use the ATAC head
    which predicts at base-pair resolution.
    
    Args:
        predictions: Model predictions dict from AlphaGenome
        targets: Ground truth dict with 'signal' key [batch, seq_len]
        track_mean: Pre-calculated mean of non-zero signal values (for normalization)
        num_segments: Number of segments for multinomial loss (default 8)
        multinomial_weight: Weight for multinomial component (default 5.0)
        
    Returns:
        Loss value (multinomial + Poisson NLL)
    """
    # Extract ATAC predictions at base-pair resolution
    # AlphaGenome's ATAC head already applies Softplus + learnable scaling
    pred_dict = predictions.get('atac', {})
    
    # Use scaled predictions (these already have softplus and per-track scaling)
    pred_signal = pred_dict['scaled_predictions_1bp']  # Shape: [batch, seq_len, num_tracks]
    
    # For single-track CUT&Tag, average across tracks dimension
    if len(pred_signal.shape) == 3:
        pred_signal = jnp.mean(pred_signal, axis=-1)  # [batch, seq_len]
    
    # Safety check: ensure predictions are finite and non-negative
    pred_signal = jnp.where(jnp.isfinite(pred_signal), pred_signal, 0.0)
    pred_signal = jnp.maximum(pred_signal, 0.0)
    
    # Target signal at base-pair resolution (raw counts in bfloat16)
    # AlphaGenome stores targets as bfloat16 for numerical efficiency
    target_signal = jnp.asarray(targets['signal'], dtype=jnp.bfloat16)  # [batch, seq_len]
    
    # Ensure shapes match
    min_len = min(pred_signal.shape[1], target_signal.shape[1])
    pred_signal = pred_signal[:, :min_len]
    target_signal = target_signal[:, :min_len]
    
    # Ensure targets are valid (bfloat16 maintains the dtype)
    target_signal = jnp.where(jnp.isfinite(target_signal), target_signal, jnp.bfloat16(0.0))
    target_signal = jnp.maximum(target_signal, jnp.bfloat16(0.0))
    
    # Apply target scaling for numerical stability
    if track_mean is None:
        # Estimate from current batch if not provided
        # Use JAX-compatible operations (no boolean indexing inside JIT)
        nonzero_mask = target_signal > 0
        nonzero_count = jnp.sum(nonzero_mask)
        nonzero_sum = jnp.sum(jnp.where(nonzero_mask, target_signal, 0.0))
        track_mean = jnp.where(nonzero_count > 0, nonzero_sum / nonzero_count, 1.0) + 1e-7
    
    # Convert track_mean to bfloat16 to match target/prediction precision
    # This maintains consistency with AlphaGenome's mixed precision policy
    track_mean = jnp.array(track_mean, dtype=jnp.bfloat16)
    
    # Scale targets (normalize + smooth clipping)
    # For CUT&Tag (like ATAC), don't use squashing (that's for RNA-seq)
    target_scaled = targets_scaling(target_signal, track_mean, apply_squashing=False)
    
    # Predictions from AlphaGenome are already in the right space (Softplus output)
    # But we need to scale them to match the target scaling
    pred_scaled = targets_scaling(pred_signal, track_mean, apply_squashing=False)
    
    # Compute multinomial + Poisson NLL loss
    loss = multinomial_nll_loss(
        pred_scaled,
        target_scaled,
        num_segments=num_segments,
        multinomial_weight=multinomial_weight
    )
    
    # Ensure loss is finite
    loss = jnp.where(jnp.isfinite(loss), loss, 1e6)
    
    return loss
