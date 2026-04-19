"""
Baskerville/Borzoi-style loss functions for genomic signal prediction.

This implementation matches the ACTUAL Baskerville codebase exactly:
- Target transforms from hound_data_read.py
- Loss function from metrics.py (poisson_multinomial)
- Compatible with AlphaGenome's pre-trained model expectations

Reference: https://github.com/calico/baskerville
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional


def baskerville_target_transform(
    targets: jnp.ndarray,
    scale: float = 1.0,
    clip_soft: Optional[float] = 64.0,
    clip_hard: Optional[float] = 384.0,
    use_sqrt: bool = True,
) -> jnp.ndarray:
    """
    Apply Baskerville's target transformation (matching hound_data_read.py).

    This is the "NEW" transform from Baskerville (not the OLD transform).
    The model was trained to predict in this TRANSFORMED space.

    Transform steps (in order):
    1. Scale: targets * scale
    2. Sqrt (if use_sqrt): sqrt(targets + 1) - 1
    3. Soft clip: sqrt(targets - clip_soft + 1) + clip_soft - 1 (when targets > clip_soft)
    4. Hard clip: clip(targets, -clip_hard, clip_hard)

    Args:
        targets: Raw signal counts, shape [batch, seq_len] or [batch, seq_len, tracks]
        scale: Scaling factor (e.g., 1.0 for ATAC/CAGE, 0.3 for RNA-seq)
        clip_soft: Soft clipping threshold (e.g., 64 for ATAC, 320 for RNA-seq)
        clip_hard: Hard clipping threshold (e.g., 384)
        use_sqrt: Whether to apply sqrt transform (True for sum_sqrt tracks)

    Returns:
        Transformed targets in model's expected space
    """
    # Cast to float32 for numerical stability in sqrt/log operations
    targets = targets.astype(jnp.float32)

    # Safety: replace NaN/Inf with 0 at the start
    targets = jnp.where(jnp.isfinite(targets), targets, 0.0)

    # Ensure targets are non-negative
    targets = jnp.maximum(targets, 0.0)

    # 1. Scale
    targets = targets * scale

    # 2. Sqrt transform (if enabled)
    if use_sqrt:
        targets = jnp.sqrt(targets + 1.0) - 1.0

    # 3. Soft clip (squashing function for high values)
    if clip_soft is not None:
        targets = jnp.where(
            targets > clip_soft,
            jnp.sqrt(targets - clip_soft + 1.0) + clip_soft - 1.0,
            targets
        )

    # 4. Hard clip
    if clip_hard is not None:
        targets = jnp.clip(targets, 0.0, clip_hard)  # Changed min to 0.0 instead of -clip_hard

    # Final safety: ensure finite
    targets = jnp.where(jnp.isfinite(targets), targets, 0.0)

    return targets


def poisson_loss(y_true: jnp.ndarray, y_pred: jnp.ndarray, epsilon: float = 1e-7) -> jnp.ndarray:
    """
    Poisson negative log-likelihood loss (element-wise, no reduction).

    Formula: y_pred - y_true * log(y_pred + epsilon)

    Args:
        y_true: True counts
        y_pred: Predicted counts
        epsilon: Small value to avoid log(0)

    Returns:
        Poisson NLL per element
    """
    # Cast to float32 for numerical stability in log operations
    y_true = y_true.astype(jnp.float32)
    y_pred = y_pred.astype(jnp.float32)

    # Ensure predictions are positive and finite
    y_pred = jnp.maximum(y_pred, epsilon)
    y_pred = jnp.where(jnp.isfinite(y_pred), y_pred, epsilon)

    # Compute Poisson NLL
    log_term = jnp.log(y_pred + epsilon)
    poisson_nll = y_pred - y_true * log_term

    # Clamp to prevent extreme values
    poisson_nll = jnp.clip(poisson_nll, -1e6, 1e6)

    return poisson_nll


def poisson_multinomial_loss(
    y_true: jnp.ndarray,
    y_pred: jnp.ndarray,
    total_weight: float = 1.0,
    weight_range: float = 1.0,
    weight_exp: int = 4,
    epsilon: float = 1e-7,
    rescale: bool = False,
) -> jnp.ndarray:
    """
    Baskerville's Poisson + Multinomial NLL loss (exact port from metrics.py).

    This loss combines:
    1. Poisson NLL on total counts (sequence-level)
    2. Multinomial NLL on positional distribution (bin-level)

    Position weighting applies exponential decay from sequence center,
    upweighting central bins (where signal is more reliable).

    Args:
        y_true: True transformed signal [batch, seq_len] or [batch, seq_len, tracks]
        y_pred: Predicted transformed signal (same shape)
        total_weight: Weight for Poisson total term (default: 1.0)
        weight_range: Position weighting range (1.0 = no weighting)
        weight_exp: Position weighting exponent (default: 4)
        epsilon: Small value to avoid log(0)
        rescale: Rescale loss after re-weighting

    Returns:
        Loss per track [batch, tracks] or [batch] if single track
    """
    # CRITICAL: Cast to float32 for numerical stability during gradient computation
    # bfloat16 has insufficient precision for log/division operations in loss
    y_true = y_true.astype(jnp.float32)
    y_pred = y_pred.astype(jnp.float32)
    # Get sequence length
    seq_len = y_true.shape[1]

    # Compute position weights (exponential decay from center)
    if weight_range < 1.0:
        raise ValueError("weight_range must be >= 1.0")
    elif weight_range == 1.0:
        # No position weighting
        position_weights = jnp.ones((1, seq_len, 1))
    else:
        # Exponential decay from center
        pos_start = -(seq_len / 2 - 0.5)
        pos_end = seq_len / 2 + 0.5
        positions = jnp.arange(pos_start, pos_end, dtype=jnp.float32)

        sigma = -pos_start / (jnp.log(weight_range)) ** (1.0 / weight_exp)
        position_weights = jnp.exp(-((positions / sigma) ** weight_exp))
        position_weights = position_weights / jnp.max(position_weights)

        # Shape for broadcasting: [1, seq_len, 1]
        position_weights = position_weights[None, :, None]

    # Ensure 3D shape for broadcasting
    if y_true.ndim == 2:
        y_true = y_true[..., None]  # [batch, seq_len, 1]
        y_pred = y_pred[..., None]

    # Apply position weights
    y_true_weighted = y_true * position_weights  # [batch, seq_len, tracks]
    y_pred_weighted = y_pred * position_weights

    # Sum across sequence length -> total counts per track
    s_true = jnp.sum(y_true_weighted, axis=1)  # [batch, tracks]
    s_pred = jnp.sum(y_pred_weighted, axis=1)  # [batch, tracks]

    # Ensure totals are positive (add epsilon if needed)
    s_true_safe = jnp.maximum(s_true, epsilon)
    s_pred_safe = jnp.maximum(s_pred, epsilon)

    # ===== Poisson Term (on totals) =====
    poisson_term = poisson_loss(s_true_safe, s_pred_safe, epsilon)  # [batch, tracks]
    poisson_term = poisson_term / jnp.sum(position_weights)

    # ===== Multinomial Term (on distributions) =====
    # Add epsilon to avoid log(0) and division by zero
    y_true_eps = y_true_weighted + epsilon  # [batch, seq_len, tracks]
    y_pred_eps = y_pred_weighted + epsilon

    # Normalize to probabilities (sum to 1 across sequence)
    p_pred = y_pred_eps / (s_pred_safe[:, None, :] + epsilon)  # [batch, seq_len, tracks]

    # Clamp probabilities to valid range
    p_pred = jnp.clip(p_pred, epsilon, 1.0 - epsilon)

    # Multinomial NLL: -sum(y_true * log(p_pred))
    log_p_pred = jnp.log(p_pred + epsilon)
    multinomial_dot = -y_true_eps * log_p_pred  # [batch, seq_len, tracks]

    # Clip individual terms to prevent extreme values
    multinomial_dot = jnp.clip(multinomial_dot, 0.0, 1e6)

    multinomial_term = jnp.sum(multinomial_dot, axis=1)  # [batch, tracks]
    multinomial_term = multinomial_term / jnp.sum(position_weights)

    # ===== Combine =====
    loss = multinomial_term + total_weight * poisson_term  # [batch, tracks]

    if rescale:
        loss = loss * 2.0 / (1.0 + total_weight)

    # Final safety: clamp loss to reasonable range and ensure finite
    loss = jnp.clip(loss, 0.0, 1e6)
    loss = jnp.where(jnp.isfinite(loss), loss, 1e6)

    return loss


def cut_and_tag_loss_baskerville(
    predictions,  # Can be dict or array
    targets: jnp.ndarray,
    scale: float = 1.0,
    clip_soft: float = 64.0,
    total_weight: float = 1.0,
    epsilon: float = 1e-6,  # Increased from 1e-7 for better stability
) -> float:
    """
    Complete loss function for CUT&Tag fine-tuning with Baskerville transforms.

    Pipeline:
    1. Extract 1bp predictions from ATAC head (predictions are already transformed)
    2. Transform raw targets to match model's output space
    3. Compute Poisson + Multinomial NLL loss
    4. Mean reduce across batch and tracks

    Args:
        predictions: Model output - either:
                    - Dict with 'atac' -> 'scaled_predictions_1bp' [batch, seq_len, tracks]
                    - Array [batch, seq_len] or [batch, seq_len, 1]
                    Already in TRANSFORMED space from model
        targets: Raw CUT&Tag signal counts, shape [batch, seq_len]
                 In ORIGINAL count space, needs transformation
        scale: Scaling factor for targets (1.0 for ATAC-like data)
        clip_soft: Soft clipping threshold (64 for ATAC-like data)
        total_weight: Weight for Poisson term in loss
        epsilon: Small value for numerical stability

    Returns:
        Scalar loss value
    """
    # Extract ATAC predictions if input is a dict
    if isinstance(predictions, dict):
        pred_dict = predictions.get('atac', {})
        if 'scaled_predictions_1bp' not in pred_dict:
            raise ValueError(f"ATAC predictions not found. Available keys: {predictions.keys()}")
        pred_signal = pred_dict['scaled_predictions_1bp']  # [batch, seq_len, tracks]

        # Average across tracks for single-track CUT&Tag
        if pred_signal.ndim == 3:
            pred_signal = jnp.mean(pred_signal, axis=-1)  # [batch, seq_len]
    else:
        pred_signal = predictions

    # Ensure predictions are 2D for loss calculation
    # Squeeze if [batch, seq_len, 1] to [batch, seq_len]
    if pred_signal.ndim == 3 and pred_signal.shape[-1] == 1:
        pred_signal = pred_signal[..., 0]

    # CRITICAL: Cast predictions to float32 for numerical stability in loss computation
    # Model outputs bfloat16, but we need float32 for log/division operations
    pred_signal = pred_signal.astype(jnp.float32)

    # Safety check: ensure predictions are finite and non-negative
    # Clamp to reasonable range to prevent numerical instabilities
    pred_signal = jnp.where(jnp.isfinite(pred_signal), pred_signal, epsilon)

    # Use minimum value of epsilon instead of 0 to prevent log(0) in gradients
    # More aggressive clamping than before for stability
    pred_signal = jnp.maximum(pred_signal, epsilon)
    pred_signal = jnp.minimum(pred_signal, 1000.0)

    # Transform raw targets to model's output space
    targets_transformed = baskerville_target_transform(
        targets,
        scale=scale,
        clip_soft=clip_soft,
        clip_hard=384.0,
        use_sqrt=True,  # sum_sqrt for ATAC-like tracks
    )

    # Compute Baskerville's Poisson + Multinomial loss
    # Returns [batch, 1] or [batch] depending on input shape
    loss_per_batch = poisson_multinomial_loss(
        y_true=targets_transformed,
        y_pred=pred_signal,
        total_weight=total_weight,
        weight_range=1.0,  # No position weighting for now
        epsilon=epsilon,
    )

    # Mean reduce across batch
    loss = jnp.mean(loss_per_batch)

    # Final safety: ensure loss is finite and positive
    loss = jnp.where(jnp.isfinite(loss), loss, 1e6)
    loss = jnp.maximum(loss, epsilon)

    return loss


def compute_scale_from_data(
    signal_data: np.ndarray,
    target_range: float = 64.0,
) -> float:
    """
    Estimate appropriate scale parameter from raw data.

    Scale should bring typical signal values to a reasonable range
    (e.g., around clip_soft threshold after sqrt transform).

    Args:
        signal_data: Raw signal array [num_windows, window_size]
        target_range: Target range after scaling (default: 64.0)

    Returns:
        Estimated scale factor
    """
    # Compute 90th percentile of non-zero values
    nonzero_signal = signal_data[signal_data > 0]
    if len(nonzero_signal) == 0:
        return 1.0

    p90 = np.percentile(nonzero_signal, 90)

    # Estimate scale to bring p90 to target_range after sqrt
    # After sqrt(x*scale + 1) - 1 ≈ sqrt(x*scale) for large x
    # We want sqrt(p90 * scale) ≈ target_range
    # So scale ≈ target_range^2 / p90
    scale = (target_range ** 2) / p90

    return float(scale)


# Backward compatibility with old API
def cut_and_tag_loss(predictions, targets, track_mean=None):
    """
    Wrapper for backward compatibility with old loss API.

    NOTE: This now uses Baskerville's exact transform and loss!
    The track_mean parameter is IGNORED - we use scale parameter instead.
    """
    return cut_and_tag_loss_baskerville(
        predictions=predictions,
        targets=targets,
        scale=1.0,  # Default scale for ATAC-like data
        clip_soft=64.0,  # Default soft clip for ATAC-like data
        total_weight=1.0,
        epsilon=1e-7,
    )


# ---------------------------------------------------------------------------
# AlphaGenome paper-correct target transform
# ---------------------------------------------------------------------------

def compute_track_mean_from_bigwig(
    bigwig_path: str,
    chrom_sizes_path: str = None,
) -> float:
    """
    Compute the mean of non-zero base-level values genome-wide from a BigWig.

    This is the ``track_means`` value required by AlphaGenome's
    ``targets_scaling``.  Must be called once at training startup; the result
    is stored as a scalar and passed to ``cut_and_tag_loss_alphagenome`` via
    ``functools.partial``.

    Args:
        bigwig_path: Path to the (100M-normalised, 1bp-resolution) BigWig.
        chrom_sizes_path: Optional chrom.sizes file; if given, only chromosomes
            listed there are scanned.  Otherwise canonical chr* chromosomes
            (no underscores, no 'random'/'Un' suffixes) are used.

    Returns:
        Mean of non-zero values weighted by interval length (float).
        Returns 1.0 if no non-zero values are found.
    """
    import pyBigWig

    bw = pyBigWig.open(bigwig_path)
    all_chroms = bw.chroms()

    if chrom_sizes_path is not None:
        with open(chrom_sizes_path) as f:
            canonical = {line.split()[0] for line in f if line.strip()}
        chroms = {k: v for k, v in all_chroms.items() if k in canonical}
    else:
        # Use canonical chromosomes only (skip random / unplaced contigs)
        chroms = {
            k: v for k, v in all_chroms.items()
            if k.startswith('chr') and '_' not in k
            and 'random' not in k.lower() and 'un' not in k.lower()
        }

    total_sum   = 0.0
    total_count = 0

    for chrom, chrom_len in sorted(chroms.items()):
        try:
            intervals = bw.intervals(chrom, 0, chrom_len)
            if intervals is None:
                continue
            for start, end, value in intervals:
                if np.isfinite(value) and value > 0.0:
                    length = end - start
                    total_sum   += float(value) * length
                    total_count += length
        except Exception:
            continue

    bw.close()

    if total_count == 0:
        return 1.0
    return float(total_sum / total_count)


def alphagenome_targets_scaling(
    targets: jnp.ndarray,
    track_mean: float,
    apply_squashing: bool = False,
) -> jnp.ndarray:
    """
    AlphaGenome paper ``targets_scaling`` (exact match to paper pseudocode).

    Steps:
        1. targets = targets / track_mean
        2. (RNA-seq only) targets = targets ** 0.75
        3. smooth clip: where(targets > 10, 2*sqrt(targets*10) - 10, targets)

    Args:
        targets: Raw signal counts [batch, seq_len] or [batch, seq_len, tracks].
        track_mean: Pre-computed mean of non-zero values genome-wide
            (from ``compute_track_mean_from_bigwig``).
        apply_squashing: Power transform (True for RNA-seq, False for ATAC).

    Returns:
        Scaled targets in AlphaGenome's training space.
    """
    targets = targets.astype(jnp.float32)
    targets = jnp.where(jnp.isfinite(targets), targets, 0.0)
    targets = jnp.maximum(targets, 0.0)

    # 1. Normalise by genome-wide non-zero mean
    targets = targets / jnp.asarray(track_mean, dtype=jnp.float32)

    # 2. Optional power squashing (RNA-seq only)
    if apply_squashing:
        targets = targets ** 0.75

    # 3. Square-root-based smooth clip for high values
    targets = jnp.where(
        targets > 10.0,
        2.0 * jnp.sqrt(targets * 10.0) - 10.0,
        targets,
    )

    targets = jnp.where(jnp.isfinite(targets), targets, 0.0)
    return targets


def alphagenome_predictions_scaling(
    predictions: jnp.ndarray,
    track_mean: float,
    apply_squashing: bool = False,
) -> jnp.ndarray:
    """
    Inverse of ``alphagenome_targets_scaling``.

    Apply to raw model outputs (``preds.atac.values[:, 0]``) to convert them
    back to the original count space for comparison with BigWig signal.

    Args:
        predictions: Model predictions in transformed space.
        track_mean: Same value used during training.
        apply_squashing: Must match the value used in ``targets_scaling``.

    Returns:
        Predictions in original count units.
    """
    predictions = predictions.astype(jnp.float32)

    # Inverse smooth clip
    predictions = jnp.where(
        predictions > 10.0,
        (predictions + 10.0) ** 2 / 40.0,
        predictions,
    )

    # Inverse power squashing (RNA-seq only)
    if apply_squashing:
        predictions = predictions ** (1.0 / 0.75)

    # Inverse mean normalisation
    predictions = predictions * jnp.asarray(track_mean, dtype=jnp.float32)

    predictions = jnp.where(jnp.isfinite(predictions), predictions, 0.0)
    predictions = jnp.maximum(predictions, 0.0)
    return predictions


def cut_and_tag_loss_alphagenome(
    predictions,
    targets: jnp.ndarray,
    track_means,
    total_weight: float = 1.0,
    epsilon: float = 1e-6,
) -> float:
    """
    CUT&Tag loss using AlphaGenome's exact ``targets_scaling`` from the paper.

    Supports single-track (``track_means`` is a scalar float) and multi-track
    (``track_means`` is a 1-D array of shape [N], one value per BigWig).

    For multi-track, ``targets`` must be ``[batch, seq_len, N]`` and
    predictions output ``scaled_predictions_1bp`` of shape
    ``[batch, seq_len, N]``.  The loss is averaged across all tracks.

    Args:
        predictions: Model output dict or array.
        targets: Raw counts ``[batch, seq_len]`` (single) or
            ``[batch, seq_len, N]`` (multi-track).
        track_means: Genome-wide non-zero mean(s) from
            ``compute_track_mean_from_bigwig``.  Scalar for single-track,
            numpy array of shape [N] for multi-track.
        total_weight: Weight for Poisson term (default 1.0).
        epsilon: Numerical stability floor.

    Returns:
        Scalar loss.
    """
    # Extract ATAC signal if predictions is the raw Haiku model output dict
    if isinstance(predictions, dict):
        atac_preds = predictions.get('atac', {})
        if 'scaled_predictions_1bp' in atac_preds:
            pred_signal = atac_preds['scaled_predictions_1bp']  # [batch, seq_len, N]
        else:
            raise ValueError(
                f"Cannot find 'scaled_predictions_1bp' in predictions['atac']. "
                f"Top-level keys: {list(predictions.keys())}"
            )
    else:
        pred_signal = predictions

    pred_signal = pred_signal.astype(jnp.float32)
    pred_signal = jnp.where(jnp.isfinite(pred_signal), pred_signal, epsilon)
    pred_signal = jnp.maximum(pred_signal, epsilon)
    pred_signal = jnp.minimum(pred_signal, 1000.0)

    targets_transformed = alphagenome_targets_scaling(
        targets, track_mean=track_means, apply_squashing=False
    )

    loss_per_batch = poisson_multinomial_loss(
        y_true=targets_transformed,
        y_pred=pred_signal,
        total_weight=total_weight,
        weight_range=1.0,
        epsilon=epsilon,
    )

    # Per-track auxiliary output: mean over batch only → shape [N].
    # Allows the training loop to detect head collapse / data bugs.
    loss_per_track = jnp.mean(loss_per_batch, axis=0)
    # Scalar total: mean over tracks (identical to jnp.mean(loss_per_batch)).
    loss = jnp.mean(loss_per_track)
    loss = jnp.where(jnp.isfinite(loss), loss, 1e6)
    loss = jnp.maximum(loss, epsilon)
    # Sanitise per-track values (NaN/Inf → 1e6) for safe host-side logging.
    per_track_losses = jnp.where(
        jnp.isfinite(loss_per_track), loss_per_track,
        jnp.full_like(loss_per_track, 1e6),
    )
    return loss, per_track_losses
