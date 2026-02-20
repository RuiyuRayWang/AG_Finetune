"""
Simplified loss function for AlphaGenome fine-tuning on CUT&Tag data.

Instead of trying to match Baskerville's complex transform pipeline, we use
a simpler approach that respects AlphaGenome's actual output format:
- Predictions come from ATAC head: softplus + learned scaling
- Targets are raw counts
- Use Poisson NLL + Multinomial NLL on SQRT-TRANSFORMED space for both
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional


def safe_sqrt_transform(x: jnp.ndarray, epsilon: float = 1e-7) -> jnp.ndarray:
    """Square root transform with numerical safety."""
    x_safe = jnp.maximum(x, 0.0)
    return jnp.sqrt(x_safe + epsilon)


def poisson_nll(y_true: jnp.ndarray, y_pred: jnp.ndarray, epsilon: float = 1e-7) -> jnp.ndarray:
    """
    Poisson negative log-likelihood with safety checks.
    
    NLL = y_pred - y_true * log(y_pred)
    """
    y_pred_safe = jnp.maximum(y_pred, epsilon)
    nll = y_pred_safe - y_true * jnp.log(y_pred_safe + epsilon)
    return jnp.clip(nll, 0.0, 1e6)


def multinomial_nll(y_true: jnp.ndarray, y_pred: jnp.ndarray, epsilon: float = 1e-7) -> jnp.ndarray:
    """
    Multinomial negative log-likelihood over sequence positions.
    
    Normalizes both to distributions and computes -sum(y_true * log(p_pred))
    """
    # Sum across sequence to get totals
    sum_true = jnp.sum(y_true, axis=1, keepdims=True) + epsilon
    sum_pred = jnp.sum(y_pred, axis=1, keepdims=True) + epsilon
    
    # Normalize to probabilities
    p_true = y_true / sum_true
    p_pred = (y_pred + epsilon) / sum_pred
    
    # Clip probabilities
    p_pred = jnp.clip(p_pred, epsilon, 1.0 - epsilon)
    
    # Multinomial NLL
    nll = -jnp.sum(p_true * jnp.log(p_pred + epsilon), axis=1)
    return jnp.clip(nll, 0.0, 1e6)


def cut_and_tag_loss_simple(
    predictions,
    targets: jnp.ndarray,
    poisson_weight: float = 1.0,
    multinomial_weight: float = 1.0,
    epsilon: float = 1e-7,
) -> float:
    """
    Simplified loss for CUT&Tag fine-tuning.
    
    Strategy:
    1. Extract ATAC 1bp predictions (already softplus + scaled)
    2. Apply sqrt transform to BOTH predictions and targets for stability
    3. Compute Poisson NLL on sqrt-transformed values
    4. Compute Multinomial NLL on sqrt-transformed distributions
    5. Combine with weights
    
    Args:
        predictions: Model output dict with ATAC predictions
        targets: Raw CUT&Tag counts [batch, seq_len]
        poisson_weight: Weight for Poisson term (total count matching)
        multinomial_weight: Weight for Multinomial term (positional distribution)
        epsilon: Small value for numerical stability
        
    Returns:
        Scalar loss value
    """
    # Extract ATAC predictions
    if isinstance(predictions, dict):
        pred_dict = predictions.get('atac', {})
        if 'scaled_predictions_1bp' not in pred_dict:
            raise ValueError(f"ATAC predictions not found. Available keys: {predictions.keys()}")
        pred_signal = pred_dict['scaled_predictions_1bp']  # [batch, seq_len, tracks]
        
        # Average across tracks
        if pred_signal.ndim == 3:
            pred_signal = jnp.mean(pred_signal, axis=-1)  # [batch, seq_len]
    else:
        pred_signal = predictions
        if pred_signal.ndim == 3 and pred_signal.shape[-1] == 1:
            pred_signal = pred_signal[..., 0]
    
    # Safety: ensure predictions are finite and non-negative
    pred_signal = jnp.where(jnp.isfinite(pred_signal), pred_signal, epsilon)
    pred_signal = jnp.maximum(pred_signal, 0.0)
    
    # Safety: ensure targets are non-negative
    targets = jnp.maximum(targets, 0.0)
    
    # Apply sqrt transform to both (for numerical stability)
    # This brings both to similar scales and reduces impact of extreme values
    pred_sqrt = safe_sqrt_transform(pred_signal, epsilon)
    targets_sqrt = safe_sqrt_transform(targets, epsilon)
    
    # Compute Poisson NLL on sqrt-transformed space
    # This measures how well total counts match
    poisson_loss = poisson_nll(targets_sqrt, pred_sqrt, epsilon)
    poisson_term = jnp.mean(poisson_loss)
    
    # Compute Multinomial NLL on sqrt-transformed space
    # This measures how well the positional distribution matches
    multinomial_loss = multinomial_nll(targets_sqrt, pred_sqrt, epsilon)
    multinomial_term = jnp.mean(multinomial_loss)
    
    # Combine terms
    total_loss = poisson_weight * poisson_term + multinomial_weight * multinomial_term
    
    # Final safety
    total_loss = jnp.where(jnp.isfinite(total_loss), total_loss, 1e6)
    total_loss = jnp.maximum(total_loss, epsilon)
    total_loss = jnp.minimum(total_loss, 1e6)
    
    return total_loss


def cut_and_tag_loss_mse(
    predictions,
    targets: jnp.ndarray,
    epsilon: float = 1e-7,
) -> float:
    """
    Even simpler MSE loss on sqrt-transformed space.
    
    This is the most stable option for initial training.
    """
    # Extract ATAC predictions
    if isinstance(predictions, dict):
        pred_dict = predictions.get('atac', {})
        pred_signal = pred_dict['scaled_predictions_1bp']
        if pred_signal.ndim == 3:
            pred_signal = jnp.mean(pred_signal, axis=-1)
    else:
        pred_signal = predictions
        if pred_signal.ndim == 3:
            pred_signal = pred_signal[..., 0]
    
    # Safety
    pred_signal = jnp.where(jnp.isfinite(pred_signal), pred_signal, epsilon)
    pred_signal = jnp.maximum(pred_signal, 0.0)
    targets = jnp.maximum(targets, 0.0)
    
    # Sqrt transform both
    pred_sqrt = safe_sqrt_transform(pred_signal, epsilon)
    targets_sqrt = safe_sqrt_transform(targets, epsilon)
    
    # MSE
    mse = jnp.mean((pred_sqrt - targets_sqrt) ** 2)
    
    return jnp.maximum(mse, epsilon)
