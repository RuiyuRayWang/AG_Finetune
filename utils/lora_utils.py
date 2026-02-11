"""
LoRA (Low-Rank Adaptation) utilities for AlphaGenome fine-tuning.

This module provides functions to:
1. Initialize LoRA parameters
2. Inject LoRA into existing model parameters
3. Extract and merge LoRA weights
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Any
import json


def initialize_lora_params(
    base_params: Dict,
    target_layers: List[str],
    rank: int = 8,
    alpha: float = 16.0,
    seed: int = 42
) -> Dict:
    """
    Initialize LoRA parameters for specified layers.
    
    LoRA decomposes weight updates as: ΔW = B @ A
    where A is [d_in, rank] and B is [rank, d_out]
    
    Args:
        base_params: Original model parameters (frozen)
        target_layers: List of layer paths to apply LoRA (e.g., 'alphagenome/transformer_tower/mha_block/q_layer/w')
        rank: LoRA rank (default 8)
        alpha: LoRA scaling factor (default 16)
        seed: Random seed for initialization
        
    Returns:
        Dictionary containing LoRA parameters with structure:
        {
            'lora_A': {layer_path: array[d_in, rank]},
            'lora_B': {layer_path: array[rank, d_out]},
            'scaling': alpha / rank
        }
    """
    rng = jax.random.PRNGKey(seed)
    lora_A = {}
    lora_B = {}
    
    for layer_path in target_layers:
        # Haiku params use flat keys like 'alphagenome/transformer_tower/mha_block/q_layer'
        # Then within that dict is 'w' for the weight matrix
        if layer_path not in base_params:
            raise KeyError(f"Layer {layer_path} not found in base_params. Available keys: {list(base_params.keys())[:10]}...")
        
        layer_dict = base_params[layer_path]
        if 'w' not in layer_dict:
            raise KeyError(f"Weight 'w' not found in {layer_path}. Available keys: {list(layer_dict.keys())}")
        
        weight = layer_dict['w']
        weight_shape = weight.shape
        if len(weight_shape) != 2:
            raise ValueError(f"Layer {layer_path} has shape {weight_shape}, expected 2D matrix")
        
        d_in, d_out = weight_shape
        
        # Initialize A with random values (Kaiming uniform)
        rng, key_a = jax.random.split(rng)
        lora_A[layer_path] = jax.random.normal(key_a, (d_in, rank)) * jnp.sqrt(1.0 / d_in)
        
        # Initialize B with zeros (so LoRA starts as identity)
        lora_B[layer_path] = jnp.zeros((rank, d_out))
    
    return {
        'lora_A': lora_A,
        'lora_B': lora_B,
        'scaling': alpha / rank,
        'rank': rank,
        'alpha': alpha,
        'target_layers': target_layers
    }


def apply_lora_to_params(
    base_params: Dict,
    lora_params: Dict
) -> Dict:
    """
    Create a new parameter PyTree with LoRA applied.
    
    For each target layer: W_new = W_frozen + (B @ A) * scaling
    
    Args:
        base_params: Original frozen parameters
        lora_params: LoRA parameters from initialize_lora_params
        
    Returns:
        New parameter PyTree with LoRA deltas added
    """
    # Deep copy the base params
    merged_params = jax.tree_util.tree_map(lambda x: x, base_params)
    
    lora_A = lora_params['lora_A']
    lora_B = lora_params['lora_B']
    scaling = lora_params['scaling']
    
    for layer_path in lora_params['target_layers']:
        # Haiku uses flat keys, then 'w' for the weight
        if layer_path not in merged_params:
            raise KeyError(f"Layer {layer_path} not found in merged_params")
        
        # Apply LoRA: W_new = W + A @ B * scaling
        # A is [d_in, rank], B is [rank, d_out]
        # A @ B = [d_in, d_out] which matches W
        base_weight = merged_params[layer_path]['w']
        lora_delta = lora_A[layer_path] @ lora_B[layer_path] * scaling
        merged_params[layer_path]['w'] = base_weight + lora_delta
    
    return merged_params


def extract_lora_gradients(
    full_grads: Dict,
    lora_params: Dict
) -> Dict:
    """
    Extract only LoRA parameter gradients from full gradient tree.
    
    Args:
        full_grads: Gradients for all parameters
        lora_params: LoRA parameters
        
    Returns:
        Dictionary with gradients for lora_A and lora_B only
    """
    lora_A_grads = {}
    lora_B_grads = {}
    
    for layer_path in lora_params['target_layers']:
        keys = layer_path.split('/')
        
        # Navigate to the gradient
        current_grad = full_grads
        for key in keys:
            current_grad = current_grad[key]
        
        # Get base weight
        keys = layer_path.split('/')
        current_base = None  # We'd need base params here for proper calculation
        
        # For now, return the full gradient
        # In practice, we'd project it onto LoRA subspace
        lora_A_grads[layer_path] = current_grad
        lora_B_grads[layer_path] = current_grad
    
    return {
        'lora_A': lora_A_grads,
        'lora_B': lora_B_grads
    }


def get_attention_qkv_layers(linear_layers_json: str = None) -> List[str]:
    """
    Get list of attention Q/K/V layer paths from the model.
    
    Args:
        linear_layers_json: Path to linear_layers_analysis.json (optional)
        
    Returns:
        List of layer paths for Q/K/V projections (without '/w' suffix)
    """
    if linear_layers_json:
        with open(linear_layers_json, 'r') as f:
            data = json.load(f)
            linear_layers = data['linear_layers']
            # Extract paths and remove '/w' suffix
            qkv_layers = [
                layer['path'].replace('/w', '') if layer['path'].endswith('/w') else layer['path']
                for layer in linear_layers
                if any(x in layer['path'] for x in ['q_layer', 'k_layer', 'v_layer'])
                and 'mha_block' in layer['path']  # Only attention layers, not other q/k/v
            ]
    else:
        # Default attention Q/K/V layers (without '/w' suffix)
        qkv_layers = []
        for i in range(9):  # 9 transformer blocks
            if i == 0:
                qkv_layers.extend([
                    "alphagenome/transformer_tower/mha_block/q_layer",
                    "alphagenome/transformer_tower/mha_block/k_layer",
                    "alphagenome/transformer_tower/mha_block/v_layer",
                ])
            else:
                qkv_layers.extend([
                    f"alphagenome/transformer_tower/mha_block_{i}/q_layer",
                    f"alphagenome/transformer_tower/mha_block_{i}/k_layer",
                    f"alphagenome/transformer_tower/mha_block_{i}/v_layer",
                ])
    
    return qkv_layers


def count_lora_parameters(lora_params: Dict) -> int:
    """Count total number of trainable LoRA parameters."""
    total = 0
    for layer_path in lora_params['target_layers']:
        A_params = lora_params['lora_A'][layer_path].size
        B_params = lora_params['lora_B'][layer_path].size
        total += A_params + B_params
    return total


def save_lora_checkpoint(lora_params: Dict, filepath: str):
    """Save LoRA parameters to file."""
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(lora_params, f)
    print(f"✓ Saved LoRA checkpoint to {filepath}")


def load_lora_checkpoint(filepath: str) -> Dict:
    """Load LoRA parameters from file."""
    import pickle
    with open(filepath, 'rb') as f:
        lora_params = pickle.load(f)
    print(f"✓ Loaded LoRA checkpoint from {filepath}")
    return lora_params
