"""
Integrated AlphaGenome Architecture Inspector & LoRA Layer Exporter

This script:
1. Loads AlphaGenome pre-trained model
2. Inspects parameter structure
3. Identifies all linear layers suitable for LoRA
4. Exports detailed analysis to JSON
5. Calculates LoRA parameter savings
6. Provides fine-tuning recommendations
"""

import jax
import jax.numpy as jnp
import json
from pathlib import Path
from alphagenome_research.model import dna_model


def count_params(tree):
    """Count total number of parameters in a PyTree."""
    arrays = jax.tree_util.tree_leaves(tree)
    return sum(x.size for x in arrays if hasattr(x, 'size'))


def inspect_pytree(tree, prefix="", max_depth=2, current_depth=0):
    """Recursively print PyTree structure with shapes."""
    if current_depth >= max_depth:
        return
    
    if isinstance(tree, dict):
        for key in sorted(tree.keys()):
            value = tree[key]
            print(f"{prefix}{key}/")
            inspect_pytree(value, prefix + "  ", max_depth, current_depth + 1)
    elif hasattr(tree, 'shape'):
        print(f"{prefix}→ shape={tree.shape}, dtype={tree.dtype}")


def find_linear_layers(tree, path=""):
    """Find all 2D weight matrices (linear layers)."""
    layers = []
    
    if isinstance(tree, dict):
        for key, value in tree.items():
            current_path = f"{path}/{key}" if path else key
            
            if hasattr(value, 'shape') and len(value.shape) == 2:
                layers.append({
                    'path': current_path,
                    'shape': list(value.shape),
                    'params': int(value.size)
                })
            
            layers.extend(find_linear_layers(value, current_path))
    
    return layers


def group_layers_by_type(linear_layers):
    """Group linear layers by type (attention, MLP, conv, etc)."""
    attention_qkv = [l for l in linear_layers if any(x in l['path'] for x in ['q_layer', 'k_layer', 'v_layer'])]
    mlp_layers = [l for l in linear_layers if 'mlp_block' in l['path'] and '/w' in l['path']]
    conv_layers = [l for l in linear_layers if 'conv' in l['path'] and '/w' in l['path']]
    other_layers = [l for l in linear_layers if l not in attention_qkv + mlp_layers + conv_layers]
    
    return {
        'attention_qkv': attention_qkv,
        'mlp': mlp_layers,
        'conv': conv_layers,
        'other': other_layers
    }


def calculate_lora_params(layers, rank):
    """Calculate LoRA parameter count for given rank."""
    return sum((l['shape'][0] + l['shape'][1]) * rank for l in layers)


def main():
    # Load the model
    print("="*60)
    print("LOADING ALPHAGENOME MODEL")
    print("="*60)
    
    checkpoint_path = "/home/ray/AG_Finetune/checkpoints/pre_trained/alphagenome-all-folds"
    ag_model = dna_model.create(
        checkpoint_path=checkpoint_path,
        organism_settings=None,
        device=None,
    )
    
    print("✓ Model loaded successfully!")
    print(f"Device: {ag_model._device_context._device}")
    
    # Access parameters
    params = ag_model._params
    state = ag_model._state
    
    # Count parameters
    print("\n" + "="*60)
    print("MODEL SIZE")
    print("="*60)
    
    total_params = count_params(params)
    total_state = count_params(state)
    
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Total state parameters: {total_state:,}")
    print(f"Total: {total_params + total_state:,}")
    print(f"Approx size: {(total_params + total_state) * 4 / (1024**3):.2f} GB")
    
    # Inspect parameter structure
    print("\n" + "="*60)
    print("PARAMETER STRUCTURE")
    print("="*60)
    
    inspect_pytree(params)
    
    # Find all linear layers
    print("\n" + "="*60)
    print("LINEAR LAYERS (LoRA Candidates)")
    print("="*60)
    
    linear_layers = find_linear_layers(params)
    print(f"Found {len(linear_layers)} linear layers\n")
    
    # Show first 15
    for i, layer in enumerate(linear_layers[:15]):
        print(f"{i+1:3d}. {layer['path']}")
        print(f"     Shape: {layer['shape']}, Params: {layer['params']:,}")
    
    if len(linear_layers) > 15:
        print(f"\n... and {len(linear_layers) - 15} more layers")
    
    # Group layers by type
    print("\n" + "="*60)
    print("LAYER GROUPS")
    print("="*60)
    
    grouped = group_layers_by_type(linear_layers)
    
    print(f"\nAttention Q/K/V layers: {len(grouped['attention_qkv'])}")
    print(f"MLP layers: {len(grouped['mlp'])}")
    print(f"Conv layers: {len(grouped['conv'])}")
    print(f"Other layers: {len(grouped['other'])}")
    
    print("\n--- Sample Attention Layers (best for LoRA) ---")
    for layer in grouped['attention_qkv'][:6]:
        print(f"{layer['path']}: {layer['shape']}")
    
    print("\n--- Sample MLP Layers ---")
    for layer in grouped['mlp'][:6]:
        print(f"{layer['path']}: {layer['shape']}")
    
    # Calculate LoRA parameter savings
    print("\n" + "="*60)
    print("LORA PARAMETER ESTIMATE")
    print("="*60)
    
    total_attn_params = sum(l['params'] for l in grouped['attention_qkv'])
    
    for rank in [4, 8, 16, 32]:
        lora_params = calculate_lora_params(grouped['attention_qkv'], rank)
        reduction = (1 - lora_params/total_attn_params) * 100
        print(f"Rank {rank:2d}: {lora_params:,} params ({reduction:.1f}% reduction)")
        print(f"         vs original {total_attn_params:,} params")
    
    # Export to JSON
    print("\n" + "="*60)
    print("EXPORTING ANALYSIS")
    print("="*60)
    
    output_file = Path("/home/ray/AG_Finetune/exports/linear_layers_analysis.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    analysis = {
        'total_params': int(total_params),
        'total_state': int(total_state),
        'num_linear_layers': len(linear_layers),
        'linear_layers': linear_layers,
        'grouped_layers': {
            'attention_qkv': grouped['attention_qkv'],
            'mlp': grouped['mlp'],
            'conv': grouped['conv'],
            'other': grouped['other']
        },
        'lora_estimates': {
            f'rank_{rank}': {
                'params': calculate_lora_params(grouped['attention_qkv'], rank),
                'reduction_pct': round((1 - calculate_lora_params(grouped['attention_qkv'], rank)/total_attn_params) * 100, 1)
            } for rank in [4, 8, 16, 32]
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"✓ Saved full analysis to: {output_file}")
    
    # Final recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("""
Key observations:
1. Model has 450M trainable parameters
2. {} attention Q/K/V layers (best LoRA targets)  
3. {} MLP layers (optional LoRA targets)
4. Transformer tower has 9 blocks with multi-head attention

Recommended approach:
- Start with LoRA on attention Q/K/V only (rank=8)
- This reduces trainable params by ~99%
- Can expand to MLP layers if needed

Next steps:
1. Implement LoRA module for JAX/Haiku
2. Create training loop with frozen base model
3. Fine-tune on your custom dataset
""".format(len(grouped['attention_qkv']), len(grouped['mlp'])))
    
    print("="*60)
    print("✓ Inspection complete!")
    print("="*60)


if __name__ == "__main__":
    main()
