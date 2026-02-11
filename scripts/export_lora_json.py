import jax
import jax.numpy as jnp
import json
from alphagenome_research.model import dna_model

# Load the model
checkpoint_path = "/home/ray/AG_Finetune/checkpoints/pre_trained/alphagenome-all-folds"
ag_model = dna_model.create(
    checkpoint_path=checkpoint_path,
    organism_settings=None,
    device=None,
)

print("✓ Model loaded successfully!")
print(f"Device: {ag_model._device_context._device}")

params = ag_model._params
state = ag_model._state

def count_params(tree):
    arrays = jax.tree_util.tree_leaves(tree)
    return sum(x.size for x in arrays if hasattr(x, 'size'))

def find_linear_layers(tree, path=""):
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

linear_layers = find_linear_layers(params)

# Save full list to JSON
output_file = "/home/ray/AG_Finetune/exports/linear_layers_analysis.json"
with open(output_file, 'w') as f:
    json.dump({
        'total_params': int(count_params(params)),
        'total_state': int(count_params(state)),
        'num_linear_layers': len(linear_layers),
        'linear_layers': linear_layers
    }, f, indent=2)

print(f"\n✓ Saved full analysis to: {output_file}")

# Identify key layers for LoRA
print("\n" + "="*60)
print("RECOMMENDED LORA TARGETS")
print("="*60)

# Group layers by type
attention_qkv = [l for l in linear_layers if any(x in l['path'] for x in ['q_layer', 'k_layer', 'v_layer'])]
mlp_layers = [l for l in linear_layers if 'mlp_block' in l['path'] and '/w' in l['path']]
conv_layers = [l for l in linear_layers if 'conv' in l['path'] and '/w' in l['path']]

print(f"\nAttention Q/K/V layers: {len(attention_qkv)}")
print(f"MLP layers: {len(mlp_layers)}")
print(f"Conv layers: {len(conv_layers)}")

print("\n--- Sample Attention Layers (best for LoRA) ---")
for i, layer in enumerate(attention_qkv[:6]):
    print(f"{layer['path']}: {layer['shape']}")

print("\n--- Sample MLP Layers ---")
for i, layer in enumerate(mlp_layers[:6]):
    print(f"{layer['path']}: {layer['shape']}")

# Calculate LoRA parameter savings
print("\n" + "="*60)
print("LORA PARAMETER ESTIMATE")
print("="*60)

total_attn_params = sum(l['params'] for l in attention_qkv)

for rank in [4, 8, 16, 32]:
    lora_params = sum((l['shape'][0] + l['shape'][1]) * rank for l in attention_qkv)
    reduction = (1 - lora_params/total_attn_params) * 100
    print(f"Rank {rank:2d}: {lora_params:,} params ({reduction:.1f}% reduction)")
    print(f"         vs original {total_attn_params:,} params")

print("\n" + "="*60)
print("NEXT STEP: Implement LoRA Fine-tuning")
print("="*60)
print("""
Key observations:
1. Model has 450M trainable parameters
2. 27 attention Q/K/V layers (best LoRA targets)  
3. 36 MLP layers (optional LoRA targets)
4. Transformer tower has 9 blocks with multi-head attention

Recommended approach:
- Start with LoRA on attention Q/K/V only (rank=8)
- This reduces trainable params by ~99%
- Can expand to MLP layers if needed
""")