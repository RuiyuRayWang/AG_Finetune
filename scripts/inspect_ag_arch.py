import jax
import jax.numpy as jnp
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

# Access the parameters directly from the loaded model
params = ag_model._params
state = ag_model._state

print("\n" + "="*60)
print("MODEL SIZE")
print("="*60)

def count_params(tree):
    """Count total number of parameters in a PyTree."""
    arrays = jax.tree_util.tree_leaves(tree)
    return sum(x.size for x in arrays if hasattr(x, 'size'))

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

inspect_pytree(params)

# Find linear layers for LoRA
print("\n" + "="*60)
print("LINEAR LAYERS (LoRA Candidates)")
print("="*60)

def find_linear_layers(tree, path=""):
    """Find all 2D weight matrices (linear layers)."""
    layers = []
    
    if isinstance(tree, dict):
        for key, value in tree.items():
            current_path = f"{path}/{key}" if path else key
            
            if hasattr(value, 'shape') and len(value.shape) == 2:
                layers.append({
                    'path': current_path,
                    'shape': value.shape,
                    'params': value.size
                })
            
            layers.extend(find_linear_layers(value, current_path))
    
    return layers

linear_layers = find_linear_layers(params)
print(f"Found {len(linear_layers)} linear layers\n")

# Show first 15
for i, layer in enumerate(linear_layers[:15]):
    print(f"{i+1:3d}. {layer['path']}")
    print(f"     Shape: {layer['shape']}, Params: {layer['params']:,}")

if len(linear_layers) > 15:
    print(f"\n... and {len(linear_layers) - 15} more layers")

print("\n✓ Inspection complete! Ready for LoRA implementation.")