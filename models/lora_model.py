"""
LoRA-wrapped AlphaGenome model for fine-tuning.

This module properly implements LoRA by recreating the Haiku transform
from the AlphaGenome model and injecting LoRA-modified parameters.
"""

import jax
import jax.numpy as jnp
import haiku as hk
import jmp
from typing import Dict, Tuple, Optional, Callable
from alphagenome_research.model import dna_model, model as alphagenome_model
import optax

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.lora_utils import (
    initialize_lora_params,
    apply_lora_to_params,
    count_lora_parameters
)


class LoRAAlphaGenomeModel:
    """
    LoRA-enabled AlphaGenome model for parameter-efficient fine-tuning.
    
    This class recreates the model's forward pass to enable LoRA injection.
    Uses JAX automatic parallelization with NamedSharding for multi-GPU training.
    """
    
    def __init__(
        self,
        base_model: dna_model.AlphaGenomeModel,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        target_layers: Optional[list] = None,
        learning_rate: float = 1e-4,
    ):
        """
        Initialize LoRA model.
        
        Args:
            base_model: Pre-trained AlphaGenome model
            lora_rank: Rank of LoRA matrices (default 8)
            lora_alpha: LoRA scaling factor (default 16)
            target_layers: List of layer paths to apply LoRA. If None, applies to all Q/K/V
            learning_rate: Learning rate for Adam optimizer
        """
        self.base_model = base_model
        self.base_params = base_model._params  # Frozen - will not be modified
        self.base_state = base_model._state
        self.device = base_model._device_context._device
        self.metadata = base_model._metadata
        self.num_devices = jax.device_count()
        
        # Recreate the forward function from dna_model.create()
        # This is the key to enabling LoRA
        self._setup_forward_function()
        
        # Initialize LoRA parameters
        if target_layers is None:
            from utils.lora_utils import get_attention_qkv_layers
            target_layers = get_attention_qkv_layers()
        
        self.lora_params = initialize_lora_params(
            self.base_params,
            target_layers=target_layers,
            rank=lora_rank,
            alpha=lora_alpha
        )
        
        # Setup optimizer with gradient clipping to prevent explosion
        # Chain: clip gradients -> apply Adam
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients to max norm of 1.0
            optax.adam(learning_rate)
        )
        trainable_params = {
            'lora_A': self.lora_params['lora_A'],
            'lora_B': self.lora_params['lora_B']
        }
        self.opt_state = self.optimizer.init(trainable_params)
        
        # Store metadata for access in train step
        self.lora_metadata = {
            'scaling': self.lora_params['scaling'],
            'rank': self.lora_params['rank'],
            'alpha': self.lora_params['alpha'],
            'target_layers': self.lora_params['target_layers']
        }
        
        # JIT-compile the train step (JAX will automatically parallelize based on input sharding)
        # Mark loss_fn as static since it's a Python function, not a JAX array
        self._train_step_jit = jax.jit(self._train_step, static_argnames=['loss_fn'])
        
        print(f"✓ LoRA model initialized")
        print(f"  Base model params: {self._count_params(self.base_params):,}")
        print(f"  LoRA params: {count_lora_parameters(self.lora_params):,}")
        print(f"  Trainable %: {100 * count_lora_parameters(self.lora_params) / self._count_params(self.base_params):.2f}%")
        if self.num_devices > 1:
            print(f"  Multi-GPU: {self.num_devices} devices (automatic parallelization)")
    
    def _setup_forward_function(self):
        """
        Recreate the forward function following dna_model.create() pattern.
        """
        jmp_policy = jmp.get_policy('params=float32,compute=bfloat16,output=bfloat16')
        
        @hk.transform_with_state
        def _forward(dna_sequence, organism_index):
            with hk.mixed_precision.push_policy(alphagenome_model.AlphaGenome, jmp_policy):
                return alphagenome_model.AlphaGenome(
                    self.metadata,
                    num_splice_sites=512,  # Default from model
                    splice_site_threshold=0.1,
                )(dna_sequence, organism_index)

        def _apply_fn(params, state, dna_sequence, organism_index):
            (predictions, _), _ = _forward.apply(
                params, state, None, dna_sequence, organism_index
            )
            return predictions
        
        self._forward_transform = _forward
        self.apply_fn = _apply_fn
    
    def _count_params(self, tree):
        """Count parameters in a PyTree."""
        arrays = jax.tree_util.tree_leaves(tree)
        return sum(x.size for x in arrays if hasattr(x, 'size'))
    
    def get_merged_params(self):
        """Get parameters with LoRA applied (for inference)."""
        return apply_lora_to_params(self.base_params, self.lora_params)
    
    def forward(self, dna_sequence, organism_index):
        """
        Forward pass with LoRA.
        
        Args:
            dna_sequence: DNA sequence tensor [batch, seq_len, 4]
            organism_index: Organism ID [batch]
            
        Returns:
            Model predictions dict
        """
        # Merge LoRA with base params
        merged_params = self.get_merged_params()
        
        # Run forward pass using the recreated apply function
        predictions = self.apply_fn(
            merged_params,
            self.base_state,
            dna_sequence,
            organism_index
        )
        
        return predictions
    
    def _train_step(
        self,
        dna_sequence: jnp.ndarray,
        organism_index: jnp.ndarray,
        targets: Dict,
        lora_trainable: Dict,  # Only lora_A and lora_B
        opt_state: optax.OptState,
        loss_fn: Callable,
    ) -> Tuple[float, Dict, Dict, optax.OptState]:
        """
        Training step with automatic parallelization.
        
        JAX automatically parallelizes computation based on input sharding.
        When inputs are sharded across devices, gradients are automatically
        averaged across devices during the backward pass.
        
        Args:
            dna_sequence: Input DNA [batch, seq_len, 4] (possibly sharded across devices)
            organism_index: Organism IDs [batch] (possibly sharded)
            targets: Ground truth labels (dict with target signals, possibly sharded)
            lora_trainable: Dict with lora_A and lora_B (trainable parameters)
            opt_state: Optimizer state
            loss_fn: Loss function(predictions, targets) -> scalar
            
        Returns:
            (loss_value, predictions, updated_lora_trainable, updated_opt_state, grad_norm)
        """
        def compute_loss(trainable_lora_params):
            # Reconstruct full lora_params with trainable parts + metadata
            full_lora_params = {
                'lora_A': trainable_lora_params['lora_A'],
                'lora_B': trainable_lora_params['lora_B'],
                'scaling': self.lora_metadata['scaling'],
                'rank': self.lora_metadata['rank'],
                'alpha': self.lora_metadata['alpha'],
                'target_layers': self.lora_metadata['target_layers']
            }
            
            # Merge LoRA params with base params
            merged_params = apply_lora_to_params(self.base_params, full_lora_params)
            
            # Forward pass
            predictions = self.apply_fn(
                merged_params,
                self.base_state,
                dna_sequence,
                organism_index
            )
            
            # Compute loss  
            loss = loss_fn(predictions, targets)
            return loss, predictions
        
        # Compute loss and gradients
        # JAX automatically averages gradients across devices when inputs are sharded
        (loss_value, predictions), grads = jax.value_and_grad(
            compute_loss, has_aux=True
        )(lora_trainable)
        
        # Compute gradient norm for monitoring
        grad_norm = optax.global_norm(grads)
        
        # Clip gradients to prevent explosion (max global norm = 1.0)
        max_norm = 1.0
        scale = jnp.minimum(1.0, max_norm / (grad_norm + 1e-7))
        grads = jax.tree.map(lambda g: g * scale, grads)
        
        # Update LoRA parameters
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        updated_trainable = optax.apply_updates(lora_trainable, updates)
        
        return loss_value, predictions, updated_trainable, new_opt_state, grad_norm
    
    def train_step(
        self,
        dna_sequence: jnp.ndarray,
        organism_index: jnp.ndarray,
        targets: Dict,
        loss_fn: Callable
    ) -> Tuple[float, Dict]:
        """
        Single training step with automatic multi-GPU parallelization.
        
        JAX automatically parallelizes based on input sharding. If inputs are
        sharded across devices, computation is distributed and gradients are
        averaged automatically.
        
        Args:
            dna_sequence: Input DNA [batch, seq_len, 4] (possibly sharded)
            organism_index: Organism IDs [batch] (possibly sharded)
            targets: Ground truth labels (dict, possibly sharded)
            loss_fn: Loss function(predictions, targets) -> scalar
            
        Returns:
            (loss_value, predictions)
        """
        lora_trainable = {
            'lora_A': self.lora_params['lora_A'],
            'lora_B': self.lora_params['lora_B']
        }
        
        # JIT-compiled training step with automatic parallelization
        loss_value, predictions, updated_lora_trainable, new_opt_state, grad_norm = self._train_step_jit(
            dna_sequence,
            organism_index,
            targets,
            lora_trainable,
            self.opt_state,
            loss_fn
        )
        
        # Update internal state
        self.lora_params['lora_A'] = updated_lora_trainable['lora_A']
        self.lora_params['lora_B'] = updated_lora_trainable['lora_B']
        self.opt_state = new_opt_state
        
        return loss_value, predictions
    
    def save_lora(self, filepath: str):
        """Save LoRA parameters."""
        from utils.lora_utils import save_lora_checkpoint
        save_lora_checkpoint(self.lora_params, filepath)
    
    def load_lora(self, filepath: str):
        """Load LoRA parameters."""
        from utils.lora_utils import load_lora_checkpoint
        self.lora_params = load_lora_checkpoint(filepath)
        trainable_params = {
            'lora_A': self.lora_params['lora_A'],
            'lora_B': self.lora_params['lora_B']
        }
        self.opt_state = self.optimizer.init(trainable_params)
