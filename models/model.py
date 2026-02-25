"""
LoRA-wrapped AlphaGenome model for fine-tuning.

This module properly implements LoRA by recreating the Haiku transform
from the AlphaGenome model and injecting LoRA-modified parameters.
"""

import jax
import jax.numpy as jnp
import haiku as hk
import jmp
import numpy as np
from typing import Dict, Tuple, Optional, Callable
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from alphagenome_research.model import dna_model, model as alphagenome_model
from alphagenome_research.model.metadata import metadata as metadata_lib
import optax
import orbax.checkpoint as ocp

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
        mesh: Optional[Mesh] = None,
        warmup_steps: int = 0,
        total_steps: int = 0,
        precision_policy: str = 'params=float32,compute=float32,output=float32',
    ):
        """
        Initialize LoRA model.
        
        Args:
            base_model: Pre-trained AlphaGenome model
            lora_rank: Rank of LoRA matrices (default 8)
            lora_alpha: LoRA scaling factor (default 16)
            target_layers: List of layer paths to apply LoRA. If None, applies to all Q/K/V
            learning_rate: Learning rate for Adam optimizer (peak LR if using warmup)
            mesh: Optional JAX device mesh for multi-GPU training
            warmup_steps: Number of warmup steps (0 = no warmup)
            total_steps: Total training steps for cosine decay (0 = constant LR)
            precision_policy: JMP mixed-precision policy string.
                             Default: float32 everywhere (safe). Use
                             'params=float32,compute=bfloat16,output=bfloat16'
                             only if bf16 NaN issue is confirmed fixed.
        """
        self.base_model = base_model
        self.base_params = base_model._params  # Frozen - will not be modified
        self.base_state = base_model._state
        self.device = base_model._device_context._device
        self.metadata = base_model._metadata
        self.num_devices = jax.device_count()
        self.mesh = mesh
        self.precision_policy = precision_policy
        
        # Setup sharding for parameter replication (data parallelism)
        if self.mesh is not None:
            # Replicate parameters across all devices (each device gets full copy)
            self.param_sharding = NamedSharding(self.mesh, P())
            print(f"  Setting up parameter replication across {self.num_devices} devices...")
        else:
            self.param_sharding = None
        
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
        
        # Replicate parameters across devices for data parallelism
        if self.param_sharding is not None:
            print(f"  Replicating base parameters across devices...")
            self.base_params = jax.device_put(self.base_params, self.param_sharding)
            print(f"  Replicating base state (BatchNorm, etc.) across devices...")
            self.base_state = jax.device_put(self.base_state, self.param_sharding)
            print(f"  Replicating LoRA parameters across devices...")
            self.lora_params['lora_A'] = jax.device_put(self.lora_params['lora_A'], self.param_sharding)
            self.lora_params['lora_B'] = jax.device_put(self.lora_params['lora_B'], self.param_sharding)
        
        # Setup optimizer with LR schedule + gradient clipping
        # Use warmup + cosine decay to stabilize sparse signal training
        if warmup_steps > 0 and total_steps > 0:
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=total_steps,
                end_value=learning_rate * 0.1  # Decay to 10% of peak
            )
            print(f"  Using LR warmup: {warmup_steps} steps, cosine decay over {total_steps} steps")
        else:
            lr_schedule = learning_rate
        
        # Chain: clip gradients -> apply Adam with LR schedule
        # Matching DeepMind's fine-tuning colab: clip_by_global_norm(0.5)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(lr_schedule)
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
        
        # JIT-compile the train step with simplified sharding (official DeepMind approach)
        # Use P('data') for batch dimension only, let JAX auto-broadcast to input shapes
        if mesh is not None:
            # Multi-GPU: simplified sharding specs
            data_sharding = NamedSharding(mesh, P('data'))
            replicated = self.param_sharding
            
            self._train_step_jit = jax.jit(
                self._train_step,
                static_argnames=['loss_fn'],
                in_shardings=(
                    data_sharding,       # dna_sequence (auto-broadcasts to [B,S,4])
                    data_sharding,       # organism_index (auto-broadcasts to [B])
                    data_sharding,       # targets (auto-broadcasts to shape)
                    {'lora_A': replicated, 'lora_B': replicated},  # trainable_params dict
                    replicated,          # base_params
                    replicated,          # base_state
                    replicated,          # opt_state
                ),
                out_shardings=(
                    replicated,          # loss_value (scalar, replicated)
                    replicated,          # predictions
                    {'lora_A': replicated, 'lora_B': replicated},  # new_trainable_params
                    replicated,          # new_opt_state
                    replicated,          # grad_norm
                )
            )
        else:
            # Single-GPU: no sharding needed
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
        jmp_policy = jmp.get_policy(self.precision_policy)
        
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
            
            # Extract signal from targets dict
            targets_signal = targets['signal']
            
            # Compute loss (loss function handles float32 conversion and all safety checks)
            loss = loss_fn(predictions, targets_signal)
            
            return loss, predictions
        
        # Compute loss and gradients
        # JAX automatically averages gradients across devices when inputs are sharded
        (loss_value, predictions), grads = jax.value_and_grad(
            compute_loss, has_aux=True
        )(lora_trainable)
        
        # Check gradient and parameter health (for sentinel codes in returned grad_norm)
        params_has_nan = jax.tree_util.tree_reduce(
            lambda acc, x: acc | jnp.any(jnp.isnan(x)), lora_trainable, initializer=jnp.bool_(False)
        )
        params_has_inf = jax.tree_util.tree_reduce(
            lambda acc, x: acc | jnp.any(jnp.isinf(x)), lora_trainable, initializer=jnp.bool_(False)
        )
        grad_has_nan = jax.tree_util.tree_reduce(
            lambda acc, x: acc | jnp.any(jnp.isnan(x)), grads, initializer=jnp.bool_(False)
        )
        grad_has_inf = jax.tree_util.tree_reduce(
            lambda acc, x: acc | jnp.any(jnp.isinf(x)), grads, initializer=jnp.bool_(False)
        )

        # Gradient norm (pre-clip; NaN/Inf from bad batches will show as sentinel)
        grad_squares = jax.tree.map(
            lambda g: jnp.sum(g.astype(jnp.float32) ** 2), grads
        )
        grad_norm = jnp.sqrt(
            jax.tree_util.tree_reduce(jnp.add, grad_squares, initializer=jnp.float32(0.0))
        )
        
        # Return debug info as part of the norm (encode flags in special values)
        # If params have NaN: return -3.0
        # If params have Inf: return -4.0
        # If grads have NaN: return -1.0
        # If grads have Inf: return -2.0
        # Otherwise: return computed norm
        grad_norm = jnp.where(params_has_nan, jnp.float32(-3.0), grad_norm)
        grad_norm = jnp.where(params_has_inf, jnp.float32(-4.0), grad_norm)
        grad_norm = jnp.where(grad_has_nan, jnp.float32(-1.0), grad_norm)
        grad_norm = jnp.where(grad_has_inf, jnp.float32(-2.0), grad_norm)
        
        # Update LoRA parameters (optimizer includes gradient clipping)
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        updated_trainable = optax.apply_updates(lora_trainable, updates)
        
        return loss_value, predictions, updated_trainable, new_opt_state, grad_norm
    
    def train_step(
        self,
        dna_sequence: jnp.ndarray,
        organism_index: jnp.ndarray,
        targets: Dict,
        loss_fn: Callable
    ) -> Tuple[float, Dict, float]:
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
            (loss_value, predictions, grad_norm)
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
        
        # Only commit the update when gradients/params are clean.
        # A negative grad_norm encodes a sentinel (NaN/Inf detected); applying
        # NaN gradients would silently corrupt self.params for all future batches.
        if float(grad_norm) >= 0:
            self.lora_params['lora_A'] = updated_lora_trainable['lora_A']
            self.lora_params['lora_B'] = updated_lora_trainable['lora_B']
            self.opt_state = new_opt_state
        
        return loss_value, predictions, grad_norm
    
    def save_lora(self, dirpath: str):
        """Save LoRA parameters using orbax."""
        import os, shutil
        if os.path.exists(dirpath):
            shutil.rmtree(dirpath)
        os.makedirs(dirpath, exist_ok=True)
        checkpointer = ocp.StandardCheckpointer()
        saveable = {
            'lora_A': self.lora_params['lora_A'],
            'lora_B': self.lora_params['lora_B'],
            'scaling': np.array(self.lora_params['scaling']),
            'rank': np.array(self.lora_params['rank']),
            'alpha': np.array(self.lora_params['alpha']),
        }
        checkpointer.save(dirpath, saveable)
        checkpointer.wait_until_finished()

    def load_lora(self, dirpath: str):
        """Load LoRA parameters using orbax."""
        checkpointer = ocp.StandardCheckpointer()
        restored = checkpointer.restore(dirpath)
        self.lora_params['lora_A'] = restored['lora_A']
        self.lora_params['lora_B'] = restored['lora_B']
        self.lora_params['scaling'] = float(restored['scaling'])
        self.lora_params['rank'] = int(restored['rank'])
        self.lora_params['alpha'] = float(restored['alpha'])
        trainable_params = {
            'lora_A': self.lora_params['lora_A'],
            'lora_B': self.lora_params['lora_B']
        }
        self.opt_state = self.optimizer.init(trainable_params)


# ---------------------------------------------------------------------------
# Head-Replacement Fine-Tuning
# ---------------------------------------------------------------------------

class HeadReplaceAlphaGenomeModel:
    """
    Full-parameter fine-tuning via head replacement (DeepMind approach).

    Rather than LoRA adapters, this strategy:
      1. Replaces the pretrained output heads with newly-initialised heads
         tailored to the target tracks (e.g. one ATAC track for CUT&Tag).
      2. Keeps the pretrained trunk weights as the initial values.
      3. Optimises ALL parameters jointly.

    This mirrors the official AlphaGenome fine-tuning colab exactly, except
    that we use our own PyTorch data pipeline and Borzoi-style loss so the
    bedgraph/FASTA workflow is unchanged.

    Precision policy is hard-wired to float32 everywhere, which avoids the
    bfloat16 forward-NaN issue observed in the LoRA experiments.
    """

    def __init__(
        self,
        params: Dict,
        state: Dict,
        apply_fn: Callable,
        mesh: Optional[Mesh] = None,
        learning_rate: float = 5e-4,
        warmup_steps: int = 0,
        total_steps: int = 0,
    ):
        """
        Args:
            params: Merged parameters (trunk from pretrained + fresh heads).
            state:  Model state (BatchNorm EMAs from pretrained checkpoint).
            apply_fn: Callable ``apply_fn(params, state, seq, org) -> predictions``.
            mesh:   JAX device mesh for data-parallel training (optional).
            learning_rate: Peak Adam LR.
            warmup_steps: Warmup steps for cosine schedule (0 = constant LR).
            total_steps:  Total steps for cosine schedule (0 = constant LR).
        """
        self.params = params
        self.state = state
        self.apply_fn = apply_fn
        self.mesh = mesh
        self.num_devices = jax.device_count()

        # Sharding: replicate params, shard batch dim via input arrays
        if mesh is not None:
            self.param_sharding = NamedSharding(mesh, P())
            print(f"  [HeadReplace] Replicating parameters across {self.num_devices} devices...")
            self.params = jax.device_put(self.params, self.param_sharding)
            self.state = jax.device_put(self.state, self.param_sharding)
        else:
            self.param_sharding = None

        # LR schedule
        if warmup_steps > 0 and total_steps > 0:
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=total_steps,
                end_value=learning_rate * 0.1,
            )
            print(f"  [HeadReplace] LR warmup: {warmup_steps} steps, cosine decay over {total_steps} steps")
        else:
            lr_schedule = learning_rate

        # Optimizer — tight clipping matching DeepMind colab
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(lr_schedule),
        )
        self.opt_state = self.optimizer.init(self.params)
        
        # Also replicate optimizer state for multi-GPU
        if mesh is not None:
            self.opt_state = jax.device_put(self.opt_state, self.param_sharding)

        # JIT-compiled train step with explicit sharding specifications
        # Simplified sharding following official DeepMind approach:
        # - Use P('data') for batch dimension only, let JAX infer the rest
        # - This matches the official finetuning notebook exactly
        if mesh is not None:
            # Multi-GPU: simplified sharding specs
            data_sharding = NamedSharding(mesh, P('data'))
            replicated = self.param_sharding
            
            self._train_step_jit = jax.jit(
                self._train_step,
                static_argnames=['loss_fn'],
                in_shardings=(
                    data_sharding,       # dna_sequence (auto-broadcasts to [B,S,4])
                    data_sharding,       # organism_index (auto-broadcasts to [B])
                    data_sharding,       # targets (auto-broadcasts to shape)
                    replicated,          # params
                    replicated,          # state
                    replicated,          # opt_state
                ),
                out_shardings=(
                    replicated,          # loss_value (scalar, replicated)
                    replicated,          # predictions
                    replicated,          # new_params
                    replicated,          # new_opt_state
                    replicated,          # grad_norm
                )
            )
        else:
            # Single-GPU: no sharding needed
            self._train_step_jit = jax.jit(self._train_step, static_argnames=['loss_fn'])

        n_params = sum(x.size for x in jax.tree_util.tree_leaves(self.params))
        print(f"✓ HeadReplaceAlphaGenomeModel initialized")
        print(f"  Total trainable params: {n_params:,}")
        if self.num_devices > 1:
            print(f"  Multi-GPU: {self.num_devices} devices")

    # ------------------------------------------------------------------
    # Internal training step (JIT-compiled)
    # ------------------------------------------------------------------

    def _train_step(
        self,
        dna_sequence: jnp.ndarray,
        organism_index: jnp.ndarray,
        targets: Dict,
        params: Dict,
        state: Dict,
        opt_state: optax.OptState,
        loss_fn: Callable,
    ):
        def compute_loss(p):
            predictions = self.apply_fn(p, state, dna_sequence, organism_index)
            return loss_fn(predictions, targets['signal']), predictions

        (loss_value, predictions), grads = jax.value_and_grad(
            compute_loss, has_aux=True
        )(params)

        # Apply optimizer update (includes gradient clipping and AllReduce for multi-GPU)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        # Compute gradient norm (use device_get to avoid potential NCCL issues)
        grad_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads))
        )

        return loss_value, predictions, new_params, new_opt_state, grad_norm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_step(
        self,
        dna_sequence: jnp.ndarray,
        organism_index: jnp.ndarray,
        targets: Dict,
        loss_fn: Callable,
    ):
        """Single training step; returns (loss, predictions, grad_norm)."""
        loss_value, predictions, new_params, new_opt_state, grad_norm = self._train_step_jit(
            dna_sequence,
            organism_index,
            targets,
            self.params,
            self.state,
            self.opt_state,
            loss_fn,
        )
        # Always apply updates (NaN/Inf checking moved to Python side in finetune.py)
        self.params = new_params
        self.opt_state = new_opt_state
        return loss_value, predictions, grad_norm

    def save(self, dirpath: str):
        """Save params and state via orbax (compatible with dna_model.create()).
        
        Note: This saves only params and state, NOT opt_state. This format is compatible
        with AlphaGenome's dna_model.create() API for loading fine-tuned checkpoints.
        The optimizer state will be reinitialized when resuming training.
        """
        import os, shutil
        if os.path.exists(dirpath):
            shutil.rmtree(dirpath)
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(
            dirpath,
            args=ocp.args.StandardSave({
                'params': self.params,
                'state': self.state,
            })
        )
        checkpointer.wait_until_finished()

    def load(self, dirpath: str):
        """Restore params and state via orbax (compatible with dna_model.create()).
        
        Note: This restores only params and state. The optimizer state is reinitialized
        from scratch, so training will start with a fresh optimizer state.
        """
        checkpointer = ocp.StandardCheckpointer()
        restored = checkpointer.restore(dirpath)
        self.params = restored['params']
        self.state = restored['state']
        # Reinitialize optimizer state with current parameters
        self.opt_state = self.optimizer.init(self.params)
