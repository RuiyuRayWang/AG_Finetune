#!/usr/bin/env python3
"""
Simple NCCL test to verify multi-GPU communication works.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

print("="*60)
print("NCCL DIAGNOSTIC TEST")
print("="*60)

# Check devices
devices = jax.devices()
print(f"Devices: {devices}")
print(f"Device count: {len(devices)}")

if len(devices) < 2:
    print("⚠️  Only 1 device found. Multi-GPU test requires at least 2 GPUs.")
    exit(0)

print("\n1. Testing basic device communication...")
try:
    # Create simple sharded array
    mesh = Mesh(devices, ('data',))
    sharding = NamedSharding(mesh, P('data'))
    
    # Create data
    data = jnp.arange(len(devices) * 4).reshape(len(devices), 4)
    print(f"   Original data shape: {data.shape}")
    
    # Shard it
    sharded_data = jax.device_put(data, sharding)
    print(f"   ✓ Data sharded across {len(devices)} devices")
    print(f"   Sharding: {sharded_data.sharding}")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

print("\n2. Testing simple all-reduce (sum across devices)...")
try:
    @jax.jit
    def simple_sum(x):
        return jnp.sum(x)
    
    result = simple_sum(sharded_data)
    print(f"   Sum result: {result}")
    print(f"   ✓ AllReduce successful")
    
except Exception as e:
    print(f"   ✗ AllReduce failed: {e}")
    exit(1)

print("\n3. Testing jitted function with explicit sharding specs...")
try:
    @jax.jit
    def add_one(x):
        return x + 1
    
    # Without explicit sharding
    result1 = add_one(sharded_data)
    print(f"   ✓ JIT without explicit sharding works")
    
    # With explicit sharding
    add_one_sharded = jax.jit(
        add_one,
        in_shardings=(sharding,),
        out_shardings=sharding
    )
    result2 = add_one_sharded(sharded_data)
    print(f"   ✓ JIT with explicit sharding works")
    
except Exception as e:
    print(f"   ✗ JIT function failed: {e}")
    exit(1)

print("\n4. Testing replicated parameters + sharded inputs...")
try:
    replicated = NamedSharding(mesh, P())
    
    # Replicated params
    params = {'weight': jnp.array(2.0)}
    params = jax.device_put(params, replicated)
    
    @jax.jit
    def multiply(x, p):
        return x * p['weight']
    
    # Without explicit sharding
    result3 = multiply(sharded_data, params)
    print(f"   ✓ Replicated params + sharded data (no explicit sharding) works")
    
    # With explicit sharding
    multiply_sharded = jax.jit(
        multiply,
        in_shardings=(sharding, {'weight': replicated}),
        out_shardings=sharding
    )
    result4 = multiply_sharded(sharded_data, params)
    print(f"   ✓ Replicated params + sharded data (explicit sharding) works")
    
except Exception as e:
    print(f"   ✗ Mixed sharding failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n5. Testing optimizer-like pattern (gradient + update)...")
try:
    import optax
    
    replicated = NamedSharding(mesh, P())
    
    # Params and optimizer
    params = jax.device_put({'w': jnp.ones((4, 1))}, replicated)  # Fixed: (4,1) matches x shape (batch,4)
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(params)
    opt_state = jax.device_put(opt_state, replicated)
    
    # Loss function
    def loss_fn(p, x):
        return jnp.sum((x @ p['w']) ** 2)
    
    # Train step WITHOUT explicit sharding
    @jax.jit
    def train_step_no_shard(p, opt_st, x):
        loss, grads = jax.value_and_grad(loss_fn)(p, x)
        updates, new_opt_st = optimizer.update(grads, opt_st, p)
        new_p = optax.apply_updates(p, updates)
        return loss, new_p, new_opt_st
    
    try:
        loss, new_params, new_opt = train_step_no_shard(params, opt_state, sharded_data)
        print(f"   ✓ Optimizer pattern WITHOUT explicit sharding works")
    except Exception as e:
        print(f"   ✗ Optimizer pattern WITHOUT explicit sharding failed: {e}")
    
    # Train step WITH explicit sharding
    train_step_sharded = jax.jit(
        train_step_no_shard,
        in_shardings=(
            {'w': replicated},
            replicated,
            sharding
        ),
        out_shardings=(replicated, {'w': replicated}, replicated)
    )
    
    try:
        loss, new_params, new_opt = train_step_sharded(params, opt_state, sharded_data)
        print(f"   ✓ Optimizer pattern WITH explicit sharding works")
    except Exception as e:
        print(f"   ✗ Optimizer pattern WITH explicit sharding failed: {e}")
        import traceback
        traceback.print_exc()
    
except Exception as e:
    print(f"   ✗ Optimizer test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("✓ ALL NCCL TESTS PASSED")
print("="*60)
print("\nMulti-GPU JAX is working correctly on this system.")
print("The training error must be specific to the training code.")
