#!/usr/bin/env python3
"""
Diagnose NCCL "corrupted comm object" error.

This error means NCCL's communicator initialization failed or the 
communicator is being used incorrectly. Test various scenarios.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import optax

print("=" * 60)
print("NCCL COMMUNICATOR DIAGNOSTICS")
print("=" * 60)

# System info
print(f"\nJAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")

# Test 1: Check if JAX can see all devices
print("\n" + "=" * 60)
print("[Test 1] Device visibility")
print("=" * 60)
try:
    for i, device in enumerate(jax.devices()):
        print(f"  Device {i}: {device}")
    print("✓ All devices visible")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 2: Simple array creation on each device
print("\n" + "=" * 60)
print("[Test 2] Array creation on individual devices")
print("=" * 60)
try:
    for i, device in enumerate(jax.devices()):
        arr = jax.device_put(jnp.ones(10), device)
        assert arr.device() == device
    print("✓ Can create arrays on all devices")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 3: Mesh creation (this initializes NCCL communicator)
print("\n" + "=" * 60)
print("[Test 3] Mesh creation (NCCL communicator init)")
print("=" * 60)
try:
    mesh = Mesh(jax.devices(), ('batch',))
    print(f"  Mesh created: {mesh}")
    print("✓ Mesh created successfully")
except Exception as e:
    print(f"✗ Failed during mesh creation: {e}")
    print("\n  → This means NCCL communicator initialization failed!")
    print("  → Check: NCCL version, CUDA version, GPU connectivity")
    exit(1)

# Test 4: Simple sharded array creation
print("\n" + "=" * 60)
print("[Test 4] Sharded array creation")
print("=" * 60)
try:
    sharding = NamedSharding(mesh, P('batch'))
    arr = jax.device_put(jnp.ones((80, 10)), sharding)
    print(f"  Array shape: {arr.shape}, sharding: {arr.sharding}")
    print("✓ Sharded array created")
except Exception as e:
    print(f"✗ Failed: {e}")
    exit(1)

# Test 5: Simple collective operation (sum) - does NOT use NCCL if not jitted
print("\n" + "=" * 60)
print("[Test 5] Non-jitted collective operation (sum)")
print("=" * 60)
try:
    sharding = NamedSharding(mesh, P('batch'))
    arr = jax.device_put(jnp.ones((80, 10)), sharding)
    result = jnp.sum(arr)
    print(f"  Sum result: {result}")
    print("✓ Non-jitted collective passed")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 6: Jitted function with sharded input - triggers NCCL AllReduce
print("\n" + "=" * 60)
print("[Test 6] Jitted function with sharded input (NCCL AllReduce)")
print("=" * 60)
try:
    sharding = NamedSharding(mesh, P('batch'))
    
    @jax.jit
    def sum_fn(x):
        return jnp.sum(x)
    
    arr = jax.device_put(jnp.ones((80, 10)), sharding)
    result = sum_fn(arr)  # This triggers NCCL AllReduce
    print(f"  Sum result: {result}")
    print("✓ Jitted collective with AllReduce passed")
except Exception as e:
    print(f"✗ Failed: {e}")
    print("\n  → NCCL communicator is corrupted!")
    print("  → The communicator was created (Test 3) but fails when used")
    exit(1)

# Test 7: Gradient computation with replicated params + sharded data
print("\n" + "=" * 60)
print("[Test 7] Gradient with replicated params + sharded data")
print("=" * 60)
try:
    batch_sharding = NamedSharding(mesh, P('batch'))
    replicated = NamedSharding(mesh, P())
    
    params = jax.device_put(jnp.ones((10, 5)), replicated)
    data = jax.device_put(jnp.ones((80, 10)), batch_sharding)
    
    def loss_fn(p, x):
        return jnp.sum((x @ p) ** 2)
    
    @jax.jit
    def grad_fn(p, x):
        return jax.grad(loss_fn)(p, x)
    
    grads = grad_fn(params, data)
    print(f"  Gradient shape: {grads.shape}")
    print("✓ Gradient computation passed")
except Exception as e:
    print(f"✗ Failed: {e}")
    print("\n  → NCCL fails during gradient AllReduce")
    exit(1)

# Test 8: Full optimizer update (closest to training)
print("\n" + "=" * 60)
print("[Test 8] Full optimizer update (optimizer.update)")
print("=" * 60)
try:
    batch_sharding = NamedSharding(mesh, P('batch'))
    replicated = NamedSharding(mesh, P())
    
    params = jax.device_put(jnp.ones((10, 5)), replicated)
    data = jax.device_put(jnp.ones((80, 10)), batch_sharding)
    
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    opt_state = jax.device_put(opt_state, replicated)
    
    def loss_fn(p, x):
        return jnp.sum((x @ p) ** 2)
    
    @jax.jit
    def train_step(p, opt_st, x):
        loss, grads = jax.value_and_grad(loss_fn)(p, x)
        updates, new_opt_st = optimizer.update(grads, opt_st, p)
        new_p = optax.apply_updates(p, updates)
        return loss, new_p, new_opt_st
    
    loss, new_params, new_opt_state = train_step(params, opt_state, data)
    print(f"  Loss: {loss}")
    print("✓ Optimizer update passed")
except Exception as e:
    print(f"✗ Failed: {e}")
    print("\n  → NCCL fails during optimizer.update AllReduce")
    print("  → This is the same pattern as your training code!")
    exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED")
print("=" * 60)
print("\nNCCL is working correctly. The training error must be:")
print("  1. Model-specific (AlphaGenome's apply_fn triggers the error)")
print("  2. Data-specific (something about the actual training data)")
print("  3. Or a more complex interaction not covered by these tests")
