#!/usr/bin/env python3
"""
Test alternative JAX/NCCL initialization strategies.

The "corrupted comm object" error often indicates improper initialization.
Try various workarounds.
"""

import os
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

print("=" * 70)
print("TESTING ALTERNATIVE NCCL INITIALIZATION STRATEGIES")
print("=" * 70)

def test_basic_collective():
    """Test if basic multi-GPU collective works."""
    try:
        mesh = Mesh(jax.devices(), ('batch',))
        sharding = NamedSharding(mesh, P('batch'))
        
        @jax.jit
        def sum_fn(x):
            return jnp.sum(x)
        
        arr = jax.device_put(jnp.ones((80, 10)), sharding)
        result = sum_fn(arr)
        print(f"  Result: {result}")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False

# Strategy 1: Default (current approach)
print("\n[Strategy 1] Default JAX initialization")
print("-" * 70)
if test_basic_collective():
    print("✓ Default approach works")
else:
    print("✗ Default approach fails")

# Strategy 2: Force synchronous compilation
print("\n[Strategy 2] Synchronous compilation (XLA_FLAGS=--xla_gpu_enable_async_collectives=false)")
print("-" * 70)
original_xla_flags = os.environ.get('XLA_FLAGS', '')
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_async_collectives=false'
jax.clear_caches()  # Force recompilation
if test_basic_collective():
    print("✓ Synchronous compilation works")
else:
    print("✗ Synchronous compilation fails")
os.environ['XLA_FLAGS'] = original_xla_flags

# Strategy 3: Disable NCCL P2P (peer-to-peer)
print("\n[Strategy 3] Disable NCCL P2P (NCCL_P2P_DISABLE=1)")
print("-" * 70)
os.environ['NCCL_P2P_DISABLE'] = '1'
jax.clear_caches()
if test_basic_collective():
    print("✓ Disabling P2P works")
else:
    print("✗ Disabling P2P fails")
del os.environ['NCCL_P2P_DISABLE']

# Strategy 4: Use shared memory for NCCL
print("\n[Strategy 4] Force shared memory transport (NCCL_SHM_DISABLE=0)")
print("-" * 70)
os.environ['NCCL_SHM_DISABLE'] = '0'
jax.clear_caches()
if test_basic_collective():
    print("✓ Shared memory works")
else:
    print("✗ Shared memory fails")
del os.environ['NCCL_SHM_DISABLE']

# Strategy 5: Disable NCCL buffering
print("\n[Strategy 5] Disable NCCL buffering (NCCL_BUFFSIZE=0)")
print("-" * 70)
os.environ['NCCL_BUFFSIZE'] = '0'
jax.clear_caches()
if test_basic_collective():
    print("✓ No buffering works")
else:
    print("✗ No buffering fails")
del os.environ['NCCL_BUFFSIZE']

# Strategy 6: Single GPU mode (bypass NCCL entirely)
print("\n[Strategy 6] Single GPU mode (CUDA_VISIBLE_DEVICES=0)")
print("-" * 70)
print("  (This will only show 1 device, bypassing NCCL)")
original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Need to restart JAX backend for this to take effect
print("  → Cannot test this within the same process")
print("  → Run: CUDA_VISIBLE_DEVICES=0 python scripts/tf-bw_ft.py ...")
os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices

# Strategy 7: Check for NVLink connectivity
print("\n[Strategy 7] Check GPU topology")
print("-" * 70)
try:
    # This is a simple test - JAX doesn't expose full topology info
    devices = jax.devices()
    print(f"  {len(devices)} devices available")
    
    # Try to transfer data between all pairs of devices
    test_arr = jnp.ones(1000)
    for i, dev1 in enumerate(devices):
        for j, dev2 in enumerate(devices):
            if i != j:
                arr1 = jax.device_put(test_arr, dev1)
                arr2 = jax.device_put(arr1, dev2)  # Transfer from dev1 to dev2
    print("✓ All device-to-device transfers work")
except Exception as e:
    print(f"✗ Device connectivity issue: {e}")

print("\n" + "=" * 70)
print("RECOMMENDATIONS:")
print("=" * 70)
print("""
If Strategy 1 (default) failed:
  → Try running training with the environment variable from a working strategy
  → Example: NCCL_P2P_DISABLE=1 python scripts/tf-bw_ft.py ...

If all strategies failed:
  → NCCL is fundamentally broken on this system
  → Check: nvidia-smi topo -m (GPU topology)
  → Check: NCCL version compatibility with CUDA version
  → Try: conda install -c nvidia nccl=2.18 (or different version)

If Strategy 6 (single GPU) is needed:
  → Run: CUDA_VISIBLE_DEVICES=0 python scripts/tf-bw_ft.py --batch-size 1 ...
  → But this defeats the purpose of having 8 GPUs!

Next steps:
  1. Run: nvidia-smi topo -m
     → Check if GPUs are connected via NVLink or PCIe
  2. Run: python -c "import jax; print(jax.lib.version)"
     → Check JAX version
  3. Run: conda list | grep nccl
     → Check NCCL version
  4. Try different NCCL version: conda install -c nvidia nccl=2.18
""")
