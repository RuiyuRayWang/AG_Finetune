# NCCL Issue Resolution - AlphaGenome Multi-GPU Debugging

**Date:** February 24-25, 2026  
**Status:** ⚠️ IMPLEMENTATION BUG (Multi-GPU IS Supported)

---

## ⚠️ CORRECTION (February 25, 2026)

**PREVIOUS INCORRECT CONCLUSION:** "AlphaGenome does NOT support multi-GPU training"  
**CORRECTED CONCLUSION:** "AlphaGenome DOES support multi-GPU training - our implementation has bugs"

**DeepMind Confirmation:**
> "Yes, the finetuning script supports multiple devices on the same host. You can see this in 'Set Up Device Mesh for Data Parallelism' cell. We've also tested this internally on a machine with 8 H100 GPUs."
> 
> — Vincent, DeepMind AlphaGenome Team

The NCCL errors we encountered are due to **bugs in our custom wrapper code** (`HeadReplaceAlphaGenomeModel`, `LoRAAlphaGenomeModel`), incorrect mesh setup, or PyTorch→JAX data pipeline issues - NOT fundamental model limitations.

---

## Executive Summary

AlphaGenome **DOES support multi-GPU training**, as confirmed by DeepMind. Our NCCL errors stem from implementation bugs in our custom wrapper code, likely related to:
- Incorrect mesh/sharding specifications
- Bugs in `HeadReplaceAlphaGenomeModel` or `LoRAAlphaGenomeModel` classes
- PyTorch→JAX data pipeline sharding issues
- Missing initialization steps from official notebook

---

## Investigation Timeline

### Phase 1: Initial Discovery
- **Issue**: Training failed with `"corrupted comm object detected"` NCCL error
- **Observation**: PyTorch-based training never had NCCL errors (only data corruption)
- **Key insight**: NCCL errors only appear during JAX gradient computation

### Phase 2: Systematic Diagnosis
Created comprehensive diagnostic tools:
- `test_nccl_simple.py` - Basic NCCL operations ✅ PASSED
- `test_nccl_workarounds.py` - Alternative NCCL configurations ✅ PASSED
- `test_alphagenome_model.py` - AlphaGenome with dummy data
  - Test 1 (single device forward): ✅ PASSED
  - Test 2 (multi-GPU forward): ✅ PASSED
  - Test 3 (jitted multi-GPU forward): ✅ PASSED
  - Test 4 (multi-GPU gradient): ❌ **FAILED** with NCCL error

### Phase 3: Root Cause Analysis
**Examined official AlphaGenome code:**

1. **Evaluation code** (`alphagenome_research/evals/track_prediction.py`):
   - Uses multi-GPU with explicit sharding
   - Only for forward passes (no gradients)
   - Pattern: `@jax.jit(in_shardings=..., out_shardings=...)`

2. **Fine-tuning code** (`alphagenome_research/finetuning/finetune.py`):
   - Uses **simple `@jax.jit`** with NO sharding
   - **Single-device training only**
   - Never mentions multi-GPU

**Conclusion:** AlphaGenome's internal operations (likely BatchNorm state management) corrupt NCCL communicators during backward passes.

**⚠️ CORRECTION:** This conclusion was incorrect. DeepMind confirmed multi-GPU training IS supported. The NCCL errors are due to bugs in our implementation, not AlphaGenome's architecture.

---

## Diagnostic Results

| Test | Description | Result |
|------|-------------|--------|
| Basic NCCL | Simple AllReduce operations | ✅ PASS |
| NCCL workarounds | Various NCCL configurations | ✅ PASS |
| AlphaGenome forward (single GPU) | Model prediction | ✅ PASS |
| AlphaGenome forward (multi-GPU) | Sharded batch prediction | ✅ PASS |
| AlphaGenome forward (jitted+sharded) | JIT compilation | ✅ PASS |
| **AlphaGenome gradients (multi-GPU)** | Gradient AllReduce | ❌ **FAIL** |

**Error message:**
```
INTERNAL: NCCL operation ncclAllReduce(...) failed: invalid argument
corrupted comm object detected
```

---

## Why PyTorch Training Never Had NCCL Errors

The PyTorch-based training (`scripts/pytorch_ft.py`) had:
- ✅ Data corruption issues (7,000 → 68,780 amplification)
- ✅ NO NCCL errors

**Reason:** PyTorch handled data loading, JAX only did forward/backward passes on individual batches without explicit multi-GPU gradient AllReduce through AlphaGenome's architecture.

---

## Solution: Single-GPU Training with Gradient Accumulation

### Implementation

Created `scripts/train_singlegpu.py` following official AlphaGenome pattern:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_singlegpu.py \\
    --config configs/config_bigwig.yaml \\
    --accumulation-steps 8 \\
    --epochs 10
```

### Key Features

1. **Force single GPU**: `CUDA_VISIBLE_DEVICES=0` or `os.environ['CUDA_VISIBLE_DEVICES'] = '0'`
2. **Gradient accumulation**:
   ```python
   for micro_step in range(8):
       loss, grads = compute_loss_and_grads(params, state, batch)
       accumulated_grads = accumulate(accumulated_grads, grads)
   
   # Average and apply
   accumulated_grads = jax.tree.map(lambda g: g / 8, accumulated_grads)
   updates, opt_state = optimizer.update(accumulated_grads, opt_state, params)
   ```
3. **Simple `@jax.jit`**: No sharding specifications
4. **Official pattern**: Based on `alphagenome_research/finetuning/finetune.py`

### Trade-offs

| Aspect | Multi-GPU (MUST FIX) | Single-GPU (NOT VIABLE) |
|--------|---------------------|-------------------------|
| Effective batch size | 8 | 8 (via accumulation) |
| Training speed | 1x | **8x slower (unacceptable)** |
| GPU utilization | 100% × 8 | 100% × 1 (wastes 7 GPUs) |
| Correctness | Should work (DeepMind confirmed) | Works but too slow |
| Total training time | ~25 hours | **~200 hours (prohibitive)** |

**Conclusion:** Single-GPU training is NOT a viable long-term solution. We MUST fix multi-GPU implementation.

---

## Recommendations

### MANDATORY: Fix Multi-GPU Implementation

**DeepMind has confirmed multi-GPU training works on 8 H100 GPUs. Our NCCL errors are implementation bugs that MUST be fixed.**

**Debugging steps (in order):**

1. **Use `jax.eval_shape` to validate training logic** (DeepMind's primary suggestion):
   ```python
   # Validate shapes without running GPU kernels
   shape_result = jax.eval_shape(train_step, params, state, dummy_batch)
   ```
   This will reveal shape mismatches or sharding issues before GPU execution.

2. **Compare with official mesh setup line-by-line**:
   - Open official notebook "Set Up Device Mesh for Data Parallelism" cell
   - Compare every line with our `models/model.py` mesh creation
   - Check for missing initialization steps

3. **Test official finetuning module first** (human data):
   - Verify multi-GPU works on our hardware with official code
   - Rules out hardware/driver issues
   - Provides working reference implementation

4. **Systematic debugging of our wrapper classes**:
   - Review `HeadReplaceAlphaGenomeModel._train_step()` sharding specs
   - Check `LoRAAlphaGenomeModel._train_step()` sharding specs
   - Validate PyTorch→JAX data pipeline doesn't corrupt sharded arrays
   - Ensure `jax.jit` has correct `in_shardings` and `out_shardings`

### Testing Tool Only: Single-GPU Script

`scripts/train_singlegpu.py` exists **only for testing data pipeline and loss computation**, not for production training:
- Use to verify data loads correctly
- Use to verify loss decreases
- **DO NOT use for full training runs** (8x slower is unacceptable)

### Resources
- Official "Set Up Device Mesh for Data Parallelism" cell in finetuning notebook
- DeepMind contact: Vincent (AlphaGenome team) - confirmed multi-GPU works
- Use `jax.eval_shape` for debugging without GPU execution

---

## Files Modified

| File | Purpose |
|------|---------|
| `scripts/test_nccl_simple.py` | Test basic NCCL operations ✅ |
| `scripts/diagnose_corrupted_comm.py` | Detailed NCCL diagnostics |
| `scripts/test_nccl_workarounds.py` | Alternative NCCL configurations ✅ |
| `scripts/test_alphagenome_model.py` | AlphaGenome multi-GPU gradient test ❌ |
| `scripts/train_singlegpu.py` | **Working single-GPU training script (temporary)** ⭐ |
| `.github/copilot-instructions.md` | Updated with corrected findings |

---

## Documented Constraints

1. **Multi-GPU training IS supported by AlphaGenome** (DeepMind confirmed on 8 H100 GPUs)
2. **Our implementation has bugs** - NCCL errors are due to incorrect mesh setup or wrapper code issues
3. **Forward passes work fine** with multi-GPU in our tests (Tests 1-3 passed)
4. **Gradient computation fails** in our implementation (Test 4) - needs debugging

---

## Next Steps

### IMMEDIATE: Fix Multi-GPU Implementation (Required)

1. **Use `jax.eval_shape`** as primary debugging tool:
   - Validate training step logic without GPU execution
   - Identify shape/sharding mismatches before NCCL calls
   - Follow DeepMind's explicit recommendation

2. **Line-by-line comparison with official notebook**:
   - "Set Up Device Mesh for Data Parallelism" cell is the reference
   - Our mesh setup in `models/model.py` must match exactly
   - Check for missing initialization or configuration steps

3. **Test official code with human data**:
   - Verify our hardware/drivers support multi-GPU training
   - Provides working baseline to compare against
   - Eliminates hardware as potential issue

4. **Fix bugs in wrapper classes**:
   - Audit `HeadReplaceAlphaGenomeModel` sharding specifications
   - Audit `LoRAAlphaGenomeModel` sharding specifications  
   - Fix PyTorch→JAX data pipeline if needed
   - Test with `test_alphagenome_model.py` after each fix

### TEST ONLY: Single-GPU Validation (Not for Production)

If needed for quick validation of data pipeline:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_singlegpu.py --config configs/config_bigwig.yaml --epochs 1
```

**Use only to verify:**
- Data loads without errors
- Loss computation works
- No crashes in first 100 steps

**DO NOT use for production training** - 8x slower is unacceptable.

---

**Status: Multi-GPU implementation required** ⚠️
