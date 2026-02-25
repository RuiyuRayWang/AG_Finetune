#!/bin/bash
# Master diagnostic script - runs all tests in sequence to find the NCCL issue

set -e  # Exit on first failure

echo "========================================================================"
echo "MASTER NCCL DIAGNOSTIC - Finding the 'corrupted comm object' root cause"
echo "========================================================================"
echo ""
echo "Based on initial tests:"
echo "  ✓ Basic NCCL operations work (test_nccl_simple.py passed)"
echo "  ✓ All NCCL workarounds work (test_nccl_workarounds.py passed)"
echo "  ✗ Training fails with 'corrupted comm object' error"
echo ""
echo "This means the issue is NOT in NCCL itself, but in how we're using it"
echo "with the AlphaGenome model or data. Let's narrow it down..."
echo ""
echo "========================================================================"

# Test 1: AlphaGenome model in isolation
echo ""
echo "[Test 1] AlphaGenome model with dummy data"
echo "------------------------------------------------------------------------"
echo "Testing the base AlphaGenome model with simple synthetic inputs."
echo "If this fails: The AlphaGenome model itself has NCCL issues."
echo "If this passes: Our wrapper or data is the problem."
echo ""
read -p "Press ENTER to run test_alphagenome_model.py..."
python scripts/test_alphagenome_model.py
TEST1_RESULT=$?

if [ $TEST1_RESULT -ne 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✗ FOUND IT! Base AlphaGenome model triggers NCCL corruption"
    echo "========================================================================"
    echo ""
    echo "The issue is in the AlphaGenome model's internal operations."
    echo ""
    echo "Possible causes:"
    echo "  1. AlphaGenome internally uses pmap/jit that conflicts with our outer jit"
    echo "  2. The model expects specific input shapes we're not providing"
    echo "  3. Haiku state management has sharding issues"
    echo "  4. AlphaGenome's BatchNorm operations trigger NCCL incorrectly"
    echo ""
    echo "Next steps:"
    echo "  1. Contact DeepMind AlphaGenome team about multi-GPU fine-tuning"
    echo "  2. Try running model.apply WITHOUT jit wrapper"
    echo "  3. Inspect AlphaGenome source for nested jit/pmap calls"
    echo "  4. Try older JAX version (0.4.20) for compatibility"
    echo ""
    exit 1
fi

echo ""
echo "✓ Test 1 passed - Base AlphaGenome model works with NCCL"
echo ""

# Test 2: Our model wrapper
echo ""
echo "[Test 2] HeadReplaceAlphaGenomeModel wrapper"
echo "------------------------------------------------------------------------"
echo "Testing our custom model wrapper with dummy data."
echo "If this fails: Our wrapper code (models/model.py) has issues."
echo "If this passes: The actual training data is the problem."
echo ""
read -p "Press ENTER to run test_wrapper_model.py..."
python scripts/test_wrapper_model.py
TEST2_RESULT=$?

if [ $TEST2_RESULT -ne 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✗ FOUND IT! Our model wrapper triggers NCCL corruption"
    echo "========================================================================"
    echo ""
    echo "The issue is in models/model.py HeadReplaceAlphaGenomeModel class."
    echo ""
    echo "Check:"
    echo "  1. _train_step_jit function - are sharding specs correct?"
    echo "  2. How we're calling model.apply - check argument order"
    echo "  3. Loss function integration - is loss_fn passed correctly?"
    echo "  4. Optimizer state sharding - is opt_state properly replicated?"
    echo ""
    echo "Inspect the error traceback above for the exact line that fails."
    echo ""
    exit 1
fi

echo ""
echo "✓ Test 2 passed - Model wrapper works with NCCL"
echo ""

# Test 3: Actual training data
echo ""
echo "[Test 3] Actual training data from TensorFlow dataset"
echo "------------------------------------------------------------------------"
echo "Testing with real genomic sequences and CUT&Tag signal."
echo "If this fails: Something in the actual data triggers NCCL corruption."
echo "If this passes: The full training script has setup issues."
echo ""
read -p "Press ENTER to run test_actual_data.py..."
python scripts/test_actual_data.py
TEST3_RESULT=$?

if [ $TEST3_RESULT -ne 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✗ FOUND IT! Actual training data triggers NCCL corruption"
    echo "========================================================================"
    echo ""
    echo "The issue is in the real genomic data or how it's processed."
    echo ""
    echo "Check:"
    echo "  1. /tmp/bad_batch.npz - the specific batch that failed"
    echo "  2. Batch signal values - are they too large/small?"
    echo "  3. Sequence encoding - any NaN/Inf values?"
    echo "  4. Window genomic coordinates - specific chromosome causing issues?"
    echo ""
    echo "Try:"
    echo "  - Skip this specific batch and test the next one"
    echo "  - Use different safe windows"
    echo "  - Test with human data instead of mouse"
    echo ""
    exit 1
fi

echo ""
echo "✓ Test 3 passed - Actual data works!"
echo ""

# All tests passed
echo ""
echo "========================================================================"
echo "✓✓✓ ALL DIAGNOSTIC TESTS PASSED ✓✓✓"
echo "========================================================================"
echo ""
echo "This is VERY strange. All components work individually:"
echo "  ✓ Basic NCCL operations"
echo "  ✓ AlphaGenome model"
echo "  ✓ Model wrapper"
echo "  ✓ Actual training data"
echo ""
echo "But the full training script (scripts/tf-bw_ft.py) still fails!"
echo ""
echo "Possible causes:"
echo "  1. Something in the full training loop that's not in these tests"
echo "  2. A specific combination of dataset iteration + model state"
echo "  3. Memory accumulation over many batches"
echo "  4. TensorFlow's prefetching interfering with JAX device management"
echo ""
echo "Next steps:"
echo "  1. Run training script with ONLY 3 batches to see if it works"
echo "  2. Add more logging to tf-bw_ft.py to see what differs"
echo "  3. Compare test_actual_data.py line-by-line with tf-bw_ft.py"
echo "  4. Check if there are any global settings in tf-bw_ft.py not here"
echo ""
