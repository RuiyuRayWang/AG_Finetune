#!/bin/bash
# Run training with NCCL debug output to diagnose the AllReduce error

export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

python scripts/tf-bw_ft.py --config configs/config_bigwig.yaml
