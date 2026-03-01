#!/bin/bash
# Quick verification test for optimizations

echo "=== Omni-Core Optimization Verification ==="
echo ""

# Test 1: Basic inference
echo "Test 1: Basic Inference"
./target/release/omni-core \
    -m ../llama.cpp/models/smollm2-1.7b-instruct-q4_k_m.gguf \
    -p "Hello, world!" \
    -n 20 \
    -t 0.0 \
    2>&1 | grep -E "(Prefill|Performance|Assistant:)" | head -5

echo ""
echo "Test 2: Longer Generation"
./target/release/omni-core \
    -m ../llama.cpp/models/smollm2-1.7b-instruct-q4_k_m.gguf \
    -p "The meaning of life is" \
    -n 50 \
    -t 0.7 \
    2>&1 | grep -E "(Prefill|Performance)" 

echo ""
echo "✅ Verification Complete"
echo ""
echo "Expected Results:"
echo "  - Prefill: >400 tok/s"
echo "  - Generation: >250 tok/s"
echo "  - No GPU errors"
echo "  - Coherent output"
