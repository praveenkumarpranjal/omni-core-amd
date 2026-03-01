#!/bin/bash
# Performance benchmark script for Omni-Core optimizations

set -e

MODEL_PATH="${1:-../llama.cpp/models/smollm2-1.7b-instruct-q4_k_m.gguf}"
PROMPT="Once upon a time"
TOKENS=200

echo "=== Omni-Core Performance Benchmark ==="
echo "Model: $MODEL_PATH"
echo "Prompt: $PROMPT"
echo "Tokens: $TOKENS"
echo ""

# Run benchmark
./target/release/omni-core \
    -m "$MODEL_PATH" \
    -p "$PROMPT" \
    -n $TOKENS \
    -t 0.0 \
    2>&1 | tee bench_result.log

echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to bench_result.log"
