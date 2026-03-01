#!/bin/bash
# Comprehensive benchmark suite for Omni-Core optimizations

set -e

echo "=== Omni-Core Optimization Benchmark Suite ==="
echo "Date: $(date)"
echo ""

# Test different model sizes if available
MODELS=(
    "../llama.cpp/models/smollm2-1.7b-instruct-q4_k_m.gguf:1.7B"
    "../llama.cpp/models/qwen2.5-0.5b-q8_0.gguf:0.5B"
)

PROMPT="Once upon a time, in a land far away, there lived a wise old wizard who"
TOKENS=200

for model_info in "${MODELS[@]}"; do
    IFS=':' read -r model_path model_name <<< "$model_info"
    
    if [ ! -f "$model_path" ]; then
        echo "⚠️  Skipping $model_name (file not found: $model_path)"
        continue
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: $model_name"
    echo "Model: $model_path"
    echo ""
    
    # Run 3 times and take average
    total_tps=0
    total_prefill=0
    runs=3
    
    for i in $(seq 1 $runs); do
        echo "Run $i/$runs..."
        output=$(./target/release/omni-core \
            -m "$model_path" \
            -p "$PROMPT" \
            -n $TOKENS \
            -t 0.0 \
            2>&1)
        
        # Extract tokens/sec
        tps=$(echo "$output" | grep "Performance:" | awk '{print $2}')
        prefill=$(echo "$output" | grep "Prefill:" | awk '{print $7}')
        
        if [ -n "$tps" ]; then
            total_tps=$(echo "$total_tps + $tps" | bc)
        fi
        if [ -n "$prefill" ]; then
            total_prefill=$(echo "$total_prefill + $prefill" | bc)
        fi
    done
    
    avg_tps=$(echo "scale=2; $total_tps / $runs" | bc)
    avg_prefill=$(echo "scale=2; $total_prefill / $runs" | bc)
    
    echo ""
    echo "Results for $model_name:"
    echo "  Average Generation: $avg_tps tok/s"
    echo "  Average Prefill: $avg_prefill tok/s"
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Benchmark suite complete!"
