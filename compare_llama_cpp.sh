#!/bin/bash
# Direct comparison with llama.cpp

MODEL="../llama.cpp/models/smollm2-1.7b-instruct-q4_k_m.gguf"
PROMPT="Once upon a time, in a land far away"
TOKENS=100

echo "=== Performance Comparison: Omni-Core vs llama.cpp ==="
echo ""

echo "Testing Omni-Core (optimized)..."
omni_output=$(./target/release/omni-core -m "$MODEL" -p "$PROMPT" -n $TOKENS -t 0.0 2>&1)
omni_prefill=$(echo "$omni_output" | grep "Prefill:" | awk '{print $7}')
omni_gen=$(echo "$omni_output" | grep "Performance:" | awk '{print $2}')

echo "  Prefill: $omni_prefill tok/s"
echo "  Generation: $omni_gen tok/s"
echo ""

echo "Testing llama.cpp..."
timeout 15s ../llama.cpp/build/bin/llama-cli -m "$MODEL" -p "$PROMPT" -n $TOKENS --temp 0.0 -ngl 99 2>&1 > /tmp/llama_output.txt
llama_prefill=$(grep "Prompt:" /tmp/llama_output.txt | awk '{print $2}')
llama_gen=$(grep "Generation:" /tmp/llama_output.txt | awk '{print $2}')

echo "  Prefill: $llama_prefill tok/s"
echo "  Generation: $llama_gen tok/s"
echo ""

echo "=== Comparison ==="
if [ -n "$omni_gen" ] && [ -n "$llama_gen" ]; then
    gen_ratio=$(echo "scale=1; ($omni_gen / $llama_gen) * 100" | bc)
    echo "Generation: Omni-Core is ${gen_ratio}% of llama.cpp speed"
fi

if [ -n "$omni_prefill" ] && [ -n "$llama_prefill" ]; then
    prefill_ratio=$(echo "scale=1; ($omni_prefill / $llama_prefill) * 100" | bc)
    echo "Prefill: Omni-Core is ${prefill_ratio}% of llama.cpp speed"
fi
