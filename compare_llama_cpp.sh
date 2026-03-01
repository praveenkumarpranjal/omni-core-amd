#!/bin/bash
# Direct comparison with llama.cpp

MODEL="../llama.cpp/models/smollm2-1.7b-instruct-q4_k_m.gguf"
PROMPT="Once upon a time, in a land far away"
TOKENS=100

echo "=== Performance Comparison: Omni-Core vs llama.cpp ==="
echo ""

echo "Testing Omni-Core (optimized)..."
omni_output=$(./target/release/omni-core -m "$MODEL" -p "$PROMPT" -n $TOKENS -t 0.0 2>&1)
omni_prefill=$(echo "$omni_output" | grep "Prefill:" | sed -n 's/.*= \(.*\) tok\/s.*/\1/p' | awk '{print $1}')
omni_gen=$(echo "$omni_output" | grep "Performance:" | awk '{print $2}')

echo "  Prefill: $omni_prefill tok/s"
echo "  Generation: $omni_gen tok/s"
echo ""

echo "Testing llama.cpp..."
cat > /tmp/parse_llama.py << 'EOF'
import sys
import pexpect

model = sys.argv[1]
prompt = sys.argv[2]
tokens = sys.argv[3]

child = pexpect.spawn(f'../llama.cpp/build/bin/llama-cli -m {model} -p "{prompt}" -n {tokens} --temp 0.0 -ngl 99', encoding='utf-8')
try:
    child.expect(r'\[ Prompt: ([\d.]+) t/s \| Generation: ([\d.]+) t/s \]', timeout=15)
    print(f"{child.match.group(1)}")
    print(f"{child.match.group(2)}")
except:
    print("0.0\n0.0")
finally:
    child.terminate(force=True)
EOF

llama_metrics=$(python3 /tmp/parse_llama.py "$MODEL" "$PROMPT" "$TOKENS")
llama_prefill=$(echo "$llama_metrics" | head -n 1)
llama_gen=$(echo "$llama_metrics" | tail -n 1)

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
