# Omni-Core: High-Performance LLM Inference Engine for AMD GPUs

A blazingly fast LLM inference engine optimized for AMD MI300X GPUs, written in Rust with HIP/ROCm acceleration.

## 🚀 Performance

**Current Performance (1.7B model):**
- **Generation**: ~415 tok/s (95% of llama.cpp) ✅
- **Prefill**: ~465 tok/s (37% of llama.cpp)

**vs Baseline:**
- Generation: +179-218% (3.5x faster)
- Prefill: +349-373% (4.5x faster)

## ✨ Features

- **Zero-allocation inference** - All buffers allocated once, reused forever
- **Optimized GPU pipeline** - Async execution with minimal synchronization
- **Quantization support** - Q4_K, Q6_K, Q8_0, Q8_1 formats
- **KV cache** - Fully implemented for fast generation
- **Interactive mode** - Chat with the model in real-time
- **Batch prefill** - Fast initial prompt processing
- **Multiple architectures** - LLaMA, Gemma-2, Qwen, Mistral, etc.

## 🛠️ Requirements

- AMD MI300X GPU (or compatible ROCm device)
- ROCm 5.0+ with HIP runtime
- rocBLAS library
- Rust 1.70+

## 📦 Installation

```bash
# Clone the repository (includes prebuilt libraries)
git clone https://github.com/praveenkumarpranjal/omni-core.git
cd omni-core

# Build in release mode
cargo build --release

# The binary will be at: target/release/omni-core
```

**Note**: The repository includes prebuilt GPU libraries in `libs/` directory (~196MB) for convenience. These are required for the inference engine to work.

## 🎯 Usage

### Basic Inference
```bash
./target/release/omni-core \
  -m path/to/model.gguf \
  -p "Once upon a time" \
  -n 100
```

### Interactive Chat Mode
```bash
./target/release/omni-core \
  -m path/to/model.gguf \
  -i \
  -t 0.7
```

### Command-Line Options
- `-m <path>` - Model path (GGUF format)
- `-p <text>` - Initial prompt
- `-n <num>` - Maximum tokens to generate (default: 2048)
- `-i` or `--interactive` - Enable interactive chat mode
- `-t <temp>` - Temperature (0.0 = greedy, higher = more creative)
- `--top-k <k>` - Top-K sampling (default: 40)
- `--top-p <p>` - Top-P sampling (default: 0.95)

### Interactive Mode Commands
- Type your message and press Enter
- Type `/exit` or `exit` to quit
- Empty line to skip

## 🧪 Benchmarking

```bash
# Quick benchmark (200 tokens)
./bench.sh

# Comprehensive benchmark suite
./benchmark_suite.sh

# Compare with llama.cpp
./compare_llama_cpp.sh
```

## 🏗️ Architecture

### Core Components

**Rust Layer** (`src/`)
- `main.rs` - CLI interface and interactive mode
- `graph.rs` - Model graph and inference logic
- `hip.rs` - HIP/ROCm GPU operations
- `gguf.rs` - GGUF model format parser
- `tokenizer.rs` - Tokenization (BPE/SentencePiece)

**GPU Kernels** (`kernels/`)
- `ggml_bridge.cpp` - C++ bridge for GPU operations
- `rope_fast.cpp` - Optimized RoPE implementation
- `asm/*.hsaco` - Hand-optimized assembly kernels

### Key Optimizations

1. **Phase 1: Sync Removal** (+89% gen, +346% prefill)
   - Eliminated synchronization from layer loop
   - Single sync before CPU readback

2. **Phase 2: Batch Infrastructure** (+28% gen, +5% prefill)
   - Added `forward_batch()` for multi-token processing
   - Prefill performance metrics

3. **Phase 4: Quantization Reuse** (+20% gen)
   - Reduced quantizations from 6→3 per layer
   - Reuse Q8_1 buffers for Q/K/V and Gate/Up

4. **Phase 5: Workspace Buffers** (0% overhead)
   - Persistent buffer allocation
   - Zero allocation during inference
   - Improved latency consistency

## 📊 Performance Details

### Generation Speed
- **Range**: 410-420 tok/s
- **Average**: ~415 tok/s
- **vs llama.cpp**: 95% ✅
- **Status**: Matches llama.cpp performance

### Prefill Speed
- **Range**: 460-470 tok/s
- **Average**: ~465 tok/s
- **vs llama.cpp**: 37%
- **Bottleneck**: Sequential GEMV instead of parallel GEMM

### Why Generation is Fast
- Optimized GEMV kernels for single-token processing
- Minimal orchestration overhead
- Efficient memory access patterns
- Zero allocation overhead
- Async GPU pipeline

### Why Prefill Lags
- Uses sequential GEMV (Matrix × Vector) per token
- llama.cpp uses parallel GEMM (Matrix × Matrix) for all tokens
- 10-20x FLOPS difference for large batches
- **Solution**: Implement GEMM batch processing (Phase 6)

## 🔮 Roadmap

### Phase 6: GEMM Batch Implementation (Planned)
**Goal**: 2-3x prefill speedup (449→1000+ tok/s)

**Required Changes**:
1. Batch embedding lookup - Load all tokens at once
2. rocBLAS GEMM - Replace sequential GEMV with parallel GEMM
3. Batch attention kernel - New C++/HIP kernel for batch attention
4. Batch RoPE/RMSNorm - Apply to all tokens simultaneously

**Expected Results**:
- Prefill: 449-473 → 1000-1100 tok/s (2.2-2.4x improvement)
- Would EXCEED llama.cpp on prefill
- Generation: No change (already optimal)

**Estimated Effort**: 2-3 days of development

## 🧩 Supported Models

- LLaMA (1, 2, 3, 3.1, 3.2, 3.3)
- Gemma-2 (with attention/logit softcapping)
- Qwen (1, 1.5, 2, 2.5)
- Mistral
- SmolLM
- Any GGUF-format model with compatible architecture

## 🔧 Technical Details

### Memory Layout
- Token embeddings: F16 or F32
- Weights: Q4_K, Q6_K, Q8_0, Q8_1, F16, F32
- KV cache: F32 [max_seq, n_head_kv, head_dim] per layer
- Workspace buffers: Allocated once, reused forever

### Quantization Strategy
- Q8_1 buffers reused across operations
- Single quantization for Q/K/V projections
- Single quantization for Gate/Up projections
- Reduced from 6 to 3 quantizations per layer

### GPU Pipeline
- All operations async on default stream
- No synchronization in layer loop
- Single sync before CPU logit readback
- Minimal CPU-GPU overhead

## 🐛 Known Issues

- ASM kernels disabled (cause output corruption, needs debugging)
- Prefill 2.2x slower than llama.cpp (needs GEMM batch)
- Some warnings in main.rs (cosmetic, no impact)

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details

## 🤝 Contributing

Contributions welcome! Areas of interest:
- GEMM batch implementation for prefill
- ASM kernel debugging
- Additional model architecture support
- Performance profiling and optimization

## 📚 References

- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

## 🙏 Acknowledgments

Built on top of GGML's excellent GPU kernels and inspired by llama.cpp's architecture.

---

**Status**: Production Ready (Phase 5 Complete)  
**Performance**: ~415 tok/s generation (matches llama.cpp)  
**Next**: Phase 6 - GEMM batch for 2-3x prefill speedup
