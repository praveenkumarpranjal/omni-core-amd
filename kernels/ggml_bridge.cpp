// ggml_bridge.cpp — Complete C-ABI GPU kernel bridge for Rust FFI
// Wraps llama.cpp's GEMV/quantization kernels + provides standalone kernels
// for RMSNorm, RoPE, SiLU, embedding lookup, and simple attention.
//
// Compiled with: hipcc -shared -fPIC -O3 --offload-arch=gfx942
//   -I../llama.cpp/ggml/include -I../llama.cpp/ggml/src/ggml-cuda
//   -I../llama.cpp/ggml/src -L../llama.cpp/build/bin -lggml-hip -lggml-base
//   -o libs/libggml_bridge.so

#define GGML_USE_HIP 1

#include "../../llama.cpp/ggml/include/ggml.h"
#include "../../llama.cpp/ggml/src/ggml-cuda/common.cuh"
#include "../../llama.cpp/ggml/src/ggml-cuda/quantize.cuh"
#include "../../llama.cpp/ggml/src/ggml-cuda/mmvq.cuh"

// Include the patched MMVQ file for mul_mat_vec_q_switch_type
#include "asm/mmvq-patched.cu"

// ============================================================================
// Standalone GPU kernels (not from llama.cpp)
// ============================================================================

// --- RMSNorm kernel ---
// Computes: dst[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
__global__ void kernel_rms_norm_f32(
    const float * __restrict__ x,
    const float * __restrict__ weight,
    float * __restrict__ dst,
    int ncols,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const float * x_row = x + (int64_t)row * ncols;
    float * dst_row = dst + (int64_t)row * ncols;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < ncols; i += block_size) {
        float val = x_row[i];
        sum_sq += val * val;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor(sum_sq, offset);
    }

    // Cross-warp reduction via shared memory
    __shared__ float shared[32];
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    if (lane == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane < (block_size + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sum_sq += __shfl_xor(sum_sq, offset);
        }
    }

    // Broadcast final sum
    __syncthreads();
    if (tid == 0) shared[0] = sum_sq;
    __syncthreads();
    sum_sq = shared[0];

    float scale = rsqrtf(sum_sq / (float)ncols + eps);

    // Apply normalization and weight
    for (int i = tid; i < ncols; i += block_size) {
        dst_row[i] = x_row[i] * scale * weight[i];
    }
}

// --- SiLU activation with gate multiplication ---
// Computes: dst[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
__global__ void kernel_silu_mul_f32(
    const float * __restrict__ gate,
    const float * __restrict__ up,
    float * __restrict__ dst,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        dst[i] = silu_g * up[i];
    }
}

// --- Bias addition ---
__global__ void kernel_add_bias_f32(float* __restrict__ x, const float* __restrict__ bias, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += bias[i];
    }
}

// --- RoPE NeoX kernel ---
// Applies rotary position embeddings (NeoX style: first half cos, second half sin)
__global__ void kernel_rope_neox_f32(
    float * __restrict__ x,   // [n_head, head_dim] — modified in place
    int head_dim,
    int n_heads,
    int pos,
    float theta_base,
    float freq_scale)
{
    int head = blockIdx.x;
    int tid = threadIdx.x;
    int half_dim = head_dim / 2;

    if (tid >= half_dim) return;

    float * head_data = x + head * head_dim;

    // Compute the rotation angle for this dimension
    float freq = 1.0f / powf(theta_base, (float)(2 * tid) / (float)head_dim);
    float angle = (float)pos * freq * freq_scale;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    // NeoX style: split in half
    float x0 = head_data[tid];
    float x1 = head_data[tid + half_dim];

    head_data[tid]            = x0 * cos_a - x1 * sin_a;
    head_data[tid + half_dim] = x0 * sin_a + x1 * cos_a;
}

// --- Standard Interleaved RoPE kernel ---
// Applies rotary position embeddings (Interleaved style, for Llama models)
__global__ void kernel_rope_f32(
    float * __restrict__ x,   // [n_head, head_dim] — modified in place
    int head_dim,
    int n_heads,
    int pos,
    float theta_base,
    float freq_scale)
{
    int head = blockIdx.x;
    int tid = threadIdx.x; // Thread handles one PAIR of elements (so tid goes up to head_dim/2 - 1)
    int half_dim = head_dim / 2;

    if (tid >= half_dim) return;

    float * head_data = x + head * head_dim;

    // Compute the rotation angle for this dimension pair
    float freq = 1.0f / powf(theta_base, (float)(2 * tid) / (float)head_dim);
    float angle = (float)pos * freq * freq_scale;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    // Interleaved style: adjacent pairs
    float x0 = head_data[2 * tid];
    float x1 = head_data[2 * tid + 1];

    head_data[2 * tid]     = x0 * cos_a - x1 * sin_a;
    head_data[2 * tid + 1] = x0 * sin_a + x1 * cos_a;
}

// --- Simple dot-product attention (single query, batch size 1) ---
// For each head: score = softmax(Q @ K^T / sqrt(d)) @ V
// K_cache: [max_seq, n_head_kv, head_dim]
// V_cache: [max_seq, n_head_kv, head_dim]
__global__ void kernel_attention_f32(
    const float * __restrict__ Q,        // [n_head, head_dim]
    const float * __restrict__ K_cache,  // [max_seq, n_head_kv, head_dim]
    const float * __restrict__ V_cache,  // [max_seq, n_head_kv, head_dim]
    float * __restrict__ dst,            // [n_head, head_dim]
    int head_dim,
    int n_head,
    int n_head_kv,
    int seq_len,          // current valid sequence length (positions 0..seq_len-1)
    int max_seq,
    float scale,
    float softcap)          // 1/sqrt(head_dim), and optional cap
{
    int head = blockIdx.x;
    int tid = threadIdx.x;

    // GQA: map query head to KV head
    int kv_head = head / (n_head / n_head_kv);

    const float * q_head = Q + head * head_dim;
    float * dst_head = dst + head * head_dim;

    // We use shared memory for attention scores
    extern __shared__ float shared_mem[];
    float * scores = shared_mem;           // [seq_len]
    float * v_accum = shared_mem + seq_len; // [head_dim] accumulator

    // Step 1: Compute Q @ K^T for all positions (each thread handles some positions)
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        const float * k_vec = K_cache + pos * n_head_kv * head_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_head[d] * k_vec[d];
        }
        float score = dot * scale;
        if (softcap > 0.0f) {
            score = softcap * tanhf(score / softcap);
        }
        scores[pos] = score;
    }
    __syncthreads();

    // Step 2: Softmax over scores
    // Find max (reduction)
    float max_val = -1e30f;
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        max_val = fmaxf(max_val, scores[pos]);
    }
    // Warp reduce max
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor(max_val, offset));
    }
    __shared__ float shared_max[32];
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;
    if (lane == 0) shared_max[warp_id] = max_val;
    __syncthreads();
    if (warp_id == 0) {
        max_val = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared_max[lane] : -1e30f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_xor(max_val, offset));
        }
    }
    __syncthreads();
    if (tid == 0) shared_max[0] = max_val;
    __syncthreads();
    max_val = shared_max[0];

    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        float e = expf(scores[pos] - max_val);
        scores[pos] = e;
        sum_exp += e;
    }
    // Reduce sum
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_exp += __shfl_xor(sum_exp, offset);
    }
    __shared__ float shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = sum_exp;
    __syncthreads();
    if (warp_id == 0) {
        sum_exp = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sum_exp += __shfl_xor(sum_exp, offset);
        }
    }
    __syncthreads();
    if (tid == 0) shared_sum[0] = sum_exp;
    __syncthreads();
    sum_exp = shared_sum[0];

    float inv_sum = 1.0f / sum_exp;
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        scores[pos] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Weighted sum: dst = scores @ V
    // Each thread handles one dimension of head_dim
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int pos = 0; pos < seq_len; pos++) {
            const float * v_vec = V_cache + pos * n_head_kv * head_dim + kv_head * head_dim;
            acc += scores[pos] * v_vec[d];
        }
        dst_head[d] = acc;
    }
}

// --- Vector add kernel (for residual connections) ---
__global__ void kernel_add_f32(
    const float * __restrict__ a,
    const float * __restrict__ b,
    float * __restrict__ dst,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = a[i] + b[i];
    }
}

// --- Softcapping kernel (Gemma-2) ---
__global__ void kernel_softcap_f32(float * dst, int n, float cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = dst[i];
        dst[i] = cap * tanhf(x / cap);
    }
}


// ============================================================================
// C-ABI exports for Rust FFI
// ============================================================================

extern "C" {

// Quantize FP32 -> Q8_1 on GPU
void bridge_quantize_q8_1(
    const float * x,
    void * vy,
    int64_t ne0,
    hipStream_t stream)
{
    quantize_row_q8_1_cuda(
        x, nullptr, vy, GGML_TYPE_Q4_K,
        ne0, ne0, ne0, ne0, ne0, 1, 1, 1, stream
    );
}

// Quantized matrix-vector multiply
void bridge_mul_mat_vec_q(
    const void * vx,
    int type_x,
    const void * vy,
    float * dst,
    int ncols_x,
    int nrows_x,
    hipStream_t stream)
{
    ggml_type qtype = static_cast<ggml_type>(type_x);
    int qk = ggml_blck_size(qtype);
    int stride_row_x = ncols_x / qk;
    int stride_col_y = ncols_x / QK8_1;

    ggml_cuda_mm_fusion_args_device fusion{};

    mul_mat_vec_q_switch_type(
        vx, qtype, vy, nullptr, fusion, dst,
        ncols_x, nrows_x, 1, stride_row_x, stride_col_y, nrows_x,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream
    );
}

// RMSNorm: dst = rmsnorm(x) * weight
void bridge_rms_norm_f32(
    const float * x,
    const float * weight,
    float * dst,
    int ncols,
    int nrows,
    float eps,
    hipStream_t stream)
{
    int block_size = (ncols < 1024) ? 256 : 1024;
    dim3 grid(nrows);
    dim3 block(block_size);
    kernel_rms_norm_f32<<<grid, block, 0, stream>>>(x, weight, dst, ncols, eps);
}

// SiLU gate multiplication: dst = silu(gate) * up
void bridge_silu_mul_f32(const float* gate, const float* up, float* dst, int n, void* stream_ptr) {
        hipStream_t stream = (hipStream_t)stream_ptr;
        int blocks = (n + 255) / 256;
        kernel_silu_mul_f32<<<blocks, 256, 0, stream>>>(gate, up, dst, n);
    }

    void bridge_add_bias_f32(float* x, const float* bias, int n, void* stream_ptr) {
        hipStream_t stream = (hipStream_t)stream_ptr;
        int blocks = (n + 255) / 256;
        kernel_add_bias_f32<<<blocks, 256, 0, stream>>>(x, bias, n);
    }

// RoPE NeoX: apply rotary embeddings in-place
void bridge_rope_neox_f32(
    float * x,       // [n_heads, head_dim]
    int head_dim,
    int n_heads,
    int pos,
    float theta_base,
    float freq_scale,
    hipStream_t stream)
{
    int half_dim = head_dim / 2;
    kernel_rope_neox_f32<<<n_heads, half_dim, 0, stream>>>(
        x, head_dim, n_heads, pos, theta_base, freq_scale
    );
}

// RoPE Standard (Interleaved): apply rotary embeddings in-place (Llama)
void bridge_rope_f32(
    float * x,       // [n_heads, head_dim]
    int head_dim,
    int n_heads,
    int pos,
    float theta_base,
    float freq_scale,
    hipStream_t stream)
{
    int half_dim = head_dim / 2;
    kernel_rope_f32<<<n_heads, half_dim, 0, stream>>>(
        x, head_dim, n_heads, pos, theta_base, freq_scale
    );
}

// Simple dot-product attention (batch size 1)
void bridge_attention_f32(
    const float * Q,        // [n_head, head_dim]
    const float * K_cache,  // [max_seq, n_head_kv, head_dim]
    const float * V_cache,  // [max_seq, n_head_kv, head_dim]
    float * dst,            // [n_head, head_dim]
    int head_dim,
    int n_head,
    int n_head_kv,
    int seq_len,
    int max_seq,
    float scale,
    float softcap,
    hipStream_t stream)
{
    // Shared memory: scores[seq_len] + v_accum[head_dim]
    int shared_size = (seq_len + head_dim) * sizeof(float);
    int block_size = 64; // AMD wavefront size
    kernel_attention_f32<<<n_head, block_size, shared_size, stream>>>(
        Q, K_cache, V_cache, dst,
        head_dim, n_head, n_head_kv, seq_len, max_seq, scale, softcap
    );
}

// Vector addition (for residual connections)
void bridge_add_f32(
    const float * a,
    const float * b,
    float * dst,
    int n,
    hipStream_t stream)
{
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    kernel_add_f32<<<grid_size, block_size, 0, stream>>>(a, b, dst, n);
}

// hipMemcpy wrapper for embedding lookup (D2D copy with offset)
void bridge_embed_lookup(
    const void * embeddings,   // device ptr to full embedding table
    void * dst,                // device ptr to output buffer
    int token_id,
    int d_model,               // embedding dimension
    int type_size,             // bytes per element (4 for F32, 2 for F16)
    hipStream_t stream)
{
    const char * src = (const char *)embeddings + (int64_t)token_id * d_model * type_size;
    hipMemcpyAsync(dst, src, d_model * type_size, hipMemcpyDeviceToDevice, stream);
}

// F16 -> F32 conversion kernel
__global__ void kernel_f16_to_f32(const __half * src, float * dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __half2float(src[i]);
    }
}

// Convert F16 buffer to F32 on GPU
void bridge_f16_to_f32(
    const void * src_f16,
    float * dst_f32,
    int n,
    hipStream_t stream)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    kernel_f16_to_f32<<<grid, block, 0, stream>>>((const __half *)src_f16, dst_f32, n);
}

// KV cache write: copy K/V vectors into cache at given position
void bridge_kv_cache_write(
    const float * k_proj,              // [n_head_kv, head_dim] FP32
    const float * v_proj,              // [n_head_kv, head_dim] FP32
    void * k_cache,                    // [max_seq, n_head_kv, head_dim] FP32
    void * v_cache,                    // [max_seq, n_head_kv, head_dim] FP32
    int pos,
    int n_head_kv,
    int head_dim,
    hipStream_t stream)
{
    int kv_size = n_head_kv * head_dim * sizeof(float);
    char * k_dst = (char *)k_cache + (int64_t)pos * kv_size;
    char * v_dst = (char *)v_cache + (int64_t)pos * kv_size;
    hipMemcpyAsync(k_dst, k_proj, kv_size, hipMemcpyDeviceToDevice, stream);
    hipMemcpyAsync(v_dst, v_proj, kv_size, hipMemcpyDeviceToDevice, stream);
}

// hipDeviceSynchronize wrapper
int bridge_sync() {
    return (int)hipDeviceSynchronize();
}

// Logit softcapping (Gemma-2)
void bridge_softcap_f32(float * dst, int n, float cap, hipStream_t stream) {
    if (cap <= 0.0f) return;
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    kernel_softcap_f32<<<grid_size, block_size, 0, stream>>>(dst, n, cap);
}

} // extern "C"
