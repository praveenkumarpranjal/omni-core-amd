#define GGML_USE_HIP 1
#define FLASH_ATTN_AVAILABLE 1

#include "../../../llama.cpp/ggml/include/ggml.h"
#include "../../../llama.cpp/ggml/src/ggml-cuda/common.cuh"
#include "../../../llama.cpp/ggml/src/ggml-cuda/fattn-common.cuh"

// Include the patched header where `static __global__` was replaced with `__global__`
#include "fattn-vec-patched.cuh"

// Explicitly instantiate the kernels for Head Size 64 and 128 (most common for Qwen/Gemma/GPTOSS)
template __global__ void flash_attn_ext_vec<64, 1, GGML_TYPE_F16, GGML_TYPE_F16, false>(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const char * __restrict__ mask, const char * __restrict__ sinks, const int  * __restrict__ KV_max,
        float * __restrict__ dst, float2 * __restrict__ dst_meta,
        const float scale, const float max_bias, const float m0, const float m1,
        const uint32_t n_head_log2, const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33);

template __global__ void flash_attn_ext_vec<128, 1, GGML_TYPE_F16, GGML_TYPE_F16, false>(
        const char * __restrict__ Q, const char * __restrict__ K, const char * __restrict__ V,
        const char * __restrict__ mask, const char * __restrict__ sinks, const int  * __restrict__ KV_max,
        float * __restrict__ dst, float2 * __restrict__ dst_meta,
        const float scale, const float max_bias, const float m0, const float m1,
        const uint32_t n_head_log2, const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33);
