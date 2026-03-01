#define GGML_USE_HIP 1

#include "../../../llama.cpp/ggml/include/ggml.h"
#include "../../../llama.cpp/ggml/src/ggml-cuda/common.cuh"
#include "../../../llama.cpp/ggml/src/ggml-cuda/mmvq.cuh"

// Include the patched file which changed `static __global__` to `__global__`
#include "mmvq-patched.cu"

// Explicitly instantiate Q4_K GEMV for batch size 1
template __global__ void mul_mat_vec_q<GGML_TYPE_Q6_K, 1, false, false>(
    const void * __restrict__ vx, 
    const void * __restrict__ vy, 
    const int32_t * __restrict__ ids, 
    const ggml_cuda_mm_fusion_args_device fusion, 
    float * __restrict__ dst,
    const uint32_t ncols_x, 
    const uint3 nchannels_y, 
    const uint32_t stride_row_x, 
    const uint32_t stride_col_y, 
    const uint32_t stride_col_dst,
    const uint3 channel_ratio, 
    const uint32_t stride_channel_x, 
    const uint32_t stride_channel_y, 
    const uint32_t stride_channel_dst,
    const uint3 sample_ratio, 
    const uint32_t stride_sample_x, 
    const uint32_t stride_sample_y, 
    const uint32_t stride_sample_dst, 
    const uint32_t ids_stride);
