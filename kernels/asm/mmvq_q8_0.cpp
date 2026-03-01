#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <math.h>

struct block_q8_0 {
    _Float16 d;       // delta
    int8_t  qs[32];   // quants
} __attribute__((packed));

extern "C" __global__ void mmvq_q8_0_f32(
    const void* __restrict__ w_ptr,
    const float* __restrict__ x,
    float* __restrict__ y,
    uint32_t k,
    uint32_t m
) {
    uint32_t warp_id = threadIdx.x / 64;
    uint32_t lane_id = threadIdx.x % 64;
    uint32_t row = blockIdx.x * 4 + warp_id;

    if (row >= m) return;

    uint32_t num_blocks = k / 32;
    // Each row of W has num_blocks blocks.
    // The cast to uint8_t* then adding stride avoids any struct alignment padding issues if they exist.
    const uint8_t* w_row_bytes = (const uint8_t*)w_ptr + (row * num_blocks * 34);

    float row_sum = 0.0f;

    // Each lane processes specific blocks
    for (uint32_t b = lane_id; b < num_blocks; b += 64) {
        const float* x_blk = x + b * 32;
        
        // Quantize X dynamically for this block
        float max_abs = 0.0f;
        for (int i = 0; i < 32; i++) {
            float val = fabsf(x_blk[i]);
            if (val > max_abs) max_abs = val;
        }

        float x_scale = max_abs / 127.0f;
        float inv_x_scale = (x_scale == 0.0f) ? 0.0f : (1.0f / x_scale);

        int8_t x_qs[32];
        for (int i = 0; i < 32; i++) {
            float val = x_blk[i] * inv_x_scale;
            int q = (int)roundf(val);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            x_qs[i] = (int8_t)q;
        }

        const uint8_t* w_blk_bytes = w_row_bytes + b * 34;
        
        // Read f16 W scale dynamically using strict explicit pointer math
        uint16_t w_scale_f16_bits = *((const uint16_t*)(w_blk_bytes));
        _Float16 w_scale_f16;
        memcpy(&w_scale_f16, &w_scale_f16_bits, 2);
        float w_scale = (float)w_scale_f16;
        
        // Read W quantized values
        const int8_t* w_qs = (const int8_t*)(w_blk_bytes + 2);

        int sumi = 0;
        const int32_t* wx_ptr = (const int32_t*)w_qs;
        const int32_t* xx_ptr = (const int32_t*)x_qs;
        
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            sumi = __builtin_amdgcn_sdot4(wx_ptr[i], xx_ptr[i], sumi, false);
        }

        row_sum += (float)sumi * w_scale * x_scale;
    }

    // Warp-level butterfly reduction
    // Using __shfl_xor for the standard butterfly pattern
    #pragma unroll
    for (int offset = 32; offset > 0; offset /= 2) {
        row_sum += __shfl_xor(row_sum, offset, 64);
    }

    if (lane_id == 0) {
        y[row] = row_sum;
    }
}
