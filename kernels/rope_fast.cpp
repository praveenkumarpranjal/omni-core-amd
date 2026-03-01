#include <hip/hip_runtime.h>
__global__ void kernel_rope_f32(
    float * __restrict__ x,
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

    float freq = 1.0f / powf(theta_base, (float)(2 * tid) / (float)head_dim);
    float angle = (float)pos * freq * freq_scale;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    float x0 = head_data[2 * tid];
    float x1 = head_data[2 * tid + 1];

    head_data[2 * tid]     = x0 * cos_a - x1 * sin_a;
    head_data[2 * tid + 1] = x0 * sin_a + x1 * cos_a;
}

extern "C" {
    void bridge_rope_f32(
        float * x,
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
}
