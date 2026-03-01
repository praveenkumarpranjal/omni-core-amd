#define GGML_USE_HIP 1

#include "../../../llama.cpp/ggml/include/ggml.h"
#include "../../../llama.cpp/ggml/src/ggml-cuda/common.cuh"
#include "../../../llama.cpp/ggml/src/ggml-cuda/quantize.cuh"

#include "quantize-patched.cu"
