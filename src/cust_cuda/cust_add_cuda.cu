#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


__global__ void add_kernel(
    const float* x,
    float* y,
    float value,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] + value;
    }
}

void add_cuda(
    torch::Tensor x,
    torch::Tensor y,
    float value
) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);

    int n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    add_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        value,
        n
    );
}
