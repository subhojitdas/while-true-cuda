#include <torch/extension.h>

void add_cuda(torch::Tensor x, torch::Tensor y, float value);

void add_forward(
    torch::Tensor x,
    torch::Tensor y,
    float value
) {
    add_cuda(x, y, value);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_forward", &add_forward, "Custom CUDA add");
}
