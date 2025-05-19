import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
std::string hello_cuda() {
    return "hello cuda";
}
"""

cudmod = load_inline(
    name="cudmod",
    cpp_sources=[cpp_source],
    functions=['hello_cuda'],
    verbose=True,
    build_directory='./tmp',
)

print(cudmod.hello_cuda())
