from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name="cust_add_ext",
    ext_modules=[
        CUDAExtension(
            name="cust_add_ext",
            sources=[
                "cust_add.cpp",
                "cust_add_cuda.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)