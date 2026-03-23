import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Allow specifying GPU architecture via environment variable, e.g.:
#   TORCH_CUDA_ARCH_LIST="8.0;8.6" pip install -e .
# Defaults to a common modern set if not specified.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0;7.5;8.0;8.6;8.9;9.0")

setup(
    name="diff_gaussian_rasterization",
    packages=["diff_gaussian_rasterization"],
    package_dir={"diff_gaussian_rasterization": ""},
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "cuda_rasterizer/ext.cpp",
            ],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "-allow-unsupported-compiler",
                    "-Xcompiler", "-fvisibility=hidden",
                ],
                "cxx": ["-O3"],
            },
            include_dirs=[
                os.path.join(os.path.dirname(__file__), "cuda_rasterizer"),
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
