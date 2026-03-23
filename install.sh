#!/usr/bin/env bash
# Install all dependencies for the Gaussian Splatting project.
#
# torch CUDA extensions must be built with --no-build-isolation so that
# the build process can import the already-installed torch (which ships
# its own CUDA headers and the cpp_extension build system).

set -e

echo "==> Installing Python dependencies..."
pip install -r requirements.txt

echo "==> Building diff_gaussian_rasterization CUDA extension..."
pip install --no-build-isolation -e ./diff_gaussian_rasterization

echo "==> Done. Run training with:"
echo "    python train.py -s <dataset_path> -m <output_path>"
