# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D Gaussian Splatting (3DGS) implementation for real-time scene reconstruction and rendering. Based on Kerbl et al. (2023): instead of ray-tracing geometry, rasterize tens-to-hundreds of thousands of learnable Gaussians initialized from a COLMAP point cloud.

## Commands

**Set up environment:**
```bash
conda env create -f environment.yml
conda activate gaussian_splatting
```

**Train (with built-in viewer):**
```bash
python run.py -s <dataset_path> -o <output_path>
```

**Train without viewer:**
```bash
python run.py -s <dataset_path> -o <output_path> --no-viewer
```

**Run tests:**
```bash
pytest
# Single test:
pytest tests/test_gaussian_model.py::test_save_load_ply -v
```

## Architecture

### Data flow

```
COLMAP sparse/ (cameras.bin, images.bin, points3D.bin)
  → Scene → Camera list + BasicPointCloud
    → GaussianModel.create_from_pcd() → N learnable Gaussians
      → train.py loop: render → L1+SSIM loss → backward → densify/prune → (optional viewer)
        → GaussianModel.save_ply() → point_cloud/iteration_N/point_cloud.ply
```

### Package structure

**`gaussian_rasterizer/`** — CUDA rasterizer loaded via `torch.utils.cpp_extension.load()` (JIT compiled on first run, cached afterward). No separate build step required. The Python `render()` call in `gaussian_splatting/renderer.py` dispatches through `ext.cpp` into `rasterizer_impl.cu`. Tile size is 16×16; Gaussians are sorted per-tile by depth using CUB DeviceRadixSort with key `(tile_id << 32 | depth_uint)`.

**`gaussian_splatting/scene/gaussian_model.py`** — `GaussianModel`: all learnable parameters (`_xyz`, `_scaling`, `_rotation`, `_opacity`, `_sh_band0`, `_sh_bands_rest`), the Adam optimizer with 6 separate parameter groups, and adaptive density control.

**`gaussian_splatting/scene/scene.py`** — Loads COLMAP or Blender-synthetic datasets, normalizes camera positions to a unit sphere, splits train/test cameras.

**`gaussian_splatting/renderer.py`** — Thin wrapper around the CUDA rasterizer. Returns `{"render", "viewspace_points", "visibility_filter", "radii"}`.

### Adaptive density control

Runs every 100 iterations (iters 500–15,000):
- **Clone**: high positional gradient + small scale → duplicate in place
- **Split**: high positional gradient + large scale → replace with 2 smaller children
- **Prune**: low opacity or oversized 2D radius → remove
- **Opacity reset**: every 3,000 iters, reset all opacities near-zero to restart the pruning cycle

The Adam moment buffers must be manually resized in sync with Gaussian tensors — see `_prune_optimizer()` and `densification_postfix()` in `gaussian_model.py`.


### SH degree warmup

`GaussianModel.oneupSHdegree()` is called every 1,000 iterations, incrementally unlocking SH bands 0→3. Training starts with DC color only (degree 0); higher bands add view-dependent appearance.

## Environment

- Python 3.13, CUDA 12.8, Conda (see `README.md`)
- NVIDIA GPU, compute capability ≥ 7.0
- The CUDA rasterizer is JIT-compiled by PyTorch on first use — no manual build step needed, but the first run will be slow while it compiles

## Dataset format

```
<dataset>/
  images/       # Training JPEGs
  sparse/0/     # COLMAP output: cameras.bin, images.bin, points3D.bin
```

Blender/NeRF-synthetic (`transforms_train.json`) is also supported via `dataset_readers.py`.
