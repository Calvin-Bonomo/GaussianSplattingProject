#pragma once

#include <vector>
#include <functional>
#include <cuda_runtime.h>

/**
 * Public C++ API for the differentiable Gaussian rasterizer.
 * Called from ext.cpp (the pybind11 shim).
 */
namespace CudaRasterizer {

/**
 * Forward rasterization pass.
 *
 * Returns the number of rendered Gaussians (after culling).
 */
int forward(
    // State allocation callback: given a required byte count, returns a
    // pointer to GPU memory.  Callers provide a lambda that allocates from
    // a pre-allocated torch Tensor so the allocation is tracked by autograd.
    std::function<char*(size_t)> geometryBuffer,
    std::function<char*(size_t)> binningBuffer,
    std::function<char*(size_t)> imageBuffer,

    int P,               // number of Gaussians
    int D,               // active SH degree (0–3)
    int M,               // number of SH coefficients per channel

    // Scene / camera
    const float* background,   // (3,) background colour
    int W, int H,              // image width and height (pixels)
    const float* means3D,      // (P, 3) world-space Gaussian centres
    const float* shs,          // (P, M, 3) SH coefficients; NULL if colors_precomp provided
    const float* colors_precomp, // (P, 3) precomputed colours; NULL if using SH
    const float* opacities,    // (P, 1) pre-sigmoid opacities
    const float* scales,       // (P, 3) log-scales (exp will be applied)
    float scale_modifier,      // global scale multiplier
    const float* rotations,    // (P, 4) quaternions (w, x, y, z)
    const float* cov3D_precomp,// (P, 6) precomputed 3D cov upper triangle; NULL if computing from scale/rot
    const float* viewmatrix,   // (16,) column-major world-to-camera 4x4
    const float* projmatrix,   // (16,) column-major full projection 4x4
    const float* cam_pos,      // (3,) camera position in world space
    float tan_fovx, float tan_fovy,

    // Outputs
    int* radii,                // (P,) 2D bounding radius per Gaussian (0 if culled)
    float* out_color,          // (3, H, W) rendered image
    float* out_depth,          // (H, W) rendered depth (or NULL)

    bool prefiltered,
    bool debug
);

/**
 * Backward rasterization pass.  All pointer arguments match the forward
 * signature; additionally receives the gradient of the output colour.
 */
void backward(
    int P, int D, int M, int R,
    const float* background,
    int W, int H,
    const float* means3D,
    const float* shs,
    const float* colors_precomp,
    const float* scales,
    float scale_modifier,
    const float* rotations,
    const float* cov3D_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float* cam_pos,
    float tan_fovx, float tan_fovy,
    const int* radii,
    char* geomBuffer,
    char* binningBuffer,
    char* imageBuffer,
    const float* dL_dpix,      // (3, H, W) gradient of loss w.r.t. rendered colour
    float* dL_dmean2D,         // (P, 3) — z component holds the positional gradient norm
    float* dL_dcov3D,          // (P, 6)
    float* dL_dcolor,          // (P, 3)
    float* dL_dopacity,        // (P, 1)
    float* dL_dmean3D,         // (P, 3)
    float* dL_dsh,             // (P, M, 3)
    float* dL_dscale,          // (P, 3)
    float* dL_drot,            // (P, 4)
    bool debug
);

}  // namespace CudaRasterizer
