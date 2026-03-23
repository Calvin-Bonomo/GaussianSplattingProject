#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace Backward {

/**
 * Backward pass through the tile renderer.
 * Replays the forward accumulation in reverse tile order to compute gradients.
 *
 * Grid: (ceil(W/BLOCK_X), ceil(H/BLOCK_Y))
 * Block: (BLOCK_X, BLOCK_Y)
 */
__global__ void renderBackwardCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float* __restrict__ bg_color,
    const float2* __restrict__ means2D,
    const float4* __restrict__ conic_opacity,
    const float* __restrict__ colors,
    const float* __restrict__ accum_alpha,   // final transmittance per pixel (from fwd)
    const uint32_t* __restrict__ n_contrib,  // contributor count per pixel (from fwd)
    const float* __restrict__ dL_dpix,       // (3, H*W) gradient of loss w.r.t. image
    float3* __restrict__ dL_dmean2D,         // (P,) gradient w.r.t. 2D position
    float4* __restrict__ dL_dconic_opacity,  // (P,) gradient w.r.t. conic + opacity
    float3* __restrict__ dL_dcolors          // (P,) gradient w.r.t. colour
);

/**
 * Backward pass through the preprocessing step.
 * Converts gradients from 2D to 3D: mean2D->mean3D, cov2D->cov3D/scale/rot/sh.
 *
 * Launch with: <<<(P+255)/256, 256>>>
 */
__global__ void preprocessBackwardCUDA(
    int P, int D, int M,
    const float3* means3D,
    const int* radii,
    const float* shs,
    const bool* clamped,
    const float3* scales,
    const float4* rotations,
    float scale_modifier,
    const float* cov3Ds,
    const float* viewmatrix,
    const float* projmatrix,
    float focal_x, float focal_y,
    float tan_fovx, float tan_fovy,
    int W, int H,
    const float3* dL_dmean2D,
    const float* dL_dconics,    // (P, 3) — from dL_dconic_opacity
    float* dL_dopacity,         // (P,) — from dL_dconic_opacity
    float* dL_dcolor,           // (P, 3)
    float* dL_dmean3D,          // (P, 3) output
    float* dL_dcov3D,           // (P, 6) output
    float* dL_dsh,              // (P, M, 3) output
    float3* dL_dscale,          // (P,) output
    float4* dL_drot             // (P,) output
);

}  // namespace Backward
