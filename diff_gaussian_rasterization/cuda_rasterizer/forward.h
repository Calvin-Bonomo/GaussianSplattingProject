#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace Forward {

/**
 * Preprocessing kernel: projects Gaussians to 2D, evaluates SH colors,
 * computes conics, and counts tiles touched by each Gaussian.
 *
 * Launch with: <<<(P+255)/256, 256>>>
 */
__global__ void preprocessCUDA(
    int P,                       // number of Gaussians
    int D,                       // active SH degree
    int M,                       // SH coefficients per channel
    const float* orig_points,    // (P, 3) world-space positions
    const float3* scales,        // (P,) per-axis scales (already exp()-activated)
    float scale_modifier,
    const float4* rotations,     // (P,) quaternions (w, x, y, z)
    const float* opacities,      // (P,) pre-sigmoid opacities
    const float* shs,            // (P, M, 3) SH coefficients; may be NULL
    bool* clamped,               // (P,) output: was this Gaussian's colour clamped?
    const float* cov3D_precomp,  // (P, 6) precomputed 3D cov; may be NULL
    const float* colors_precomp, // (P, 3) precomputed colours; may be NULL
    const float* viewmatrix,     // (16,) column-major world-to-camera
    const float* projmatrix,     // (16,) column-major full projection
    const float* cam_pos,        // (3,) camera position in world space
    int W, int H,
    float tan_fovx, float tan_fovy,
    float focal_x, float focal_y,
    int* radii,                  // (P,) output: 2D radius (0 = culled)
    float2* means2D,             // (P,) output: projected screen-space centre
    float* depths,               // (P,) output: view-space depth
    float* cov3Ds,               // (P, 6) output: world-space cov upper triangle
    float* rgb,                  // (P, 3) output: evaluated colour
    float4* conic_opacity,       // (P,) output: (a, b, c, opacity)
    const dim3 grid,             // tile grid dimensions
    uint32_t* tiles_touched,     // (P,) output: tiles overlapped
    uint2* tile_min,             // (P,) output: tile bounding box min
    uint2* tile_max,             // (P,) output: tile bounding box max
    bool prefiltered
);

/**
 * Kernel to scatter (tile_id, depth) sort keys into the binning buffers.
 * One thread per Gaussian; each Gaussian writes tiles_touched[i] entries.
 *
 * Launch with: <<<(P+255)/256, 256>>>
 */
__global__ void duplicateWithKeys(
    int P,
    const float* depths,
    const uint32_t* offsets,     // prefix-sum output (point_offsets)
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted,
    const int* radii,
    const uint2* tile_min,       // precomputed tile bounding box min from preprocessCUDA
    const uint2* tile_max,       // precomputed tile bounding box max from preprocessCUDA
    int grid_x                   // tile grid x dimension (for tile_id computation)
);

/**
 * After sorting, identify the start and end index in point_list for
 * each tile.  One thread per entry in the sorted list.
 *
 * Launch with: <<<(num_rendered+255)/256, 256>>>
 */
__global__ void identifyTileRanges(
    int L,                       // total number of (Gaussian, tile) pairs
    uint64_t* point_list_keys,   // sorted keys
    uint2* ranges                // (num_tiles,) output: [start, end)
);

/**
 * Main tile renderer forward pass.
 *
 * Grid: (ceil(W/BLOCK_X), ceil(H/BLOCK_Y))
 * Block: (BLOCK_X, BLOCK_Y) = (16, 16)
 */
__global__ void renderCUDA(
    const uint2* __restrict__ ranges,        // tile ranges
    const uint32_t* __restrict__ point_list, // sorted Gaussian indices
    int W, int H,
    const float2* __restrict__ means2D,
    const float* __restrict__ colors,        // (P, 3) RGB per Gaussian
    const float4* __restrict__ conic_opacity,// (P,) (a,b,c,opacity)
    uint32_t* __restrict__ n_contrib,        // (H*W,) output: contributor count per pixel
    const float* __restrict__ bg_color,      // (3,) background
    float* __restrict__ out_color,           // (3, H*W) output image
    float* __restrict__ accum_alpha          // (H*W,) output: final transmittance
);

}  // namespace Forward
