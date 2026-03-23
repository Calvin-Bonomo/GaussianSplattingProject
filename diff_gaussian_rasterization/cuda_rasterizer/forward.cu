#include "forward.h"
#include "auxiliary.h"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ============================================================================
// SH evaluation (device, inline)
// ============================================================================

// SH basis constants
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[5] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f,
};
__device__ const float SH_C3[7] = {
    -0.5900435899266435f,
     2.890611442640554f,
    -0.4570457994644658f,
     0.3731763325901154f,
    -0.4570457994644658f,
     1.4453057213903705f,
    -0.5900435899266435f,
};

/**
 * Evaluate SH color for a single Gaussian given view direction.
 * sh is a pointer to (M, 3) floats stored as (R0,G0,B0, R1,G1,B1, ...).
 * Returns the clamped RGB colour and writes the clamp mask to `clamped`.
 */
__device__ float3 computeColorFromSH(
    int idx, int deg, int max_coeffs,
    const float3* means3D,
    const float3 campos,
    const float* shs,
    bool* clamped
) {
    float3 pos = means3D[idx];
    float3 dir = {pos.x - campos.x, pos.y - campos.y, pos.z - campos.z};
    float len = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
    dir = {dir.x/len, dir.y/len, dir.z/len};

    const float* sh = shs + idx * max_coeffs * 3;

    float3 result = {
        SH_C0 * sh[0],
        SH_C0 * sh[1],
        SH_C0 * sh[2],
    };

    if (deg > 0) {
        float x = dir.x, y = dir.y, z = dir.z;
        result.x += -SH_C1 * y * sh[3] + SH_C1 * z * sh[6]  - SH_C1 * x * sh[9];
        result.y += -SH_C1 * y * sh[4] + SH_C1 * z * sh[7]  - SH_C1 * x * sh[10];
        result.z += -SH_C1 * y * sh[5] + SH_C1 * z * sh[8]  - SH_C1 * x * sh[11];

        if (deg > 1) {
            float xx = x*x, yy = y*y, zz = z*z;
            float xy = x*y, yz = y*z, xz = x*z;
            result.x += (SH_C2[0]*xy*sh[12] + SH_C2[1]*yz*sh[15] +
                         SH_C2[2]*(2.f*zz-xx-yy)*sh[18] +
                         SH_C2[3]*xz*sh[21] + SH_C2[4]*(xx-yy)*sh[24]);
            result.y += (SH_C2[0]*xy*sh[13] + SH_C2[1]*yz*sh[16] +
                         SH_C2[2]*(2.f*zz-xx-yy)*sh[19] +
                         SH_C2[3]*xz*sh[22] + SH_C2[4]*(xx-yy)*sh[25]);
            result.z += (SH_C2[0]*xy*sh[14] + SH_C2[1]*yz*sh[17] +
                         SH_C2[2]*(2.f*zz-xx-yy)*sh[20] +
                         SH_C2[3]*xz*sh[23] + SH_C2[4]*(xx-yy)*sh[26]);

            if (deg > 2) {
                result.x += (SH_C3[0]*y*(3.f*xx-yy)*sh[27] +
                             SH_C3[1]*xy*z*sh[30] +
                             SH_C3[2]*y*(4.f*zz-xx-yy)*sh[33] +
                             SH_C3[3]*z*(2.f*zz-3.f*xx-3.f*yy)*sh[36] +
                             SH_C3[4]*x*(4.f*zz-xx-yy)*sh[39] +
                             SH_C3[5]*z*(xx-yy)*sh[42] +
                             SH_C3[6]*x*(xx-3.f*yy)*sh[45]);
                result.y += (SH_C3[0]*y*(3.f*xx-yy)*sh[28] +
                             SH_C3[1]*xy*z*sh[31] +
                             SH_C3[2]*y*(4.f*zz-xx-yy)*sh[34] +
                             SH_C3[3]*z*(2.f*zz-3.f*xx-3.f*yy)*sh[37] +
                             SH_C3[4]*x*(4.f*zz-xx-yy)*sh[40] +
                             SH_C3[5]*z*(xx-yy)*sh[43] +
                             SH_C3[6]*x*(xx-3.f*yy)*sh[46]);
                result.z += (SH_C3[0]*y*(3.f*xx-yy)*sh[29] +
                             SH_C3[1]*xy*z*sh[32] +
                             SH_C3[2]*y*(4.f*zz-xx-yy)*sh[35] +
                             SH_C3[3]*z*(2.f*zz-3.f*xx-3.f*yy)*sh[38] +
                             SH_C3[4]*x*(4.f*zz-xx-yy)*sh[41] +
                             SH_C3[5]*z*(xx-yy)*sh[44] +
                             SH_C3[6]*x*(xx-3.f*yy)*sh[47]);
            }
        }
    }

    // Shift DC so that 0-initialised SH maps to 0.5 gray
    result.x += 0.5f;
    result.y += 0.5f;
    result.z += 0.5f;

    // Clamp to [0, 1] and record clamp mask
    clamped[idx] = (result.x < 0 || result.y < 0 || result.z < 0 ||
                    result.x > 1 || result.y > 1 || result.z > 1);
    result.x = clampf(result.x, 0.f, 1.f);
    result.y = clampf(result.y, 0.f, 1.f);
    result.z = clampf(result.z, 0.f, 1.f);
    return result;
}

// ============================================================================
// preprocessCUDA
// ============================================================================

__global__ void Forward::preprocessCUDA(
    int P, int D, int M,
    const float* orig_points,
    const float3* scales,
    float scale_modifier,
    const float4* rotations,
    const float* opacities,
    const float* shs,
    bool* clamped,
    const float* cov3D_precomp,
    const float* colors_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float* cam_pos,
    int W, int H,
    float tan_fovx, float tan_fovy,
    float focal_x, float focal_y,
    int* radii,
    float2* means2D,
    float* depths,
    float* cov3Ds,
    float* rgb,
    float4* conic_opacity,
    const dim3 grid,
    uint32_t* tiles_touched,
    uint2* tile_min,
    uint2* tile_max,
    bool prefiltered
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;

    // Default: cull this Gaussian
    radii[idx] = 0;
    tiles_touched[idx] = 0;

    // -----------------------------------------------------------------------
    // 1. Transform to clip space and perform frustum culling
    // -----------------------------------------------------------------------
    float4 p_hom = {
        orig_points[3*idx+0],
        orig_points[3*idx+1],
        orig_points[3*idx+2],
        1.0f
    };

    // Transform to clip space: p_proj = projmatrix * p_hom
    // projmatrix is stored column-major
    auto mvmul4 = [&](const float* M4, float4 v) -> float4 {
        return {
            M4[0]*v.x + M4[4]*v.y + M4[8]*v.z  + M4[12]*v.w,
            M4[1]*v.x + M4[5]*v.y + M4[9]*v.z  + M4[13]*v.w,
            M4[2]*v.x + M4[6]*v.y + M4[10]*v.z + M4[14]*v.w,
            M4[3]*v.x + M4[7]*v.y + M4[11]*v.z + M4[15]*v.w,
        };
    };

    float4 p_proj = mvmul4(projmatrix, p_hom);

    // Perspective divide -> NDC
    if (p_proj.w <= 0.0f) return;  // behind camera
    float inv_w = 1.0f / p_proj.w;
    float ndc_x = p_proj.x * inv_w;
    float ndc_y = p_proj.y * inv_w;
    float ndc_z = p_proj.z * inv_w;

    // View-space position (for Jacobian computation)
    float4 p_view = mvmul4(viewmatrix, p_hom);
    if (p_view.z <= 0.2f) return;  // too close / behind near plane

    // Coarse frustum cull: NDC bounds [-1.3, 1.3] with margin
    const float MARGIN = 1.3f;
    if (ndc_x < -MARGIN || ndc_x > MARGIN ||
        ndc_y < -MARGIN || ndc_y > MARGIN) return;

    // -----------------------------------------------------------------------
    // 2. Compute 3D covariance
    // -----------------------------------------------------------------------
    float cov3D_arr[6];
    const float* cov3D = nullptr;
    if (cov3D_precomp != nullptr) {
        cov3D = cov3D_precomp + 6 * idx;
    } else {
        computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3D_arr);
        // Write back for the backward pass
        for (int i = 0; i < 6; i++) cov3Ds[6*idx+i] = cov3D_arr[i];
        cov3D = cov3D_arr;
    }

    // -----------------------------------------------------------------------
    // 3. Project to 2D using EWA Jacobian
    // -----------------------------------------------------------------------
    float3 t = {p_view.x, p_view.y, p_view.z};
    float3 cov2D;
    if (!computeCov2D(t, focal_x, focal_y, viewmatrix, cov3D, cov2D)) return;

    // -----------------------------------------------------------------------
    // 4. Compute conic (inverse 2D covariance)
    // -----------------------------------------------------------------------
    float3 conic;
    if (!computeConic(cov2D.x, cov2D.y, cov2D.z, conic)) return;

    // -----------------------------------------------------------------------
    // 5. Screen-space centre (pixel coords)
    // -----------------------------------------------------------------------
    float2 point_image = {ndc2pix(ndc_x, W), ndc2pix(ndc_y, H)};
    means2D[idx] = point_image;
    depths[idx] = p_view.z;

    // -----------------------------------------------------------------------
    // 6. Evaluate SH color (or use precomputed)
    // -----------------------------------------------------------------------
    float3 color;
    if (colors_precomp != nullptr) {
        color = {colors_precomp[3*idx], colors_precomp[3*idx+1], colors_precomp[3*idx+2]};
        clamped[idx] = false;
    } else {
        color = computeColorFromSH(
            idx, D, M,
            reinterpret_cast<const float3*>(orig_points),
            {cam_pos[0], cam_pos[1], cam_pos[2]},
            shs, clamped
        );
    }
    rgb[3*idx+0] = color.x;
    rgb[3*idx+1] = color.y;
    rgb[3*idx+2] = color.z;

    // -----------------------------------------------------------------------
    // 7. Compute 2D bounding box and count tiles
    // -----------------------------------------------------------------------
    // 3-sigma radius along the largest axis of the 2D Gaussian
    float lambda1, lambda2;
    {
        float a = cov2D.x, b = cov2D.y, c = cov2D.z;
        float trace = a + c;
        float det_val = a * c - b * b;
        float disc = sqrtf(fmaxf(0.f, trace*trace*0.25f - det_val));
        lambda1 = trace * 0.5f + disc;
        lambda2 = trace * 0.5f - disc;
    }
    float my_radius = ceilf(3.f * sqrtf(fmaxf(lambda1, lambda2)));
    radii[idx] = (int)my_radius;

    // Tile bounds (inclusive, in tile coords) — stored for use by duplicateWithKeys
    uint2 rect_min = {
        (uint32_t)fmaxf(0.f, (point_image.x - my_radius) / BLOCK_X),
        (uint32_t)fmaxf(0.f, (point_image.y - my_radius) / BLOCK_Y),
    };
    uint2 rect_max = {
        (uint32_t)fminf((float)grid.x, ceilf((point_image.x + my_radius) / BLOCK_X)),
        (uint32_t)fminf((float)grid.y, ceilf((point_image.y + my_radius) / BLOCK_Y)),
    };
    tile_min[idx] = rect_min;
    tile_max[idx] = rect_max;

    uint32_t n_tiles = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
    tiles_touched[idx] = n_tiles;

    // Write conic + opacity for the renderer
    conic_opacity[idx] = {
        conic.x, conic.y, conic.z,
        1.f / (1.f + __expf(-opacities[idx]))  // sigmoid activation
    };
}

// ============================================================================
// duplicateWithKeys
// ============================================================================

__global__ void Forward::duplicateWithKeys(
    int P,
    const float* depths,
    const uint32_t* offsets,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted,
    const int* radii,
    const uint2* tile_min,
    const uint2* tile_max,
    int grid_x
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;
    if (radii[idx] <= 0) return;

    // Use the SAME bounding box computed in preprocessCUDA to guarantee
    // we write exactly tiles_touched[idx] = (max-min) entries.
    uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
    uint2 rmin = tile_min[idx];
    uint2 rmax = tile_max[idx];

    for (uint32_t ty = rmin.y; ty < rmax.y; ty++) {
        for (uint32_t tx = rmin.x; tx < rmax.x; tx++) {
            uint32_t tile_id = ty * (uint32_t)grid_x + tx;
            gaussian_keys_unsorted[off] = makeKey(tile_id, depths[idx]);
            gaussian_values_unsorted[off] = (uint32_t)idx;
            off++;
        }
    }
}

// ============================================================================
// identifyTileRanges
// ============================================================================

__global__ void Forward::identifyTileRanges(
    int L,
    uint64_t* point_list_keys,
    uint2* ranges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L) return;

    uint32_t cur_tile  = (uint32_t)(point_list_keys[idx] >> 32);
    if (idx == 0) {
        ranges[cur_tile].x = 0;
    } else {
        uint32_t prev_tile = (uint32_t)(point_list_keys[idx - 1] >> 32);
        if (cur_tile != prev_tile) {
            ranges[prev_tile].y = idx;
            ranges[cur_tile].x  = idx;
        }
    }
    if (idx == L - 1) {
        ranges[cur_tile].y = L;
    }
}

// ============================================================================
// renderCUDA  (forward tile renderer)
// ============================================================================

__global__ void Forward::renderCUDA(
    const uint2* __restrict__ ranges,
    const uint32_t* __restrict__ point_list,
    int W, int H,
    const float2* __restrict__ means2D,
    const float* __restrict__ colors,
    const float4* __restrict__ conic_opacity,
    uint32_t* __restrict__ n_contrib,
    const float* __restrict__ bg_color,
    float* __restrict__ out_color,
    float* __restrict__ accum_alpha
) {
    // -----------------------------------------------------------------------
    // Tile / pixel identification
    // -----------------------------------------------------------------------
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
    uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
    uint32_t pix_id = W * pix.y + pix.x;
    float2 pixf = {(float)pix.x, (float)pix.y};

    bool inside = (pix.x < (uint32_t)W && pix.y < (uint32_t)H);
    bool done = !inside;

    // -----------------------------------------------------------------------
    // Tile range in the sorted list
    // -----------------------------------------------------------------------
    uint32_t tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
    uint2 range = ranges[tile_id];
    int rounds = ((range.y - range.x) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // -----------------------------------------------------------------------
    // Shared memory cooperative load buffers
    // -----------------------------------------------------------------------
    __shared__ int      collected_id[BLOCK_SIZE];
    __shared__ float2   collected_xy[BLOCK_SIZE];
    __shared__ float4   collected_conic_opacity[BLOCK_SIZE];
    __shared__ float3   collected_color[BLOCK_SIZE];

    // -----------------------------------------------------------------------
    // Per-pixel accumulators
    // -----------------------------------------------------------------------
    float T = 1.0f;          // transmittance
    uint32_t contributor = 0;
    uint32_t last_contributor = 0;
    float3 C = {0.f, 0.f, 0.f};

    // Linear thread index within the block
    int thread_rank = block.thread_rank();

    for (int i = 0; i < rounds; i++, range.x += BLOCK_SIZE) {
        // ----------------------------------------------------------------
        // Cooperative load: thread t loads element (range.x + t)
        // ----------------------------------------------------------------
        int progress = range.x + thread_rank;
        if (progress < (int)range.y) {
            int gaussian_id = point_list[progress];
            collected_id[thread_rank]             = gaussian_id;
            collected_xy[thread_rank]              = means2D[gaussian_id];
            collected_conic_opacity[thread_rank]   = conic_opacity[gaussian_id];
            collected_color[thread_rank]            = {
                colors[3*gaussian_id],
                colors[3*gaussian_id+1],
                colors[3*gaussian_id+2]
            };
        }
        block.sync();

        // ----------------------------------------------------------------
        // Each thread composites up to BLOCK_SIZE Gaussians onto its pixel
        // ----------------------------------------------------------------
        for (int j = 0; !done && j < fminf(BLOCK_SIZE, range.y - range.x); j++) {
            contributor++;

            float2 xy = collected_xy[j];
            float2 d = {pixf.x - xy.x, pixf.y - xy.y};
            float4 con_o = collected_conic_opacity[j];

            float alpha = computeAlpha(d.x, d.y, con_o.x, con_o.y, con_o.z, con_o.w);
            if (alpha < ALPHA_THRESHOLD) continue;

            float test_T = T * (1.f - alpha);
            if (test_T < T_THRESHOLD) { done = true; break; }

            float3 col = collected_color[j];
            C.x += alpha * T * col.x;
            C.y += alpha * T * col.y;
            C.z += alpha * T * col.z;

            T = test_T;
            last_contributor = contributor;
        }

        if (__syncthreads_or(done)) break;  // entire tile done when all threads are saturated
    }

    // -----------------------------------------------------------------------
    // Write output
    // -----------------------------------------------------------------------
    if (inside) {
        n_contrib[pix_id] = last_contributor;
        accum_alpha[pix_id] = T;

        // Add background colour weighted by remaining transmittance
        out_color[0 * H * W + pix_id] = C.x + T * bg_color[0];
        out_color[1 * H * W + pix_id] = C.y + T * bg_color[1];
        out_color[2 * H * W + pix_id] = C.z + T * bg_color[2];
    }
}
