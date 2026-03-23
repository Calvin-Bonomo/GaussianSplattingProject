#include "rasterizer_impl.h"
#include "rasterizer.h"
#include "forward.h"
#include "backward.h"
#include "auxiliary.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <functional>

// Debug helper: sync and check for errors after every step.
// Set GAUSS_DEBUG_SYNC=1 in the environment to enable.
static bool _debug_sync_enabled() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("GAUSS_DEBUG_SYNC"); v = (e && e[0]=='1') ? 1 : 0; }
    return v == 1;
}
#define CHECK_CUDA(label) do { \
    if (_debug_sync_enabled()) { \
        cudaError_t _e = cudaDeviceSynchronize(); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(_e)); \
            throw std::runtime_error(std::string("CUDA error at ") + label); \
        } \
    } \
} while(0)

namespace CudaRasterizer {

// ============================================================================
// State buffer constructors
// ============================================================================

GeometryState GeometryState::fromChunk(char*& chunk, size_t P) {
    GeometryState geom;
    geom.depths         = obtainPtr<float>(chunk, P);
    geom.cov3D          = obtainPtr<float>(chunk, P * 6);
    geom.means2D        = obtainPtr<float2>(chunk, P);
    geom.rgb            = obtainPtr<float>(chunk, P * 3);
    geom.conic_opacity  = obtainPtr<float4>(chunk, P);
    geom.clamped        = obtainPtr<bool>(chunk, P);
    geom.tiles_touched  = obtainPtr<uint32_t>(chunk, P);
    geom.point_offsets  = obtainPtr<uint32_t>(chunk, P);
    geom.tile_min       = obtainPtr<uint2>(chunk, P);
    geom.tile_max       = obtainPtr<uint2>(chunk, P);
    // CUB temporary buffer for prefix sum (will be sized dynamically)
    cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.point_offsets, P);
    geom.scanning_space = obtainPtr<char>(chunk, geom.scan_size);
    return geom;
}

BinningState BinningState::fromChunk(char*& chunk, size_t P) {
    BinningState binning;
    binning.point_list_keys_unsorted   = obtainPtr<uint64_t>(chunk, P);
    binning.point_list_keys            = obtainPtr<uint64_t>(chunk, P);
    binning.point_list_unsorted        = obtainPtr<uint32_t>(chunk, P);
    binning.point_list                 = obtainPtr<uint32_t>(chunk, P);
    // CUB sort scratch
    cub::DeviceRadixSort::SortPairs(
        nullptr, binning.sorting_size,
        binning.point_list_keys_unsorted, binning.point_list_keys,
        binning.point_list_unsorted,      binning.point_list,
        P);
    binning.list_sorting_space = obtainPtr<char>(chunk, binning.sorting_size);
    return binning;
}

ImageState ImageState::fromChunk(char*& chunk, size_t N, size_t W, size_t H) {
    ImageState img;
    img.ranges      = obtainPtr<uint2>(chunk, N);     // N = num_tiles
    img.n_contrib   = obtainPtr<uint32_t>(chunk, W * H);
    img.accum_alpha = obtainPtr<float>(chunk, W * H);
    return img;
}

// Helper: compute how many bits we actually need to sort (avoids sorting
// all 64 bits when most tile IDs are small).
static int numBitsForSorting(int num_tiles) {
    int bits = 0;
    while ((1 << bits) < num_tiles) bits++;
    return bits + 32;  // lower 32 bits hold the depth key
}

// ============================================================================
// forward
// ============================================================================

int forward(
    std::function<char*(size_t)> geometryBuffer,
    std::function<char*(size_t)> binningBuffer,
    std::function<char*(size_t)> imageBuffer,
    int P, int D, int M,
    const float* background,
    int W, int H,
    const float* means3D,
    const float* shs,
    const float* colors_precomp,
    const float* opacities,
    const float* scales,
    float scale_modifier,
    const float* rotations,
    const float* cov3D_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float* cam_pos,
    float tan_fovx, float tan_fovy,
    int* radii,
    float* out_color,
    float* out_depth,
    bool prefiltered,
    bool debug
) {
    if (P == 0) return 0;

    static int fwd_call = 0;
    ++fwd_call;
    if (_debug_sync_enabled()) { fprintf(stderr, "[DBG] >>> forward call #%d P=%d\n", fwd_call, P); fflush(stderr); }

    float focal_y = H / (2.f * tan_fovy);
    float focal_x = W / (2.f * tan_fovx);

    // Tile grid
    dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);
    int num_tiles = tile_grid.x * tile_grid.y;

    // -----------------------------------------------------------------------
    // Allocate state buffers
    // -----------------------------------------------------------------------
    size_t geom_chunk_size = 0;
    {
        // Dry run to compute total size
        char* tmp = nullptr;
        GeometryState::fromChunk(tmp, P);
        geom_chunk_size = (size_t)tmp + 127;  // upper bound; use required_size pattern
    }
    // Simpler: just allocate a conservatively large buffer
    // +2*sizeof(uint2) for tile_min and tile_max arrays
    geom_chunk_size = (P * (sizeof(float)*(1+6+3+4+1) + sizeof(float2) + sizeof(uint32_t)*2 + sizeof(uint2)*2) + P * 4096 + 65536);
    char* geom_ptr = geometryBuffer(geom_chunk_size);
    GeometryState geomState = GeometryState::fromChunk(geom_ptr, P);

    size_t img_chunk_size = (num_tiles * sizeof(uint2) + W*H * (sizeof(uint32_t) + sizeof(float)) + 65536);
    char* img_ptr = imageBuffer(img_chunk_size);
    ImageState imgState = ImageState::fromChunk(img_ptr, num_tiles, W, H);

    // Zero tile ranges (some tiles may have no Gaussians)
    cudaMemset(imgState.ranges, 0, num_tiles * sizeof(uint2));

    // -----------------------------------------------------------------------
    // Preprocessing pass
    // -----------------------------------------------------------------------
    Forward::preprocessCUDA<<<(P+255)/256, 256>>>(
        P, D, M,
        means3D,
        reinterpret_cast<const float3*>(scales),
        scale_modifier,
        reinterpret_cast<const float4*>(rotations),
        opacities,
        shs,
        geomState.clamped,
        cov3D_precomp,
        colors_precomp,
        viewmatrix,
        projmatrix,
        cam_pos,
        W, H,
        tan_fovx, tan_fovy,
        focal_x, focal_y,
        radii,
        geomState.means2D,
        geomState.depths,
        geomState.cov3D,
        geomState.rgb,
        geomState.conic_opacity,
        tile_grid,
        geomState.tiles_touched,
        geomState.tile_min,
        geomState.tile_max,
        prefiltered
    );
    CHECK_CUDA("fwd::preprocess");

    if (_debug_sync_enabled()) {
        uint32_t tt[8] = {};
        cudaMemcpy(tt, geomState.tiles_touched, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[DBG] tiles_touched[0..7]: %u %u %u %u %u %u %u %u\n",
            tt[0], tt[1], tt[2], tt[3], tt[4], tt[5], tt[6], tt[7]);
        fflush(stderr);
    }

    // -----------------------------------------------------------------------
    // Prefix sum to get per-Gaussian offsets
    // -----------------------------------------------------------------------
    cub::DeviceScan::InclusiveSum(
        geomState.scanning_space, geomState.scan_size,
        geomState.tiles_touched, geomState.point_offsets, P
    );
    CHECK_CUDA("fwd::inclusive_sum");

    if (_debug_sync_enabled()) {
        uint32_t first_offsets[8] = {};
        cudaMemcpy(first_offsets, geomState.point_offsets, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[DBG] offsets[0..7]: %u %u %u %u %u %u %u %u\n",
            first_offsets[0], first_offsets[1], first_offsets[2], first_offsets[3],
            first_offsets[4], first_offsets[5], first_offsets[6], first_offsets[7]);
        fflush(stderr);
    }

    // Total number of (Gaussian, tile) pairs
    uint32_t num_rendered = 0;
    cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (_debug_sync_enabled()) { fprintf(stderr, "[DBG] forward #%d: P=%d num_rendered=%u num_tiles=%d\n", fwd_call, P, num_rendered, num_tiles); fflush(stderr); }
    if (num_rendered == 0) return 0;

    // -----------------------------------------------------------------------
    // Allocate binning state now that we know num_rendered
    // -----------------------------------------------------------------------
    size_t sorting_size = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sorting_size,
        (uint64_t*)nullptr, (uint64_t*)nullptr,
        (uint32_t*)nullptr, (uint32_t*)nullptr, num_rendered);
    size_t bin_chunk_size = (num_rendered * (2*sizeof(uint64_t) + 2*sizeof(uint32_t)) + sorting_size + 65536);
    char* bin_ptr = binningBuffer(bin_chunk_size);
    char* bin_ptr_saved = bin_ptr;
    BinningState binState = BinningState::fromChunk(bin_ptr, num_rendered);
    if (_debug_sync_enabled()) {
        size_t consumed = (size_t)(bin_ptr - bin_ptr_saved);
        fprintf(stderr, "[DBG] bin #%d: sort_scratch=%zu chunk=%zu consumed=%zu fit=%s fromChunk_sort=%zu\n",
            fwd_call, sorting_size, bin_chunk_size, consumed,
            (consumed <= bin_chunk_size) ? "OK" : "OVERFLOW", binState.sorting_size);
        fflush(stderr);
    }

    // -----------------------------------------------------------------------
    // Scatter (tile_id, depth) pairs into binning buffers
    // -----------------------------------------------------------------------
    Forward::duplicateWithKeys<<<(P+255)/256, 256>>>(
        P,
        geomState.depths,
        geomState.point_offsets,
        binState.point_list_keys_unsorted,
        binState.point_list_unsorted,
        radii,
        geomState.tile_min,
        geomState.tile_max,
        (int)tile_grid.x
    );
    CHECK_CUDA("fwd::duplicate_keys");

    // -----------------------------------------------------------------------
    // Sort by (tile_id, depth)
    // -----------------------------------------------------------------------
    int sort_bits = numBitsForSorting(num_tiles);
    cub::DeviceRadixSort::SortPairs(
        binState.list_sorting_space, binState.sorting_size,
        binState.point_list_keys_unsorted, binState.point_list_keys,
        binState.point_list_unsorted,      binState.point_list,
        num_rendered, 0, sort_bits
    );
    CHECK_CUDA("fwd::radix_sort");

    // -----------------------------------------------------------------------
    // Identify per-tile ranges in the sorted list
    // -----------------------------------------------------------------------
    Forward::identifyTileRanges<<<(num_rendered+255)/256, 256>>>(
        num_rendered,
        binState.point_list_keys,
        imgState.ranges
    );
    CHECK_CUDA("fwd::tile_ranges");

    // -----------------------------------------------------------------------
    // Tile renderer forward
    // -----------------------------------------------------------------------
    Forward::renderCUDA<<<tile_grid, block>>>(
        imgState.ranges,
        binState.point_list,
        W, H,
        geomState.means2D,
        geomState.rgb,
        geomState.conic_opacity,
        imgState.n_contrib,
        background,
        out_color,
        imgState.accum_alpha
    );
    CHECK_CUDA("fwd::render");

    return (int)num_rendered;
}

// ============================================================================
// backward
// ============================================================================

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
    const float* dL_dpix,
    float* dL_dmean2D,
    float* dL_dcov3D,
    float* dL_dcolor,
    float* dL_dopacity,
    float* dL_dmean3D,
    float* dL_dsh,
    float* dL_dscale,
    float* dL_drot,
    bool debug
) {
    if (P == 0) return;

    float focal_y = H / (2.f * tan_fovy);
    float focal_x = W / (2.f * tan_fovx);

    dim3 tile_grid((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Re-create state views from the opaque buffers saved in the forward pass.
    int num_tiles = tile_grid.x * tile_grid.y;
    GeometryState geomState = GeometryState::fromChunk(geomBuffer, P);
    BinningState  binState  = BinningState::fromChunk(binningBuffer, R);
    ImageState    imgState  = ImageState::fromChunk(imageBuffer, num_tiles, W, H);

    // -----------------------------------------------------------------------
    // Tile renderer backward
    // -----------------------------------------------------------------------
    Backward::renderBackwardCUDA<<<tile_grid, block>>>(
        imgState.ranges,
        binState.point_list,
        W, H,
        background,
        geomState.means2D,
        geomState.conic_opacity,
        geomState.rgb,
        imgState.accum_alpha,
        imgState.n_contrib,
        dL_dpix,
        reinterpret_cast<float3*>(dL_dmean2D),
        reinterpret_cast<float4*>(dL_dcolor + P*3),   // use dL_dcolor tail as conic scratch
        reinterpret_cast<float3*>(dL_dcolor)
    );
    CHECK_CUDA("bwd::render_backward");

    // -----------------------------------------------------------------------
    // Preprocessing backward
    // -----------------------------------------------------------------------
    Backward::preprocessBackwardCUDA<<<(P+255)/256, 256>>>(
        P, D, M,
        reinterpret_cast<const float3*>(means3D),
        radii,
        shs,
        geomState.clamped,
        reinterpret_cast<const float3*>(scales),
        reinterpret_cast<const float4*>(rotations),
        scale_modifier,
        geomState.cov3D,
        viewmatrix,
        projmatrix,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        W, H,
        reinterpret_cast<const float3*>(dL_dmean2D),
        dL_dcolor + P*3,        // dL_dconics (written above as conic scratch)
        dL_dopacity,
        dL_dcolor,
        dL_dmean3D,
        dL_dcov3D,
        dL_dsh,
        reinterpret_cast<float3*>(dL_dscale),
        reinterpret_cast<float4*>(dL_drot)
    );
    CHECK_CUDA("bwd::preprocess_backward");
}

}  // namespace CudaRasterizer
