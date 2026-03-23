#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <functional>

namespace CudaRasterizer {

/**
 * Per-Gaussian state computed during preprocessing (lives on the GPU).
 * Allocated once per forward call as a flat byte buffer.
 */
struct GeometryState {
    size_t scan_size;          // temporary scratch size needed by CUB prefix sum

    // Per-Gaussian fields (arrays of length P)
    float*    depths;          // view-space depth (z in camera coords)
    float*    cov3D;           // 3D covariance upper triangle (P * 6)
    float2*   means2D;         // projected 2D screen-space centre (pixels)
    float*    rgb;             // precomputed per-Gaussian RGB colour (P * 3)
    float4*   conic_opacity;   // (a, b, c from inv-2D-cov) + activated opacity
    bool*     clamped;         // per-Gaussian SH colour clamp mask (P)
    uint32_t* tiles_touched;   // number of tiles overlapped by this Gaussian
    uint32_t* point_offsets;   // prefix sum of tiles_touched
    uint2*    tile_min;        // precomputed tile bounding box min (P)
    uint2*    tile_max;        // precomputed tile bounding box max (P)
    char*     scanning_space;  // CUB temporary buffer

    static GeometryState fromChunk(char*& chunk, size_t P);
};

/**
 * Per (Gaussian, tile) pair state: sort keys + sorted Gaussian indices.
 * Allocated once per forward call as a flat byte buffer.
 */
struct BinningState {
    size_t   sorting_size;          // temporary scratch size needed by CUB sort

    uint64_t* point_list_keys_unsorted;  // (tile_id << 32 | depth_uint), length = num_rendered
    uint64_t* point_list_keys;           // sorted version
    uint32_t* point_list_unsorted;       // Gaussian index, length = num_rendered
    uint32_t* point_list;                // sorted version
    char*     list_sorting_space;        // CUB temporary buffer

    static BinningState fromChunk(char*& chunk, size_t P);
};

/**
 * Per-pixel state: tile ranges into the sorted list, contributor counts,
 * final accumulated transmittance (needed by the backward pass).
 */
struct ImageState {
    uint2*    ranges;       // [start, end) into point_list for each tile, length = num_tiles
    uint32_t* n_contrib;    // number of Gaussians that contributed to each pixel
    float*    accum_alpha;  // final transmittance T for each pixel (after all Gaussians)

    static ImageState fromChunk(char*& chunk, size_t N, size_t W, size_t H);
};

// -------------------------------------------------------------------------
// Helper: return the required size of a CUB DeviceInclusiveScan output buffer
// and a CUB DeviceRadixSort buffer.  Implemented in rasterizer_impl.cu.
// -------------------------------------------------------------------------
size_t required_cub_sort_scratch(size_t num_elements);
size_t required_cub_scan_scratch(size_t num_elements);

// -------------------------------------------------------------------------
// Convenience: obtain a pointer into a flat allocation and advance the chunk
// pointer by the requested size (aligned to 128 bytes).
// -------------------------------------------------------------------------
template <typename T>
static T* obtainPtr(char*& chunk, size_t count) {
    T* ptr = reinterpret_cast<T*>(chunk);
    chunk += ((count * sizeof(T) + 127) / 128) * 128;
    return ptr;
}

}  // namespace CudaRasterizer
