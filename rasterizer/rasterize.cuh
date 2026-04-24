#ifndef RASTERIZE_CUH
#define RASTERIZE_CUH

#include <cstdint>

struct plane;

__global__ void projectGaussians(
        long long numGaussians,
        float3 *means,
        float3 *scales,
        float4 *rotations,
        float3 *cov2D,
        uint2 *means2D,
        float *depths,
        float4 *aabb,
        float *viewTransform,
        float *opacities,
        uint32_t *tilesTouched,
        plane *clipPlanes,
        float2 focal,
        float zNear,
        float zFar,
        int width, int height);

__global__ void duplicateWithKeys(
        long long numGaussians,
        uint2 *means2D,
        float *depths,
        float3 *cov2D,
        float4 *aabb,
        uint32_t *tilesTouched,
        uint64_t *gaussianKeys,
        uint64_t *gaussianIndices,
        uint32_t *gaussianOffsets,
        int xTiles, int yTiles);

__global__ void identifyTileRanges(
        int totalTiles,
        int totalTilesTouched,
        uint64_t *gaussianKeys, 
        int2 *tileRanges);

__global__ void rasterize(
        long long numGaussians,
        float *opacities,
        uint64_t *gaussianKeys,
        uint64_t *gaussianIndices,
        int xTiles, int yTiles,
        int width, int height);

#endif
