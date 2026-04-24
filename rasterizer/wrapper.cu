#include "wrapper.hpp"

#include <cstdint>
#include <cub/cub.cuh>

#include "rasterize.cuh"
#include "util.cuh"


void forwardCUDA(
        long long numGaussians,
        float *means,
        float *scales,
        float *rotations,
        float *cov2Ds,
        uint32_t *means2D,
        float *depths,
        float *aabbs,
        uint32_t *gaussianOffsets,
        float *colors,
        float *opacities,
        float *viewTransform,
        uint32_t *tilesTouched,
        smartCudaPtr<uint64_t> &&gaussianIndices,
        smartCudaPtr<uint64_t> &&gaussianKeys,
        uint8_t *image,
        float focalX, float focalY,
        float zNear, float zFar,
        int width, int height)
{
    // Step 1: Process gaussians (do transformations and stuff)
    // Initialize clipping planes
    float3 p1 = normalize((float3){-0.5f * focalX, 0, zFar}),
           p2 = normalize((float3){0.5f * focalX, 0, zFar}),
           p3 = normalize((float3){0, -0.5f * focalY, zFar}),
           p4 = normalize((float3){0, 0.5f * focalY, zFar});
    
    plane clipPlanes[4] = {
        { .planeDir = p1, .normal = {p1.z, 0, -p1.x} },
        { .planeDir = p2, .normal = {-p2.z, 0, p2.x} },
        { .planeDir = p3, .normal = {0, -p3.z, p3.y} },
        { .planeDir = p4, .normal = {0, p3.z, -p3.y} }
    };

    float3 *cov2D = reinterpret_cast<float3 *>(cov2Ds);
    uint2 *means2DCU = reinterpret_cast<uint2 *>(means2D);
    float4 *aabb = reinterpret_cast<float4 *>(aabbs);
    
    projectGaussians<<<numGaussians / 256, 256>>>(
            numGaussians,
            reinterpret_cast<float3 *>(means),
            reinterpret_cast<float3 *>(scales),
            reinterpret_cast<float4 *>(rotations),
            cov2D,
            means2DCU,
            depths,
            aabb,
            viewTransform,
            opacities,
            tilesTouched,
            clipPlanes,
            { focalX, focalY },
            zNear, zFar,
            width,
            height);

    // Step 2: Determine # of tiles required
    int xTiles = ceil(float(width) / TILE_SIZE),
        yTiles = ceil(float(height) / TILE_SIZE);

    // Step 3: Assign each gaussian a key with the tile and depth
    // Find out how many tiles were touched
    void *tempStorage = nullptr;
    size_t tempStorageBytes = 0;
    cub::DeviceScan::InclusiveSum(tempStorage, tempStorageBytes, tilesTouched, gaussianOffsets, numGaussians);
    cudaMallocAsync(&tempStorage, tempStorageBytes, cudaStreamPerThread);
    cub::DeviceScan::InclusiveSum(tempStorage, tempStorageBytes, tilesTouched, gaussianOffsets, numGaussians);
    cudaFreeAsync(tempStorage, cudaStreamPerThread);
    
    // Setup initial keys and indices
    uint64_t *gaussianKeysIn = nullptr,
             *gaussianIndicesIn = nullptr,
             *gaussianKeysOut = nullptr,
             *gaussianIndicesOut = nullptr;
    int totalTilesTouched = gaussianOffsets[numGaussians - 1]; // TODO: Make this one call and split up the memory manually
    cudaMallocAsync(&gaussianKeysIn, totalTilesTouched * sizeof(uint64_t), cudaStreamPerThread);
    cudaMallocAsync(&gaussianIndicesIn, totalTilesTouched * sizeof(uint64_t), cudaStreamPerThread);
    cudaMallocAsync(&gaussianKeysOut, totalTilesTouched * sizeof(uint64_t), cudaStreamPerThread);
    cudaMallocAsync(&gaussianIndicesOut, totalTilesTouched * sizeof(uint64_t), cudaStreamPerThread);

    duplicateWithKeys<<<numGaussians / 256, 256>>>(
            numGaussians,
            means2DCU, 
            depths, 
            cov2D, 
            aabb, 
            tilesTouched, 
            gaussianKeysIn,
            gaussianIndicesIn, 
            gaussianOffsets,
            xTiles, yTiles);
    
    // Step 4: Sort gaussians using the cub radix sort
    tempStorage = nullptr;
    tempStorageBytes = 0;
    cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, gaussianKeysIn, gaussianKeysOut, gaussianIndicesIn, gaussianIndicesOut, totalTilesTouched);
    cudaMallocAsync(&tempStorage, tempStorageBytes, cudaStreamPerThread);
    cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, gaussianKeysIn, gaussianKeysOut, gaussianIndicesIn, gaussianIndicesOut, totalTilesTouched);
    cudaFreeAsync(tempStorage, cudaStreamPerThread);

    // Step 5: Rasterize gaussian tiles
}
