#include "wrapper.hpp"

#include <cstdint>
#include <cub/cub.cuh>

#include "rasterize.cuh"
#include "util.cuh"
#include "settings.hpp"


void forwardCUDA(
        long long numGaussians,
        float *means,
        float *scales,
        float *rotations,
        float *cov2Ds,
        float *invCov2Ds,
        float *means2D,
        float *depths,
        float *aabbs,
        uint32_t *gaussianOffsets,
        float *colors,
        float *opacities,
        float *viewTransform,
        uint32_t *tilesTouched,
        smartCudaPtr<uint64_t> &&gaussianIndices,
        smartCudaPtr<uint64_t> &&gaussianKeys,
        int32_t *tileRanges,
        uint8_t *image,
        float focalX, float focalY,
        float zNear, float zFar,
        int xTiles, int yTiles,
        int width, int height,
        float *timeElapsedMS)
{
    // Setup timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Step 1: Process gaussians (do transformations and stuff)
    // Initialize clipping planes
    float halfW = (width  * 0.5f) / focalX;  // tan of half horizontal FOV
    float halfH = (height * 0.5f) / focalY;  // tan of half vertical FOV

    // Unnormalized normals to the four side planes, all passing through origin.
    // Each is the cross product of the "up/right" axis with the edge ray direction.
    float3 leftNormal   = normalize((float3){  1.f,    0.f, halfW }); // points into frustum from left plane
    float3 rightNormal  = normalize((float3){ -1.f,    0.f, halfW });
    float3 bottomNormal = normalize((float3){  0.f,    1.f, halfH });
    float3 topNormal    = normalize((float3){  0.f,   -1.f, halfH });

    plane clipPlanes[4] = {
        { .normal = leftNormal },
        { .normal = rightNormal },
        { .normal = bottomNormal },
        { .normal = topNormal }
    };

    float3 *cov2D = reinterpret_cast<float3 *>(cov2Ds);
    float3 *invCov2D = reinterpret_cast<float3 *>(invCov2Ds);
    float2 *means2DCU = reinterpret_cast<float2 *>(means2D);
    float4 *aabb = reinterpret_cast<float4 *>(aabbs);
    int2 *ranges = reinterpret_cast<int2 *>(tileRanges);
    float3 *colorsCU = reinterpret_cast<float3 *>(colors);
    if (numGaussians <= 0) 
        return;
    projectGaussians<<<numGaussians / 256, 256>>>(
            numGaussians,
            reinterpret_cast<float3 *>(means),
            reinterpret_cast<float3 *>(scales),
            reinterpret_cast<float4 *>(rotations),
            cov2D,
            invCov2D,
            means2DCU,
            depths,
            aabb,
            viewTransform,
            opacities,
            tilesTouched,
            clipPlanes,
            { focalX, focalY },
            zNear, zFar,
            xTiles, yTiles,
            width, height);
#ifdef DEBUG_CUDA
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // Step 3: Assign each gaussian a key with the tile and depth
    // Find out how many tiles were touched
    void *tempStorage = nullptr;
    size_t tempStorageBytes = 0;
    cub::DeviceScan::ExclusiveSum(tempStorage, tempStorageBytes, tilesTouched, gaussianOffsets, numGaussians);
    cudaMalloc(&tempStorage, tempStorageBytes);
    cub::DeviceScan::ExclusiveSum(tempStorage, tempStorageBytes, tilesTouched, gaussianOffsets, numGaussians);
    cudaFree(tempStorage);

    // Setup initial keys and indices
    uint64_t *gaussianKeysIn = nullptr,
             *gaussianIndicesIn = nullptr,
             *gaussianKeysOut = nullptr,
             *gaussianIndicesOut = nullptr;
    int totalTilesTouched, lastCount;
    cudaMemcpy(&totalTilesTouched, &gaussianOffsets[numGaussians - 1], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lastCount, &tilesTouched[numGaussians - 1], sizeof(int), cudaMemcpyDeviceToHost);
    totalTilesTouched += lastCount;
    if (totalTilesTouched <= 0) // Integer overflow
        return;
    cudaMalloc(&gaussianKeysIn, totalTilesTouched * sizeof(uint64_t));
    cudaMalloc(&gaussianIndicesIn, totalTilesTouched * sizeof(uint64_t));
    cudaMalloc(&gaussianKeysOut, totalTilesTouched * sizeof(uint64_t));
    cudaMalloc(&gaussianIndicesOut, totalTilesTouched * sizeof(uint64_t));

    duplicateWithKeys<<<(numGaussians + 256 - 1) / 256, 256>>>(
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
#ifdef DEBUG_CUDA
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // Step 4: Sort gaussians using the cub radix sort
    tempStorage = nullptr;
    tempStorageBytes = 0;
    cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, gaussianKeysIn, gaussianKeysOut, gaussianIndicesIn, gaussianIndicesOut, totalTilesTouched, 0, sizeof(uint64_t) * 8);
    cudaMalloc(&tempStorage, tempStorageBytes);
    cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, gaussianKeysIn, gaussianKeysOut, gaussianIndicesIn, gaussianIndicesOut, totalTilesTouched, 0, sizeof(uint64_t) * 8);
    cudaFree(tempStorage);
    cudaFree(gaussianKeysIn);
    cudaFree(gaussianIndicesIn);

    identifyTileRanges<<<(totalTilesTouched + 256 - 1) / 256, 256>>>( // Each thread is a tile
                xTiles * yTiles,
                totalTilesTouched, 
                gaussianKeysOut,
                ranges);
#ifdef DEBUG_CUDA
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    // Step 5: Rasterize gaussian tiles
    dim3 gridSize(xTiles, yTiles);
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    rasterize<<<gridSize, blockSize>>>( // Each thread is a pixel on a tile
            numGaussians,
            means2DCU,
            invCov2D,
            opacities,
            colorsCU,
            gaussianIndicesOut,
            ranges,
            image,
            xTiles, yTiles,
            width, height);
#ifdef DEBUG_CUDA
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    cudaFree(gaussianKeysOut);
    cudaFree(gaussianIndicesOut);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(timeElapsedMS, start, stop);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in forward: ") + cudaGetErrorString(err));
    }
}
