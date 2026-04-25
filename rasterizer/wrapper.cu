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
    float3 *invCov2D = reinterpret_cast<float3 *>(invCov2Ds);
    float2 *means2DCU = reinterpret_cast<float2 *>(means2D);
    float4 *aabb = reinterpret_cast<float4 *>(aabbs);
    int2 *ranges = reinterpret_cast<int2 *>(tileRanges);
    float3 *colorsCU = reinterpret_cast<float3 *>(colors);
    
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
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

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
    printf("Touched %d tiles\n", totalTilesTouched);
    cudaMalloc(&gaussianKeysIn, totalTilesTouched * sizeof(uint64_t));
    cudaMalloc(&gaussianIndicesIn, totalTilesTouched * sizeof(uint64_t));
    cudaMalloc(&gaussianKeysOut, totalTilesTouched * sizeof(uint64_t));
    cudaMalloc(&gaussianIndicesOut, totalTilesTouched * sizeof(uint64_t));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int32_t> hostTiles(numGaussians);
    cudaMemcpy(hostTiles.data(), tilesTouched, 
               numGaussians * sizeof(int32_t), cudaMemcpyDeviceToHost);

    int32_t maxT = 0; int worstIdx = 0;
    int64_t sumT = 0;
    int hugeCount = 0;
    for (int i = 0; i < numGaussians; i++) {
        sumT += hostTiles[i];
        if (hostTiles[i] > maxT) { maxT = hostTiles[i]; worstIdx = i; }
        if (hostTiles[i] > 1000) hugeCount++;
    }
    std::cerr << "tilesTouched: sum=" << sumT 
              << " max=" << maxT 
              << " huge_gaussians=" << hugeCount << "\n";
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
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 4: Sort gaussians using the cub radix sort
    tempStorage = nullptr;
    tempStorageBytes = 0;
    cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, gaussianKeysIn, gaussianKeysOut, gaussianIndicesIn, gaussianIndicesOut, totalTilesTouched, 0, sizeof(uint64_t) * 8);
    cudaMalloc(&tempStorage, tempStorageBytes);
    cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, gaussianKeysIn, gaussianKeysOut, gaussianIndicesIn, gaussianIndicesOut, totalTilesTouched, 0, sizeof(uint64_t) * 8);
    cudaFree(tempStorage);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(gaussianKeysIn);
    cudaFree(gaussianIndicesIn);

    identifyTileRanges<<<(totalTilesTouched + 256 - 1) / 256, 256>>>( // Each thread is a tile
                xTiles * yTiles,
                totalTilesTouched, 
                gaussianKeysOut,
                ranges);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i : {0, 100, 1000, 10000, 100000}) {
        if (i >= numGaussians) continue;
        float2 m; cudaMemcpy(&m, means2D + i, sizeof(float2), cudaMemcpyDeviceToHost);
        float3 c; cudaMemcpy(&c, cov2D + i, sizeof(float3), cudaMemcpyDeviceToHost);
        float3 ic; cudaMemcpy(&ic, invCov2D + i, sizeof(float3), cudaMemcpyDeviceToHost);
        float3 col; cudaMemcpy(&col, colors + i, sizeof(float3), cudaMemcpyDeviceToHost);
        std::cerr << "g[" << i << "]: mean=(" << m.x << "," << m.y << ")"
                  << " cov=(" << c.x << "," << c.y << "," << c.z << ")"
                  << " invCov=(" << ic.x << "," << ic.y << "," << ic.z << ")"
                  << " color=(" << col.x << "," << col.y << "," << col.z << ")\n";
    }
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
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(gaussianKeysOut);
    cudaFree(gaussianIndicesOut);
}
