#ifndef WRAPPER_HPP
#define WRAPPER_HPP

#include <cstdint>
#include <memory>

template<typename T>
extern void cudaAsyncDeleter(T *t);

template<typename T>
using smartCudaPtr = std::unique_ptr<T, void(*)(T *)>;

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
        int width, int height);

        #endif
