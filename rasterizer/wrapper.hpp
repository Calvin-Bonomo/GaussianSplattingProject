#ifndef WRAPPER_HPP
#define WRAPPER_HPP

#include <cstdint>
#include <memory>

template<typename T>
void cudaAsyncDeleter(T *t);

template<typename T>
using smartCudaPtr = std::unique_ptr<T, void(*)(T *)>;

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
        smartCudaPtr<uint64_t> gaussianIndices,
        smartCudaPtr<uint64_t> gaussianKeys,
        uint8_t *image,
        float focalX, float focalY,
        float zNear, float zFar,
        int width, int height);

#endif
