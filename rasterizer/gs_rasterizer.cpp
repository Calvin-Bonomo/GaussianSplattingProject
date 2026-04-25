#include <cassert>

#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <cstdio>

#include "wrapper.hpp"
#include "settings.hpp"

namespace py = pybind11;

torch::Tensor forward(
        torch::Tensor means, 
        torch::Tensor scales, 
        torch::Tensor rotations, 
        torch::Tensor colors, 
        torch::Tensor opacities,
        torch::Tensor viewTransform,
        float focalX,
        float focalY,
        float zNear,
        float zFar,
        int width,
        int height)
{
    static int callCount = 0;
    assert(width > 0 && height > 0 && "width and height must be greater than 0");

    int xTiles = (width + TILE_SIZE - 1) / TILE_SIZE,
        yTiles = (height + TILE_SIZE - 1) / TILE_SIZE;

    int64_t numGaussians = means.size(0);

    // Setup different tensor types
    auto floatCuda = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto int32Cuda = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto uint8Cuda = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);

    auto tilesTouched     = torch::zeros({numGaussians}, int32Cuda);
    auto means2D          = torch::zeros({numGaussians, 2}, floatCuda);
    auto cov2D            = torch::zeros({numGaussians, 3}, floatCuda);
    auto invCov2D         = torch::zeros({numGaussians, 3}, floatCuda);
    auto depths           = torch::zeros({numGaussians}, floatCuda);
    auto aabb             = torch::zeros({numGaussians, 4}, floatCuda);
    auto gaussianOffsets  = torch::zeros({numGaussians}, int32Cuda);
    auto image            = torch::zeros({height, width, 3}, uint8Cuda);
    auto tileRanges       = torch::zeros({yTiles, xTiles, 2}, int32Cuda);

    auto meansC = means.contiguous();
    auto scalesC = scales.contiguous();
    auto rotationsC = rotations.contiguous();
    auto colorsC = colors.contiguous();
    auto opacitiesC = opacities.contiguous();
    auto viewTransformC = viewTransform.contiguous();

    // Prepare the data to go to CUDA
    float *pMeans = (float *)meansC.data_ptr(),
          *pScales = (float *)scalesC.data_ptr(),
          *pRotations = (float *)rotationsC.data_ptr(),
          *pColors = (float *)colorsC.data_ptr(),
          *pOpacities = (float *)opacitiesC.data_ptr(),
          *pViewTransform = (float *)viewTransformC.data_ptr(),
          *pCov2D = (float *)cov2D.data_ptr(),
          *pDepths = (float *)depths.data_ptr(),
          *pAabbs = (float *)aabb.data_ptr(),
          *pInvCov2D = (float *)invCov2D.data_ptr(),
          *pMeans2D = (float *)means2D.data_ptr();
    uint32_t *pTilesTouched = (uint32_t *)tilesTouched.data_ptr(),
             *pGaussianOffsets = (uint32_t *)gaussianOffsets.data_ptr();
    uint8_t *pImage = (uint8_t *)image.data_ptr();
    int32_t *pTileRanges = (int32_t *)tileRanges.data_ptr();

    smartCudaPtr<uint64_t> gaussianIndices(nullptr, nullptr),
                           gaussianKeys(nullptr, nullptr);
    
    float timeElapsedMS;

    forwardCUDA(
            numGaussians, 
            pMeans, 
            pScales, 
            pRotations, 
            pCov2D,
            pInvCov2D,
            pMeans2D,
            pDepths,
            pAabbs,
            pGaussianOffsets,
            pColors, 
            pOpacities, 
            pViewTransform, 
            pTilesTouched, 
            std::move(gaussianIndices),
            std::move(gaussianKeys),
            pTileRanges,
            pImage,
            focalX, focalY, 
            zNear, zFar,
            xTiles, yTiles,
            width, height,
            &timeElapsedMS);

    printf("Rasterized frame in %f ms\n", timeElapsedMS);
    return image;
}

void backward()
{

}

void drawView()
{

}

PYBIND11_MODULE(gs_rasterizer, m, py::mod_gil_not_used()) {
    m.doc() = "A gaussian splat rasterizer written in cuda.";

    m.def("forward", &forward, "Performs the forward pass for gaussian splatting.");
    m.def("backward", &backward, "Performs the backward pass for gaussian splatting.");
    m.def("draw_view", &drawView, "Draws the user view of the gaussian splatting scene.");
}
