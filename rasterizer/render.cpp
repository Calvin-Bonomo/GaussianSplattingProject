#include <cassert>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>

#include "wrapper.hpp"
#include "settings.hpp"

namespace py = pybind11;

void forward(
        torch::Tensor means, 
        torch::Tensor scales, 
        torch::Tensor rotations, 
        torch::Tensor colors, 
        torch::Tensor opacities,
        torch::Tensor viewTransform,
        py::array_t<float> focal,
        float zNear,
        float zFar,
        int width,
        int height)
{
    assert(width > 0 && height > 0 && "width and height must be greater than 0");

    int xTiles = ceil(width / TILE_SIZE),
        yTiles = ceil(height / TILE_SIZE);

    int64_t numGaussians = means.numel();
    torch::Tensor tilesTouched = torch::zeros({numGaussians}, torch::kInt32),
                  means2D = torch::zeros({numGaussians * 2}, torch::kInt32),
                  cov2D = torch::zeros({numGaussians * 3}, torch::kFloat32),
                  depths = torch::zeros({numGaussians}, torch::kFloat32),
                  aabb = torch::zeros({numGaussians * 4}, torch::kFloat32),
                  gaussianOffsets = torch::zeros({numGaussians}, torch::kInt32),
                  image = torch::zeros({width * height * 3}, torch::kInt8),
                  tileRanges = torch::zeros({xTiles * yTiles * 2}, torch::kInt32);

    // Prepare the data to go to CUDA
    float *pMeans = (float *)means.contiguous().data_ptr(),
          *pScales = (float *)scales.contiguous().data_ptr(),
          *pRotations = (float *)rotations.contiguous().data_ptr(),
          *pColors = (float *)colors.contiguous().data_ptr(),
          *pOpacities = (float *)opacities.contiguous().data_ptr(),
          *pViewTransform = (float *)viewTransform.contiguous().data_ptr(),
          *pCov2D = (float *)cov2D.data_ptr(),
          *pDepths = (float *)depths.data_ptr(),
          *pAabbs = (float *)aabb.data_ptr();
    uint32_t *pTilesTouched = (uint32_t *)tilesTouched.data_ptr(),
             *pMeans2D = (uint32_t *)means2D.data_ptr(),
             *pGaussianOffsets = (uint32_t *)gaussianOffsets.data_ptr();
    uint8_t *pImage = (uint8_t *)image.data_ptr();
    int32_t *pTileRanges = (int32_t *)tileRanges.data_ptr();

    smartCudaPtr<uint64_t> gaussianIndices(nullptr, nullptr),
                           gaussianKeys(nullptr, nullptr);


    // TODO: get output, package it and send it back
    forwardCUDA(
            numGaussians, 
            pMeans, 
            pScales, 
            pRotations, 
            pCov2D,
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
            focal.at(0), focal.at(1), 
            zNear, zFar,
            xTiles, yTiles,
            width, height);
}

void backward()
{

}

void drawView()
{

}

PYBIND11_MODULE(gaussian_splat, m, py::mod_gil_not_used()) {
    m.doc() = "A gaussian splat rasterizer written in cuda.";

    m.def("forward", &forward, "Performs the forward pass for gaussian splatting.");
    m.def("backward", &backward, "Performs the backward pass for gaussian splatting.");
    m.def("draw_view", &drawView, "Draws the user view of the gaussian splatting scene.");
}
