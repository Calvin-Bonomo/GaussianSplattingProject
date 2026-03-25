#pragma once
#include <tuple>
#include <torch/extension.h>

// ── Forward ────────────────────────────────────────────────────────────────────
//
// Tile-based, depth-sorted alpha-compositing rasterizer (forward pass).
// Gaussians outside the frustum or behind the camera are culled (radii = 0).
//
// Inputs
//   means2d   (N, 2) float32  screen-space Gaussian centers (pixels)
//   cov2d     (N, 3) float32  upper-triangle 2-D covariance [a, b, c]
//   colors    (N, 3) float32  per-Gaussian RGB in [0, 1]
//   opacities (N,)   float32  per-Gaussian opacity in [0, 1]
//   background (3,)  float32  background RGB
//   image_height / image_width – output resolution
//
// Returns
//   rendered            (3, H, W) float32  composited image
//   radii               (N,)      int32    screen-space radius per Gaussian (0 = culled)
//   point_list          (K,)      int32    depth-sorted Gaussian indices, tiles concatenated
//   tile_ranges         (T, 2)    int32    [start, end) into point_list for each tile T
//   final_transmittance (H, W)    float32  transmittance remaining after last Gaussian per pixel
std::tuple<
    torch::Tensor,  // rendered
    torch::Tensor,  // radii
    torch::Tensor,  // point_list
    torch::Tensor,  // tile_ranges
    torch::Tensor   // final_transmittance
>
rasterize_forward_cuda(
    torch::Tensor means2d,
    torch::Tensor cov2d,
    torch::Tensor colors,
    torch::Tensor opacities,
    torch::Tensor background,
    int image_height,
    int image_width
);

// ── Backward ──────────────────────────────────────────────────────────────────
//
// Inputs mirror what the forward saved plus the upstream gradient.
//
// Returns
//   grad_means2d   (N, 2) float32
//   grad_cov2d     (N, 3) float32
//   grad_colors    (N, 3) float32
//   grad_opacities (N,)   float32
std::tuple<
    torch::Tensor,  // grad_means2d
    torch::Tensor,  // grad_cov2d
    torch::Tensor,  // grad_colors
    torch::Tensor   // grad_opacities
>
rasterize_backward_cuda(
    torch::Tensor means2d,
    torch::Tensor cov2d,
    torch::Tensor colors,
    torch::Tensor opacities,
    torch::Tensor background,
    torch::Tensor radii,
    torch::Tensor point_list,
    torch::Tensor tile_ranges,
    torch::Tensor final_transmittance,
    torch::Tensor grad_rendered,
    int image_height,
    int image_width
);
