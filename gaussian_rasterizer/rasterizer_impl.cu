#include "rasterizer_impl.h"

// TODO: implement CUDA kernels.
//
// Architecture notes (from CLAUDE.md):
//   - Tile size: 16 × 16 pixels
//   - Gaussians are sorted per-tile by depth using CUB DeviceRadixSort
//     with composite key  (tile_id << 32 | depth_uint)
//   - Forward: for each tile, blend Gaussians front-to-back using
//     alpha compositing:  C += T * α_i * c_i,  T *= (1 - α_i)
//   - Backward: reverse the tile traversal, accumulating gradients
//     into grad_means2d, grad_cov2d, grad_colors, grad_opacities

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
>
rasterize_forward_cuda(
    torch::Tensor means2d,
    torch::Tensor cov2d,
    torch::Tensor colors,
    torch::Tensor opacities,
    torch::Tensor background,
    int image_height,
    int image_width
) {
    TORCH_CHECK(false, "rasterize_forward_cuda: CUDA kernel not yet implemented");
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
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
) {
    TORCH_CHECK(false, "rasterize_backward_cuda: CUDA kernel not yet implemented");
}
