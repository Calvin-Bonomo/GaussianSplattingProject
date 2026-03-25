#include <torch/extension.h>
#include "rasterizer_impl.h"

// ── Autograd Function ─────────────────────────────────────────────────────────
//
// Wraps the CUDA forward/backward as a differentiable operation.
// Saved tensors (9):  means2d, cov2d, colors, opacities, background,
//                     radii, point_list, tile_ranges, final_transmittance
// Saved ints (2):     image_height, image_width

class RasterizeGaussiansFunction
    : public torch::autograd::Function<RasterizeGaussiansFunction> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor means2d,
        torch::Tensor cov2d,
        torch::Tensor colors,
        torch::Tensor opacities,
        torch::Tensor background,
        int64_t image_height,
        int64_t image_width
    ) {
        auto [rendered, radii, point_list, tile_ranges, final_transmittance] =
            rasterize_forward_cuda(
                means2d, cov2d, colors, opacities, background,
                (int)image_height, (int)image_width
            );

        ctx->save_for_backward({
            means2d, cov2d, colors, opacities, background,
            radii, point_list, tile_ranges, final_transmittance
        });
        ctx->saved_data["image_height"] = image_height;
        ctx->saved_data["image_width"]  = image_width;

        // radii is int32 and carries no gradient
        ctx->mark_non_differentiable({radii});

        return {rendered, radii};
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto saved              = ctx->get_saved_variables();
        auto means2d            = saved[0];
        auto cov2d              = saved[1];
        auto colors             = saved[2];
        auto opacities          = saved[3];
        auto background         = saved[4];
        auto radii              = saved[5];
        auto point_list         = saved[6];
        auto tile_ranges        = saved[7];
        auto final_transmittance = saved[8];

        int64_t H = ctx->saved_data["image_height"].toInt();
        int64_t W = ctx->saved_data["image_width"].toInt();

        // grad_outputs[0] = ∂L/∂rendered  (grad_outputs[1] is undefined: radii has no grad)
        auto [grad_means2d, grad_cov2d, grad_colors, grad_opacities] =
            rasterize_backward_cuda(
                means2d, cov2d, colors, opacities, background,
                radii, point_list, tile_ranges, final_transmittance,
                grad_outputs[0],
                (int)H, (int)W
            );

        // One gradient per forward input (in the same order):
        // means2d, cov2d, colors, opacities, background, image_height, image_width
        return {
            grad_means2d,
            grad_cov2d,
            grad_colors,
            grad_opacities,
            torch::Tensor(),  // background – not differentiated
            torch::Tensor(),  // image_height – not a tensor
            torch::Tensor(),  // image_width  – not a tensor
        };
    }
};

// ── Python-callable entry point ───────────────────────────────────────────────
//
// Matches the call in renderer.py:
//   rendered, radii = ext.rasterize_gaussians(
//       means2d, cov2d, colors, opacities, height, width, background)

std::tuple<torch::Tensor, torch::Tensor> rasterize_gaussians(
    torch::Tensor means2d,
    torch::Tensor cov2d,
    torch::Tensor colors,
    torch::Tensor opacities,
    int64_t image_height,
    int64_t image_width,
    torch::Tensor background
) {
    TORCH_CHECK(means2d.is_cuda(),   "means2d must be a CUDA tensor");
    TORCH_CHECK(cov2d.is_cuda(),     "cov2d must be a CUDA tensor");
    TORCH_CHECK(colors.is_cuda(),    "colors must be a CUDA tensor");
    TORCH_CHECK(opacities.is_cuda(), "opacities must be a CUDA tensor");

    TORCH_CHECK(means2d.dim()   == 2 && means2d.size(1)   == 2, "means2d must be (N, 2)");
    TORCH_CHECK(cov2d.dim()     == 2 && cov2d.size(1)     == 3, "cov2d must be (N, 3)");
    TORCH_CHECK(colors.dim()    == 2 && colors.size(1)    == 3, "colors must be (N, 3)");
    TORCH_CHECK(opacities.dim() == 1,                           "opacities must be (N,)");
    TORCH_CHECK(background.dim() == 1 && background.size(0) == 3, "background must be (3,)");

    auto result = RasterizeGaussiansFunction::apply(
        means2d, cov2d, colors, opacities, background,
        image_height, image_width
    );
    return {result[0], result[1]};
}

// ── Module registration ───────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Tile-based depth-sorted Gaussian splatting rasterizer";
    m.def("rasterize_gaussians", &rasterize_gaussians,
        "Rasterize N 2-D Gaussians onto an (H x W) image.\n\n"
        "Args:\n"
        "  means2d    (N, 2) float32 – screen-space centers (pixels)\n"
        "  cov2d      (N, 3) float32 – upper-triangle 2-D covariance [a,b,c]\n"
        "  colors     (N, 3) float32 – per-Gaussian RGB in [0,1]\n"
        "  opacities  (N,)   float32 – per-Gaussian opacity in [0,1]\n"
        "  height     int            – output image height\n"
        "  width      int            – output image width\n"
        "  background (3,)   float32 – background RGB\n\n"
        "Returns:\n"
        "  rendered   (3, H, W) float32 – composited image\n"
        "  radii      (N,)      int32   – screen-space radius per Gaussian (0=culled)"
    );
}
