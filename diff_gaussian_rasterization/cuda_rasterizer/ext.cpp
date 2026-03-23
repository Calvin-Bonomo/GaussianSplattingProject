#include <torch/extension.h>
#include "rasterizer.h"

#include <functional>

// ---------------------------------------------------------------------------
// Helper: wrap a torch::Tensor allocation into a std::function<char*(size_t)>
// that resizes the tensor on first call.  The tensor's data_ptr is stable
// because we pre-allocate it at the required size.
// ---------------------------------------------------------------------------
std::function<char*(size_t)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t size) -> char* {
        t.resize_({(long long)size});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

// ---------------------------------------------------------------------------
// Forward binding
// ---------------------------------------------------------------------------
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool debug
) {
    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must be (P, 3)");
    }

    const int P = means3D.size(0);
    const int H = image_height;
    const int W = image_width;

    auto int_opts    = means3D.options().dtype(torch::kInt32);
    auto float_opts  = means3D.options().dtype(torch::kFloat32);

    torch::Tensor out_color = torch::full({3, H, W}, 0.0, float_opts);
    torch::Tensor radii     = torch::full({P},       0,   int_opts);

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer   = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer    = torch::empty({0}, options.device(device));

    // Depth output is optional (pass nullptr for now)
    int num_rendered = CudaRasterizer::forward(
        resizeFunctional(geomBuffer),
        resizeFunctional(binningBuffer),
        resizeFunctional(imgBuffer),
        P, degree,
        (sh.numel() != 0) ? ((sh.size(1) * sh.size(2))) / 3 : 0,
        background.contiguous().data_ptr<float>(),
        W, H,
        means3D.contiguous().data_ptr<float>(),
        (sh.numel() != 0) ? sh.contiguous().data_ptr<float>() : nullptr,
        (colors.numel() != 0) ? colors.contiguous().data_ptr<float>() : nullptr,
        opacity.contiguous().data_ptr<float>(),
        (scales.numel() != 0) ? scales.contiguous().data_ptr<float>() : nullptr,
        scale_modifier,
        (rotations.numel() != 0) ? rotations.contiguous().data_ptr<float>() : nullptr,
        (cov3D_precomp.numel() != 0) ? cov3D_precomp.contiguous().data_ptr<float>() : nullptr,
        viewmatrix.contiguous().data_ptr<float>(),
        projmatrix.contiguous().data_ptr<float>(),
        campos.contiguous().data_ptr<float>(),
        tan_fovx, tan_fovy,
        radii.data_ptr<int>(),
        out_color.data_ptr<float>(),
        /* out_depth */ nullptr,
        prefiltered,
        debug
    );

    return std::make_tuple(num_rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer);
}

// ---------------------------------------------------------------------------
// Backward binding
// ---------------------------------------------------------------------------
std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const torch::Tensor& geomBuffer,
    const int R,   // num_rendered
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const bool debug
) {
    const int P = means3D.size(0);
    const int H = dL_dout_color.size(1);
    const int W = dL_dout_color.size(2);

    int M = 0;
    if (sh.numel() != 0) {
        M = sh.size(1) * sh.size(2) / 3;
    }

    auto float_opts = means3D.options().dtype(torch::kFloat32);

    torch::Tensor dL_dmeans3D    = torch::zeros({P, 3},    float_opts);
    torch::Tensor dL_dmeans2D    = torch::zeros({P, 3},    float_opts);
    torch::Tensor dL_dcolors     = torch::zeros({P, 3+4},  float_opts);  // +4 for float4 conic_opacity scratch
    torch::Tensor dL_dcov3D      = torch::zeros({P, 6},    float_opts);
    torch::Tensor dL_dopacity    = torch::zeros({P, 1},    float_opts);
    torch::Tensor dL_dsh         = torch::zeros({P, M, 3}, float_opts);
    torch::Tensor dL_dscales     = torch::zeros({P, 3},    float_opts);
    torch::Tensor dL_drotations  = torch::zeros({P, 4},    float_opts);

    CudaRasterizer::backward(
        P, degree, M, R,
        background.contiguous().data_ptr<float>(),
        W, H,
        means3D.contiguous().data_ptr<float>(),
        (sh.numel() != 0) ? sh.contiguous().data_ptr<float>() : nullptr,
        (colors.numel() != 0) ? colors.contiguous().data_ptr<float>() : nullptr,
        (scales.numel() != 0) ? scales.contiguous().data_ptr<float>() : nullptr,
        scale_modifier,
        (rotations.numel() != 0) ? rotations.contiguous().data_ptr<float>() : nullptr,
        (cov3D_precomp.numel() != 0) ? cov3D_precomp.contiguous().data_ptr<float>() : nullptr,
        viewmatrix.contiguous().data_ptr<float>(),
        projmatrix.contiguous().data_ptr<float>(),
        campos.contiguous().data_ptr<float>(),
        tan_fovx, tan_fovy,
        radii.contiguous().data_ptr<int>(),
        reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
        reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
        dL_dout_color.contiguous().data_ptr<float>(),
        dL_dmeans2D.data_ptr<float>(),
        dL_dcov3D.data_ptr<float>(),
        dL_dcolors.data_ptr<float>(),
        dL_dopacity.data_ptr<float>(),
        dL_dmeans3D.data_ptr<float>(),
        dL_dsh.data_ptr<float>(),
        dL_dscales.data_ptr<float>(),
        dL_drotations.data_ptr<float>(),
        debug
    );

    return std::make_tuple(
        dL_dmeans3D, dL_dmeans2D,
        dL_dcolors.slice(1, 0, 3),   // strip the conic scratch
        dL_dcov3D,
        dL_dopacity, dL_dsh,
        dL_dscales, dL_drotations,
        torch::Tensor()  // unused placeholder
    );
}

// ---------------------------------------------------------------------------
// pybind11 module registration
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_gaussians",          &RasterizeGaussiansCUDA);
    m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
}
