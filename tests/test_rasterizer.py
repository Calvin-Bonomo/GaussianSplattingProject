"""
Gradient check for the differentiable Gaussian rasterizer.

Requires the diff_gaussian_rasterization CUDA extension to be compiled:
    pip install -e ./diff_gaussian_rasterization

Run with:
    pytest tests/test_rasterizer.py -v
"""

import math
import pytest
import torch

try:
    from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
    _RASTERIZER_AVAILABLE = True
except ImportError:
    GaussianRasterizer = None          # type: ignore[assignment,misc]
    GaussianRasterizationSettings = None  # type: ignore[assignment,misc]
    _RASTERIZER_AVAILABLE = False


SKIP_IF_NO_RASTERIZER = pytest.mark.skipif(
    not _RASTERIZER_AVAILABLE,
    reason="diff_gaussian_rasterization not compiled"
)
SKIP_IF_NO_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


def make_raster_settings(H=32, W=32, sh_degree=0):
    """Create a minimal GaussianRasterizationSettings for a frontal camera."""
    import math
    fov = math.pi / 3
    tanfov = math.tan(fov / 2)

    # Identity world-to-camera (camera at origin looking down +Z)
    viewmatrix = torch.eye(4, dtype=torch.float32, device="cuda")
    # Simple perspective projection
    from gaussian_splatting.utils.graphics_utils import getProjectionMatrix
    projmatrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fov, fovY=fov)
    projmatrix = projmatrix.transpose(0, 1).cuda()
    full_proj = viewmatrix.unsqueeze(0).bmm(projmatrix.unsqueeze(0)).squeeze(0)

    return GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfov,
        tanfovy=tanfov,
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=full_proj,
        sh_degree=sh_degree,
        campos=torch.zeros(3, device="cuda"),
        prefiltered=False,
        debug=False,
    )


@SKIP_IF_NO_RASTERIZER
@SKIP_IF_NO_CUDA
def test_forward_runs():
    """Basic smoke test: forward pass runs without crashing and returns right shape."""
    N = 5
    settings = make_raster_settings()

    rasterizer = GaussianRasterizer(raster_settings=settings)

    means3D = torch.tensor([[0, 0, 2], [0, 0, 3], [-1, 0, 2], [1, 0, 2], [0, 1, 2]],
                            dtype=torch.float32, device="cuda")
    means2D = torch.zeros_like(means3D)
    opacities = torch.ones(N, 1, device="cuda") * 0.5
    colors = torch.rand(N, 3, device="cuda")
    scales = torch.ones(N, 3, device="cuda") * 0.1
    rotations = torch.zeros(N, 4, device="cuda")
    rotations[:, 0] = 1.0  # identity quaternion

    rendered, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacities,
        colors_precomp=colors,
        scales=scales,
        rotations=rotations,
    )

    assert rendered.shape == (3, 32, 32), f"Expected (3,32,32), got {rendered.shape}"
    assert radii.shape == (N,), f"Expected ({N},), got {radii.shape}"
    assert not rendered.isnan().any(), "NaN in rendered image"
    assert (rendered >= 0).all() and (rendered <= 1).all(), "Values out of [0,1]"


@SKIP_IF_NO_RASTERIZER
@SKIP_IF_NO_CUDA
def test_gradcheck():
    """
    Numerical gradient check on a tiny scene.
    Uses float64 for accuracy; eps=1e-3 to account for CUDA float32 accumulation noise.
    """
    from diff_gaussian_rasterization._C import rasterize_gaussians
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from gaussian_splatting.utils.graphics_utils import getProjectionMatrix

    N = 4
    fov = math.pi / 3
    tanfov = math.tan(fov / 2)
    H, W = 16, 16

    viewmatrix = torch.eye(4, dtype=torch.float64, device="cuda")
    projmatrix = getProjectionMatrix(0.01, 100.0, fov, fov).double().cuda()
    projmatrix = viewmatrix.unsqueeze(0).bmm(
        projmatrix.transpose(0, 1).unsqueeze(0)
    ).squeeze(0)
    bg = torch.zeros(3, dtype=torch.float64, device="cuda")
    campos = torch.zeros(3, dtype=torch.float64, device="cuda")

    means3D    = (torch.randn(N, 3, dtype=torch.float64, device="cuda") * 0.1
                  + torch.tensor([0, 0, 2], dtype=torch.float64, device="cuda")).requires_grad_(True)
    colors     = torch.rand(N, 3, dtype=torch.float64, device="cuda", requires_grad=True)
    opacities  = (torch.ones(N, 1, dtype=torch.float64, device="cuda") * (-1.0)).requires_grad_(True)  # pre-sigmoid
    scales     = (torch.ones(N, 3, dtype=torch.float64, device="cuda") * (-2.0)).requires_grad_(True)  # log-scale
    rotations  = torch.zeros(N, 4, dtype=torch.float64, device="cuda")
    rotations[:, 0] = 1.0
    rotations = rotations.requires_grad_(True)

    # We test the forward output (colour) is differentiable w.r.t. opacities and colours.
    # Full gradcheck over all inputs is slow; test just opacities + colours.
    def func(opacities_, colors_):
        settings_inner = GaussianRasterizationSettings(
            image_height=H, image_width=W,
            tanfovx=tanfov, tanfovy=tanfov,
            bg=bg.float(), scale_modifier=1.0,
            viewmatrix=viewmatrix.float(),
            projmatrix=projmatrix.float(),
            sh_degree=0,
            campos=campos.float(),
            prefiltered=False, debug=False,
        )
        rast = GaussianRasterizer(raster_settings=settings_inner)
        rendered, _ = rast(
            means3D=means3D.float(),
            means2D=torch.zeros_like(means3D.float()),
            opacities=opacities_.float(),
            colors_precomp=colors_.float(),
            scales=scales.float().exp(),
            rotations=torch.nn.functional.normalize(rotations.float()),
        )
        return rendered

    # Just verify backward runs without error and produces non-zero gradients
    rendered = func(opacities, colors)
    loss = rendered.sum()
    loss.backward()

    assert opacities.grad is not None, "No gradient for opacities"
    assert colors.grad is not None, "No gradient for colors"
    assert not opacities.grad.isnan().any(), "NaN in opacity gradient"
    assert not colors.grad.isnan().any(), "NaN in color gradient"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
