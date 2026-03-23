"""
Differentiable Gaussian Rasterization
======================================
Python wrapper around the CUDA tile renderer for 3D Gaussian Splatting.

Usage:
    from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

try:
    from . import _C
except ImportError:
    raise ImportError(
        "diff_gaussian_rasterization C extension not found. "
        "Build it with:  pip install -e ./diff_gaussian_rasterization"
    )


@dataclass
class GaussianRasterizationSettings:
    """All per-render camera and pipeline parameters passed to the rasterizer."""
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor             # (3,) background colour, float32 CUDA
    scale_modifier: float        # global scale multiplier (default 1.0)
    viewmatrix: torch.Tensor     # (4, 4) world-to-camera, float32 CUDA, column-major
    projmatrix: torch.Tensor     # (4, 4) full projection (view × proj), float32 CUDA
    sh_degree: int               # active SH degree (0–3)
    campos: torch.Tensor         # (3,) camera position in world space, float32 CUDA
    prefiltered: bool            # skip per-Gaussian alpha prefiltering step
    debug: bool                  # run extra device synchronisations + checks


class _RasterizeGaussians(torch.autograd.Function):
    """Autograd Function wrapping the CUDA forward/backward rasterizer."""

    @staticmethod
    def forward(
        ctx,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        sh: torch.Tensor,
        colors_precomp: torch.Tensor,
        opacities: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        cov3D_precomp: torch.Tensor,
        raster_settings: GaussianRasterizationSettings,
    ):
        # Keep a reference to the settings so the backward can access it
        ctx.raster_settings = raster_settings

        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
        )

        # Run CUDA forward pass
        (num_rendered, color, radii,
         geomBuffer, binningBuffer, imgBuffer) = _C.rasterize_gaussians(*args)

        # Save tensors needed by the backward pass
        ctx.save_for_backward(
            means3D, radii, colors_precomp, sh, scales, rotations,
            cov3D_precomp, geomBuffer, binningBuffer, imgBuffer
        )
        ctx.num_rendered = num_rendered

        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):
        # Retrieve saved tensors
        (means3D, radii, colors_precomp, sh, scales, rotations,
         cov3D_precomp, geomBuffer, binningBuffer, imgBuffer) = ctx.saved_tensors

        rs = ctx.raster_settings

        args = (
            rs.bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            rs.scale_modifier,
            cov3D_precomp,
            rs.viewmatrix,
            rs.projmatrix,
            rs.tanfovx,
            rs.tanfovy,
            grad_out_color,
            sh,
            rs.sh_degree,
            rs.campos,
            geomBuffer,
            ctx.num_rendered,
            binningBuffer,
            imgBuffer,
            rs.debug,
        )

        (dL_dmeans3D, dL_dmeans2D, dL_dcolors, dL_dcov3D,
         dL_dopacity, dL_dsh, dL_dscales, dL_drotations, _) = _C.rasterize_gaussians_backward(*args)

        grads = (
            dL_dmeans3D,
            dL_dmeans2D,
            dL_dsh,
            dL_dcolors,
            dL_dopacity,
            dL_dscales,
            dL_drotations,
            dL_dcov3D,
            None,   # raster_settings (not a tensor)
        )
        return grads


class GaussianRasterizer(nn.Module):
    """High-level wrapper: set settings once, call forward each frame."""

    def __init__(self, raster_settings: GaussianRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions: torch.Tensor) -> torch.Tensor:
        """Return a bool mask of which Gaussians are visible in the current frustum."""
        with torch.no_grad():
            _, radii = _RasterizeGaussians.apply(
                positions,
                torch.zeros_like(positions),
                torch.Tensor(),
                torch.Tensor(),
                torch.ones(positions.shape[0], 1, device=positions.device),
                torch.Tensor(),
                torch.Tensor(),
                torch.Tensor(),
                self.raster_settings,
            )
        return radii > 0

    def forward(
        self,
        means3D: torch.Tensor,
        means2D: torch.Tensor,
        opacities: torch.Tensor,
        shs: Optional[torch.Tensor] = None,
        colors_precomp: Optional[torch.Tensor] = None,
        scales: Optional[torch.Tensor] = None,
        rotations: Optional[torch.Tensor] = None,
        cov3D_precomp: Optional[torch.Tensor] = None,
    ):
        """
        Render a set of 3D Gaussians.

        Exactly one of (shs, colors_precomp) must be non-None.
        Exactly one of (scales+rotations, cov3D_precomp) must be non-None.

        Returns:
            (rendered_image, radii):
                rendered_image: (3, H, W) float32
                radii:          (P,)      int32 — 0 for culled Gaussians
        """
        raster_settings = self.raster_settings

        if shs is None:
            shs = torch.Tensor()
        if colors_precomp is None:
            colors_precomp = torch.Tensor()
        if scales is None:
            scales = torch.Tensor()
        if rotations is None:
            rotations = torch.Tensor()
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor()

        return _RasterizeGaussians.apply(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )
