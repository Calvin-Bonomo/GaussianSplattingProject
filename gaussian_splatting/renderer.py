"""Thin wrapper: sets up GaussianRasterizationSettings and calls the rasterizer."""

import torch
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.sh_utils import eval_sh


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None):
    """
    Render the scene from a single viewpoint.

    Args:
        viewpoint_camera: Camera object with projection matrices and image dims
        pc: GaussianModel holding current parameters
        pipe: PipelineParams (compute_cov3D_python, convert_SHs_python flags)
        bg_color: (3,) background colour tensor (CUDA)
        scaling_modifier: global scale multiplier
        override_color: (N, 3) precomputed colours (skips SH evaluation)

    Returns:
        dict with keys:
            "render":            (3, H, W) rendered image
            "viewspace_points":  (N, 3) screenspace position tensor (holds 2D grad)
            "visibility_filter": (N,) bool mask of contributing Gaussians
            "radii":             (N,) per-Gaussian 2D radius
    """
    # Create a zero tensor that autograd will write 2D position gradients into.
    # We retain its grad so densification can read ||grad_means2D||.
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=viewpoint_camera.tanfovx,
        tanfovy=viewpoint_camera.tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # Covariance: either compute in Python (for gradients via Python autograd)
    # or let the CUDA kernel compute it from scale + rotation
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
        scales        = None
        rotations     = None
    else:
        cov3D_precomp = None
        scales        = pc.get_scaling
        rotations     = pc.get_rotation

    # Colour: either evaluate SH in Python or let the CUDA kernel do it
    if override_color is not None:
        colors_precomp = override_color
        shs            = None
    elif pipe.convert_SHs_python:
        # Python-side SH evaluation
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        dir_pp_norm = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_norm)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        shs            = None
    else:
        colors_precomp = None
        shs            = pc.get_features

    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
