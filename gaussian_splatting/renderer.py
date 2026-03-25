import os

import torch
import torch.nn.functional as F
import torch.utils.cpp_extension

from gaussian_splatting.scene.dataset import Camera
from gaussian_splatting.scene.gaussian_model import GaussianModel

from gaussian_splatting import sh_constants

_rasterizer_ext = None


def _load_rasterizer():
    global _rasterizer_ext
    if _rasterizer_ext is None:
        src_dir = os.path.join(os.path.dirname(__file__), '..', 'gaussian_rasterizer')
        _rasterizer_ext = torch.utils.cpp_extension.load(
            name='gaussian_rasterizer',
            sources=[
                os.path.join(src_dir, 'ext.cpp'),
                os.path.join(src_dir, 'rasterizer_impl.cu'),
            ],
            verbose=True,
        )
    return _rasterizer_ext


def _compute_view_dirs(xyz: torch.Tensor, camera: Camera) -> torch.Tensor:
    """Unit vectors from each Gaussian center to the camera origin in world space."""
    R = torch.tensor(camera.R, dtype=torch.float32, device=xyz.device)  # (3, 3)
    T = torch.tensor(camera.T, dtype=torch.float32, device=xyz.device)  # (3,)
    cam_pos = -(R.T @ T)                   # world-space camera origin: -R^T t
    dirs = cam_pos.unsqueeze(0) - xyz      # (N, 3) vectors toward camera
    return F.normalize(dirs, dim=-1)


def _eval_sh(sh_band0: torch.Tensor, sh_bands_rest: torch.Tensor,
             active_degree: int, view_dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate the SH expansion at each view direction to get per-Gaussian RGB colors.

    sh_band0      : (N, 1, 3)  – DC coefficient
    sh_bands_rest : (N, 15, 3) – higher-order coefficients (bands 1–3)
    Returns       : (N, 3) clamped to [0, ∞)
    """
    result = sh_constants.C0 * sh_band0[:, 0, :]   # (N, 3)

    if active_degree > 0:
        x = view_dirs[:, 0:1]   # (N, 1) – broadcasts over RGB
        y = view_dirs[:, 1:2]
        z = view_dirs[:, 2:3]
        result = result + sh_constants.C1 * (
            -y * sh_bands_rest[:, 0] +
             z * sh_bands_rest[:, 1] +
            -x * sh_bands_rest[:, 2]
        )

        if active_degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, xz, yz = x * y, x * z, y * z
            result = result + (
                sh_constants.C2[0] *  xy               * sh_bands_rest[:, 3] +
                sh_constants.C2[1] *  yz               * sh_bands_rest[:, 4] +
                sh_constants.C2[2] * (2*zz - xx - yy)  * sh_bands_rest[:, 5] +
                sh_constants.C2[3] *  xz               * sh_bands_rest[:, 6] +
                sh_constants.C2[4] * (xx - yy)         * sh_bands_rest[:, 7]
            )

            if active_degree > 2:
                result = result + (
                    sh_constants.C3[0] * y * (3*xx - yy)           * sh_bands_rest[:, 8]  +
                    sh_constants.C3[1] * xy * z                     * sh_bands_rest[:, 9]  +
                    sh_constants.C3[2] * y * (4*zz - xx - yy)      * sh_bands_rest[:, 10] +
                    sh_constants.C3[3] * z * (2*zz - 3*xx - 3*yy)  * sh_bands_rest[:, 11] +
                    sh_constants.C3[4] * x * (4*zz - xx - yy)      * sh_bands_rest[:, 12] +
                    sh_constants.C3[5] * z * (xx - yy)              * sh_bands_rest[:, 13] +
                    sh_constants.C3[6] * x * (xx - 3*yy)            * sh_bands_rest[:, 14]
                )

    return torch.clamp(result + 0.5, min=0.0)


def _compute_3d_covariance(scaling: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Build the 3D covariance matrix Σ = R S S^T R^T for each Gaussian."""
    S = torch.diag_embed(torch.exp(scaling))   # (N, 3, 3)

    q = F.normalize(rotation, dim=-1)          # (N, 4) unit quaternion
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),   2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)               # (N, 3, 3)

    RS = R @ S
    return RS @ RS.transpose(-1, -2)           # (N, 3, 3)


def _project_to_screen(xyz: torch.Tensor, cov3d: torch.Tensor,
                        camera: Camera) -> tuple[torch.Tensor, torch.Tensor]:
    """EWA-project Gaussians into screen space. Returns (means2d, cov2d)."""
    R = torch.tensor(camera.R, dtype=torch.float32, device=xyz.device)  # (3, 3)
    T = torch.tensor(camera.T, dtype=torch.float32, device=xyz.device)  # (3,)

    # Camera-space positions
    t = (xyz @ R.T) + T                        # (N, 3)
    tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

    # Perspective projection → pixel coords
    means2d = torch.stack([
        camera.fx * tx / tz + camera.cx,
        camera.fy * ty / tz + camera.cy,
    ], dim=-1)                                 # (N, 2)

    # EWA Jacobian of the perspective map: ∂(u,v)/∂(x,y,z) at t
    tz_sq  = tz * tz
    zeros  = torch.zeros_like(tz)
    J = torch.stack([
        camera.fx / tz,  zeros,           -camera.fx * tx / tz_sq,
        zeros,           camera.fy / tz,  -camera.fy * ty / tz_sq,
    ], dim=-1).reshape(-1, 2, 3)              # (N, 2, 3)

    # Σ_2d = J R_cam Σ_3d R_cam^T J^T
    W  = R.unsqueeze(0).expand(len(xyz), -1, -1)   # (N, 3, 3)
    JW = J @ W                                      # (N, 2, 3)
    cov2d_full = JW @ cov3d @ JW.transpose(-1, -2)  # (N, 2, 2)

    # Small low-pass filter on diagonal for numerical stability
    cov2d_full = cov2d_full + torch.eye(2, device=xyz.device) * 0.3

    # Pack symmetric matrix as upper-triangle (a, b, c) = ([0,0], [0,1], [1,1])
    cov2d = torch.stack([
        cov2d_full[:, 0, 0],
        cov2d_full[:, 0, 1],
        cov2d_full[:, 1, 1],
    ], dim=-1)                                 # (N, 3)

    return means2d, cov2d


def _rasterize(means2d: torch.Tensor, cov2d: torch.Tensor, colors: torch.Tensor,
               opacity: torch.Tensor, camera: Camera):
    """Tile-based depth-sorted rasterizer. Dispatches to the gaussian_rasterizer CUDA extension.
    Returns (render, viewspace_points, visibility_filter, radii)."""
    ext = _load_rasterizer()
    background = torch.zeros(3, device=means2d.device)
    means2d.retain_grad()   # needed so densify_and_prune can read ∂L/∂μ_screen

    rendered, radii = ext.rasterize_gaussians(
        means2d,
        cov2d,
        colors,
        torch.sigmoid(opacity).squeeze(-1),   # (N,) in [0, 1]
        camera.height,
        camera.width,
        background,
    )

    visibility_filter = radii > 0
    return rendered, means2d, visibility_filter, radii


def render(camera: Camera, gaussians: GaussianModel) -> dict:
    view_dirs = _compute_view_dirs(gaussians._xyz, camera)
    colors    = _eval_sh(gaussians._sh_band0, gaussians._sh_bands_rest,
                         gaussians.active_sh_degree, view_dirs)
    cov3d              = _compute_3d_covariance(gaussians._scaling, gaussians._rotation)
    means2d, cov2d     = _project_to_screen(gaussians._xyz, cov3d, camera)
    rendered, viewspace_points, visibility_filter, radii = _rasterize(
        means2d, cov2d, colors, gaussians._opacity, camera
    )

    return {
        'render':            rendered,
        'viewspace_points':  viewspace_points,
        'visibility_filter': visibility_filter,
        'radii':             radii,
    }
