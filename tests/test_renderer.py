"""Tests for pure-Python renderer helpers (no CUDA rasterizer required)."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from gaussian_splatting import sh_constants
from gaussian_splatting.renderer import (
    _compute_view_dirs,
    _eval_sh,
    _compute_3d_covariance,
    _project_to_screen,
)
from gaussian_splatting.scene.dataset import Camera


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_camera(
    R=None, T=None,
    fx=500.0, fy=500.0, cx=320.0, cy=240.0,
    width=640, height=480,
):
    """Return a Camera with sensible defaults (identity pose if R/T omitted)."""
    if R is None:
        R = np.eye(3, dtype=np.float64)
    if T is None:
        T = np.zeros(3, dtype=np.float64)
    return Camera(
        uid=0, image=np.zeros((height, width, 3), dtype=np.uint8),
        width=width, height=height,
        fx=fx, fy=fy, cx=cx, cy=cy,
        R=R, T=T,
    )


# ── _compute_view_dirs ────────────────────────────────────────────────────────

class TestComputeViewDirs:
    def test_output_shape(self):
        N = 5
        xyz = torch.randn(N, 3)
        cam = make_camera()
        out = _compute_view_dirs(xyz, cam)
        assert out.shape == (N, 3)

    def test_unit_length(self):
        xyz = torch.randn(10, 3)
        cam = make_camera()
        dirs = _compute_view_dirs(xyz, cam)
        norms = dirs.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(10), atol=1e-6)

    def test_direction_points_toward_camera(self):
        # Camera at origin (R=I, T=0) → cam_pos = -R^T T = 0
        # Point at (0, 0, 5) → dir should point toward origin = (0, 0, -1)
        xyz = torch.tensor([[0.0, 0.0, 5.0]])
        cam = make_camera()  # R=I, T=0  →  cam_pos = (0,0,0)
        dirs = _compute_view_dirs(xyz, cam)
        expected = torch.tensor([[0.0, 0.0, -1.0]])
        assert torch.allclose(dirs, expected, atol=1e-6)

    def test_camera_translation(self):
        # Camera translated to (0, 0, 10): R=I, T=(0,0,10) → cam_pos = (0,0,-10)
        # Point at origin → dir = normalize((0,0,-10) - (0,0,0)) = (0,0,-1)
        xyz = torch.tensor([[0.0, 0.0, 0.0]])
        cam = make_camera(T=np.array([0.0, 0.0, 10.0]))
        dirs = _compute_view_dirs(xyz, cam)
        expected = torch.tensor([[0.0, 0.0, -1.0]])
        assert torch.allclose(dirs, expected, atol=1e-6)

    def test_single_gaussian(self):
        xyz = torch.tensor([[1.0, 2.0, 3.0]])
        cam = make_camera()
        out = _compute_view_dirs(xyz, cam)
        assert out.shape == (1, 3)
        assert abs(out.norm().item() - 1.0) < 1e-6

    def test_device_preserved(self):
        xyz = torch.randn(4, 3)
        cam = make_camera()
        out = _compute_view_dirs(xyz, cam)
        assert out.device == xyz.device


# ── _eval_sh ──────────────────────────────────────────────────────────────────

def make_sh(N=4, seed=0):
    """Return zero-initialized SH tensors of the right shape."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    band0 = torch.randn(N, 1, 3, generator=rng)
    rest  = torch.randn(N, 15, 3, generator=rng)
    return band0, rest


class TestEvalSH:
    def test_output_shape(self):
        N = 6
        band0, rest = make_sh(N)
        dirs = F.normalize(torch.randn(N, 3), dim=-1)
        out = _eval_sh(band0, rest, active_degree=0, view_dirs=dirs)
        assert out.shape == (N, 3)

    def test_output_nonnegative(self):
        """clamp(... + 0.5, min=0) must always be ≥ 0."""
        N = 20
        band0, rest = make_sh(N)
        dirs = F.normalize(torch.randn(N, 3), dim=-1)
        for deg in range(4):
            out = _eval_sh(band0, rest, active_degree=deg, view_dirs=dirs)
            assert (out >= 0).all(), f"Negative values at degree {deg}"

    def test_degree0_uses_only_band0(self):
        """With degree=0 the higher-order coefficients must not affect the result."""
        N = 4
        band0, rest = make_sh(N)
        dirs = F.normalize(torch.randn(N, 3), dim=-1)
        out_zero_rest = _eval_sh(band0, torch.zeros_like(rest), active_degree=0, view_dirs=dirs)
        out_rand_rest = _eval_sh(band0, rest,                   active_degree=0, view_dirs=dirs)
        assert torch.allclose(out_zero_rest, out_rand_rest)

    def test_degree0_matches_formula(self):
        """C0 * band0[:,0,:] + 0.5 for degree 0."""
        N = 3
        band0, rest = make_sh(N)
        dirs = F.normalize(torch.randn(N, 3), dim=-1)
        expected = torch.clamp(sh_constants.C0 * band0[:, 0, :] + 0.5, min=0.0)
        out = _eval_sh(band0, rest, active_degree=0, view_dirs=dirs)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_higher_degrees_view_dependent(self):
        """Different view dirs should give different colors when degree > 0."""
        N = 4
        band0, rest = make_sh(N)
        dirs1 = F.normalize(torch.randn(N, 3), dim=-1)
        dirs2 = F.normalize(torch.randn(N, 3), dim=-1)
        out1 = _eval_sh(band0, rest, active_degree=1, view_dirs=dirs1)
        out2 = _eval_sh(band0, rest, active_degree=1, view_dirs=dirs2)
        assert not torch.allclose(out1, out2)

    def test_degree_monotone_uses_more_bands(self):
        """Adding bands can change the result (unless coefficients happen to be 0)."""
        N = 4
        band0, rest = make_sh(N, seed=42)
        dirs = F.normalize(torch.randn(N, 3), dim=-1)
        out1 = _eval_sh(band0, rest, active_degree=1, view_dirs=dirs)
        out2 = _eval_sh(band0, rest, active_degree=2, view_dirs=dirs)
        assert not torch.allclose(out1, out2)


# ── _compute_3d_covariance ────────────────────────────────────────────────────

class TestCompute3dCovariance:
    def _identity_params(self, N=4):
        scaling  = torch.zeros(N, 3)           # exp(0) = 1 → S = I
        rotation = torch.tensor([[1., 0., 0., 0.]]).expand(N, -1)  # identity quaternion
        return scaling, rotation

    def test_output_shape(self):
        N = 5
        scaling, rotation = self._identity_params(N)
        out = _compute_3d_covariance(scaling, rotation)
        assert out.shape == (N, 3, 3)

    def test_identity_rotation_scale_one(self):
        """R=I, S=I → Σ = I."""
        N = 3
        scaling, rotation = self._identity_params(N)
        cov = _compute_3d_covariance(scaling, rotation)
        I = torch.eye(3).unsqueeze(0).expand(N, -1, -1)
        assert torch.allclose(cov, I, atol=1e-6)

    def test_symmetry(self):
        """Σ must be symmetric: Σ = Σ^T."""
        N = 8
        scaling  = torch.randn(N, 3)
        rotation = torch.randn(N, 4)
        cov = _compute_3d_covariance(scaling, rotation)
        assert torch.allclose(cov, cov.transpose(-1, -2), atol=1e-6)

    def test_positive_semidefinite(self):
        """All eigenvalues of Σ must be ≥ 0."""
        N = 6
        scaling  = torch.randn(N, 3)
        rotation = torch.randn(N, 4)
        cov = _compute_3d_covariance(scaling, rotation)
        eigvals = torch.linalg.eigvalsh(cov)   # symmetric → real eigenvalues
        assert (eigvals >= -1e-5).all()

    def test_scaling_effect(self):
        """Uniform scale s → Σ = s^2 I."""
        N = 2
        s = 2.0
        scaling  = torch.full((N, 3), fill_value=np.log(s))  # exp(log s) = s
        rotation = torch.tensor([[1., 0., 0., 0.]]).expand(N, -1)
        cov = _compute_3d_covariance(scaling, rotation)
        expected = (s ** 2) * torch.eye(3).unsqueeze(0).expand(N, -1, -1)
        assert torch.allclose(cov, expected, atol=1e-5)

    def test_rotation_invariant_trace(self):
        """Rotating a sphere (S = sI) doesn't change Σ."""
        N = 4
        scaling  = torch.ones(N, 3)
        rot1 = torch.tensor([[1., 0., 0., 0.]]).expand(N, -1)
        rot2 = torch.randn(N, 4)   # arbitrary rotation
        cov1 = _compute_3d_covariance(scaling, rot1)
        cov2 = _compute_3d_covariance(scaling, rot2)
        # Trace is rotation-invariant
        assert torch.allclose(cov1.diagonal(dim1=-2, dim2=-1).sum(-1),
                               cov2.diagonal(dim1=-2, dim2=-1).sum(-1), atol=1e-5)


# ── _project_to_screen ────────────────────────────────────────────────────────

class TestProjectToScreen:
    def _simple_cov3d(self, N):
        """Isotropic unit covariance for all Gaussians."""
        return torch.eye(3).unsqueeze(0).expand(N, -1, -1).clone()

    def test_output_shapes(self):
        N = 5
        xyz   = torch.tensor([[0., 0., 2.]] * N)
        cov3d = self._simple_cov3d(N)
        cam   = make_camera()
        means2d, cov2d = _project_to_screen(xyz, cov3d, cam)
        assert means2d.shape == (N, 2)
        assert cov2d.shape   == (N, 3)

    def test_principal_axis_projects_to_principal_point(self):
        """A point on the optical axis (R=I, T=0) projects to (cx, cy)."""
        xyz   = torch.tensor([[0., 0., 3.]])
        cov3d = self._simple_cov3d(1)
        cam   = make_camera(fx=500., fy=500., cx=320., cy=240.)
        means2d, _ = _project_to_screen(xyz, cov3d, cam)
        assert torch.allclose(means2d[0], torch.tensor([320., 240.]), atol=1e-4)

    def test_perspective_divide(self):
        """fx * x/z + cx formula for off-axis point."""
        fx, fy, cx, cy = 400., 400., 200., 200.
        x, y, z = 2.0, 1.0, 4.0
        xyz   = torch.tensor([[x, y, z]])
        cov3d = self._simple_cov3d(1)
        cam   = make_camera(fx=fx, fy=fy, cx=cx, cy=cy)
        means2d, _ = _project_to_screen(xyz, cov3d, cam)
        expected_u = fx * x / z + cx
        expected_v = fy * y / z + cy
        assert abs(means2d[0, 0].item() - expected_u) < 1e-4
        assert abs(means2d[0, 1].item() - expected_v) < 1e-4

    def test_cov2d_symmetric(self):
        """Packed (a, b, c) → the full 2×2 is symmetric (b appears once but a=a, c=c)."""
        N = 8
        xyz   = torch.randn(N, 3)
        xyz[:, 2] = xyz[:, 2].abs() + 1.0   # keep in front of camera
        cov3d = self._simple_cov3d(N)
        cam   = make_camera()
        _, cov2d = _project_to_screen(xyz, cov3d, cam)
        # cov2d[:, 0] = σ_uu, [:, 1] = σ_uv, [:, 2] = σ_vv
        # Just verify shapes and that diagonal entries are > 0 (filter + PSD)
        assert (cov2d[:, 0] > 0).all()
        assert (cov2d[:, 2] > 0).all()

    def test_low_pass_filter_on_diagonal(self):
        """Diagonal of cov2d_full gets +0.3; with Σ_3d=0 should give exactly 0.3."""
        xyz   = torch.tensor([[0., 0., 5.]])
        cov3d = torch.zeros(1, 3, 3)           # degenerate Gaussian
        cam   = make_camera()
        _, cov2d = _project_to_screen(xyz, cov3d, cam)
        # With zero 3D covariance only the filter contributes to the diagonal
        assert abs(cov2d[0, 0].item() - 0.3) < 1e-5
        assert abs(cov2d[0, 2].item() - 0.3) < 1e-5

    def test_farther_point_smaller_projected_cov(self):
        """Moving a Gaussian farther away should shrink its projected covariance."""
        cov3d_near = self._simple_cov3d(1)
        cov3d_far  = self._simple_cov3d(1)
        cam = make_camera()
        _, cov_near = _project_to_screen(torch.tensor([[0., 0., 2.]]), cov3d_near, cam)
        _, cov_far  = _project_to_screen(torch.tensor([[0., 0., 10.]]), cov3d_far,  cam)
        # Trace of the 2×2 shrinks with distance
        trace_near = cov_near[0, 0] + cov_near[0, 2]
        trace_far  = cov_far[0, 0]  + cov_far[0, 2]
        assert trace_near > trace_far
