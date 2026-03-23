"""Projection math sanity checks."""

import math
import numpy as np
import torch
import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gaussian_splatting.utils.graphics_utils import (
    focal2fov, fov2focal, getWorld2View2, getProjectionMatrix
)


def test_focal_fov_roundtrip():
    """focal2fov and fov2focal should be inverses."""
    focal = 800.0
    pixels = 1024
    fov = focal2fov(focal, pixels)
    recovered = fov2focal(fov, pixels)
    assert abs(recovered - focal) < 1e-4, f"{recovered} != {focal}"


def test_identity_view_matrix():
    """Identity rotation + zero translation should give the identity view matrix."""
    R = np.eye(3)
    T = np.zeros(3)
    W2C = getWorld2View2(R, T)
    np.testing.assert_allclose(W2C[:3, :3], R, atol=1e-6)
    np.testing.assert_allclose(W2C[:3, 3], T, atol=1e-6)
    assert W2C[3, 3] == 1.0


def test_projection_matrix_shape():
    """Projection matrix should be (4, 4) and have sensible values."""
    P = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=math.pi / 3, fovY=math.pi / 3)
    assert P.shape == (4, 4), f"Expected (4,4), got {P.shape}"
    # The (3, 2) entry carries the z_sign; should be 1 for our convention
    assert P[3, 2] == 1.0 or P[3, 2] == -1.0


def test_projection_matrix_point_at_near_plane():
    """A point exactly at z=znear in camera space should map to z_ndc ≈ 0."""
    znear, zfar = 0.01, 100.0
    fov = math.pi / 2  # 90 degrees: tan(fov/2) = 1
    P = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fov, fovY=fov)

    # Camera-space point at the near plane, at the optical axis
    p_cam = torch.tensor([0.0, 0.0, znear, 1.0])
    p_clip = P @ p_cam
    z_ndc = p_clip[2] / p_clip[3]
    # In the [0,1] z range, near maps to 0
    assert abs(float(z_ndc)) < 1e-4, f"z_ndc at near = {z_ndc}"


def test_world_to_camera_translation():
    """A camera translated 5 units along z should shift the z-coord of world points."""
    R = np.eye(3)
    T = np.array([0.0, 0.0, 5.0])  # camera at (0,0,5) in world, or t=(0,0,5) in camera space
    W2C = getWorld2View2(R, T)

    # A world point at the origin should appear at z=5 in camera space
    # (since T encodes the camera origin in camera space, not world space)
    p_world = np.array([0.0, 0.0, 0.0, 1.0])
    p_cam = W2C @ p_world
    assert abs(p_cam[2] - 5.0) < 1e-5, f"Expected z=5, got z={p_cam[2]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
