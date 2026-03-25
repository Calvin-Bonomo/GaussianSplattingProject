"""Tests for gaussian_splatting/math_utils.py"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from gaussian_splatting.math_utils import quat_to_rotation_np, quat_to_rotation


# ── helpers ───────────────────────────────────────────────────────────────────

def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle / 2)
    return np.array([np.cos(angle / 2), axis[0]*s, axis[1]*s, axis[2]*s])


# ── shared adapter fixture ────────────────────────────────────────────────────
# Normalises both backends to the same signature:
#   (qvec: np.ndarray shape (4,)) -> np.ndarray shape (3, 3)
# so every correctness test below runs for both implementations.

@pytest.fixture(params=["np", "torch"])
def quat_rot(request):
    if request.param == "np":
        return quat_to_rotation_np
    def _torch(q):
        t = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
        return quat_to_rotation(t)[0].numpy()
    return _torch


# ── correctness tests (parametrised over both backends) ───────────────────────

def test_identity(quat_rot):
    assert np.allclose(quat_rot(np.array([1., 0., 0., 0.])), np.eye(3), atol=1e-6)

def test_output_shape(quat_rot):
    assert quat_rot(np.array([1., 0., 0., 0.])).shape == (3, 3)

def test_det_is_one(quat_rot):
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.standard_normal(4); q /= np.linalg.norm(q)
        assert abs(np.linalg.det(quat_rot(q)) - 1.0) < 1e-5

def test_orthogonal(quat_rot):
    rng = np.random.default_rng(1)
    q = rng.standard_normal(4); q /= np.linalg.norm(q)
    R = quat_rot(q)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-5)

@pytest.mark.parametrize("axis,angle,expected", [
    ([1, 0, 0], np.pi / 2, rot_x(np.pi / 2)),
    ([0, 1, 0], np.pi / 2, rot_y(np.pi / 2)),
    ([0, 0, 1], np.pi / 2, rot_z(np.pi / 2)),
    ([1, 0, 0], np.pi,     rot_x(np.pi)),
])
def test_known_rotations(quat_rot, axis, angle, expected):
    assert np.allclose(quat_rot(quat_from_axis_angle(axis, angle)), expected, atol=1e-6)

def test_antipodal_same_rotation(quat_rot):
    """q and -q represent the same rotation."""
    q = quat_from_axis_angle([1, 1, 0], np.pi / 3)
    assert np.allclose(quat_rot(q), quat_rot(-q), atol=1e-6)

def test_inverse_is_transpose(quat_rot):
    q = quat_from_axis_angle([1, 2, 3], np.pi / 5)
    q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
    assert np.allclose(quat_rot(q_inv), quat_rot(q).T, atol=1e-6)

def test_composition(quat_rot):
    """R(q1 * q2) == R(q1) @ R(q2)."""
    q1 = quat_from_axis_angle([1, 0, 0], np.pi / 4)
    q2 = quat_from_axis_angle([0, 1, 0], np.pi / 3)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    q12 = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
    assert np.allclose(quat_rot(q12), quat_rot(q1) @ quat_rot(q2), atol=1e-6)


# ── torch-only tests ──────────────────────────────────────────────────────────

def test_torch_output_shape():
    q = F.normalize(torch.randn(7, 4), dim=-1)
    assert quat_to_rotation(q).shape == (7, 3, 3)

def test_torch_batched_matches_np():
    """Each row of the batched result must equal the numpy result for that row."""
    rng = np.random.default_rng(2)
    qs = rng.standard_normal((6, 4))
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    Rs_torch = quat_to_rotation(torch.tensor(qs, dtype=torch.float32)).numpy()
    for i, q in enumerate(qs):
        assert np.allclose(Rs_torch[i], quat_to_rotation_np(q), atol=1e-5)
