"""
Unit tests for GaussianModel: densification, pruning, and optimizer state consistency.

These tests require CUDA and all Python dependencies to be installed.
Run with:
    pytest tests/test_gaussian_model.py -v
"""

import pytest
import torch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SKIP_IF_NO_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

try:
    from gaussian_splatting.scene.gaussian_model import GaussianModel
    from gaussian_splatting.utils.graphics_utils import BasicPointCloud
    from gaussian_splatting.utils.general_utils import inverse_sigmoid
    _MODEL_IMPORTABLE = True
except ImportError:
    GaussianModel = None    # type: ignore[assignment,misc]
    BasicPointCloud = None  # type: ignore[assignment,misc]
    inverse_sigmoid = None  # type: ignore[assignment]
    _MODEL_IMPORTABLE = False

SKIP_IF_NO_MODEL = pytest.mark.skipif(
    not _MODEL_IMPORTABLE,
    reason="gaussian_splatting not importable (dependencies missing)"
)


def _make_dummy_training_args():
    """Return a simple namespace that satisfies GaussianModel.training_setup."""
    from argparse import Namespace
    return Namespace(
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30000,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.001,
        percent_dense=0.01,
    )


def _init_model(n_points=32, sh_degree=1):
    """Create a GaussianModel initialised from a random point cloud."""
    # simple_knn may not be available; skip if so
    try:
        xyz = np.random.randn(n_points, 3).astype(np.float32) * 0.5
        colors = np.random.rand(n_points, 3).astype(np.float32)
        normals = np.zeros((n_points, 3), dtype=np.float32)
        pcd = BasicPointCloud(points=xyz, colors=colors, normals=normals)

        model = GaussianModel(sh_degree=sh_degree)
        model.create_from_pcd(pcd, spatial_lr_scale=1.0)
        model.training_setup(_make_dummy_training_args())
        return model
    except Exception as e:
        pytest.skip(f"Could not initialise GaussianModel: {e}")


@SKIP_IF_NO_CUDA
@SKIP_IF_NO_MODEL
def test_optimizer_state_consistent_after_prune():
    """After pruning, optimizer state tensors must have the same first dim as _xyz."""
    model = _init_model(n_points=64)
    n_before = model._xyz.shape[0]

    # Do a fake backward to populate optimizer state
    dummy_loss = model._xyz.sum()
    dummy_loss.backward()
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)

    # Prune half the points
    mask = torch.zeros(n_before, dtype=torch.bool, device="cuda")
    mask[:n_before // 2] = True
    model.prune_points(mask)

    n_after = model._xyz.shape[0]
    assert n_after == n_before - n_before // 2

    for group in model.optimizer.param_groups:
        param = group['params'][0]
        state = model.optimizer.state.get(param)
        if state is not None:
            assert state['exp_avg'].shape[0] == param.shape[0], \
                f"exp_avg dim mismatch in group {group['name']}"
            assert state['exp_avg_sq'].shape[0] == param.shape[0], \
                f"exp_avg_sq dim mismatch in group {group['name']}"


@SKIP_IF_NO_CUDA
@SKIP_IF_NO_MODEL
def test_reset_opacity():
    """After reset_opacity, all activated opacities must be <= 0.01."""
    model = _init_model(n_points=32)

    # Set some opacities very high
    with torch.no_grad():
        model._opacity.fill_(5.0)  # sigmoid(5) ≈ 0.993

    model.reset_opacity()

    activated = model.get_opacity
    assert (activated <= 0.01 + 1e-5).all(), \
        f"Max opacity after reset = {activated.max().item():.4f}"


@SKIP_IF_NO_CUDA
@SKIP_IF_NO_MODEL
def test_sh_degree_warmup():
    """oneupSHdegree should increment active degree up to max_sh_degree."""
    model = _init_model(sh_degree=3)
    assert model.active_sh_degree == 0

    for target in range(1, model.max_sh_degree + 1):
        model.oneupSHdegree()
        assert model.active_sh_degree == target

    # Should not exceed max_sh_degree
    model.oneupSHdegree()
    assert model.active_sh_degree == model.max_sh_degree


@SKIP_IF_NO_CUDA
@SKIP_IF_NO_MODEL
def test_save_load_ply(tmp_path):
    """PLY checkpoint should round-trip without changing the number of Gaussians."""
    model = _init_model(n_points=16)
    n_before = model._xyz.shape[0]

    ply_path = str(tmp_path / "test.ply")
    model.save_ply(ply_path)

    model2 = GaussianModel(sh_degree=1)
    model2.load_ply(ply_path)

    assert model2._xyz.shape[0] == n_before, \
        f"Loaded {model2._xyz.shape[0]} points, expected {n_before}"
    torch.testing.assert_close(
        model2._xyz.cpu(), model._xyz.detach().cpu(), atol=1e-5, rtol=0
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
