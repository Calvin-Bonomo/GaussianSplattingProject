"""Tests for gaussian_splatting/loss.py"""

import torch

from gaussian_splatting.loss import l1_loss, _gaussian_kernel, ssim


# ── l1_loss ───────────────────────────────────────────────────────────────────

class TestL1Loss:
    def test_identical_inputs_zero(self):
        x = torch.rand(3, 64, 64)
        assert l1_loss(x, x).item() == 0.0

    def test_scalar_difference(self):
        pred = torch.ones(3, 4, 4)
        gt   = torch.zeros(3, 4, 4)
        assert abs(l1_loss(pred, gt).item() - 1.0) < 1e-6

    def test_nonnegative(self):
        pred = torch.randn(3, 32, 32)
        gt   = torch.randn(3, 32, 32)
        assert l1_loss(pred, gt).item() >= 0.0

    def test_symmetric(self):
        pred = torch.randn(3, 16, 16)
        gt   = torch.randn(3, 16, 16)
        assert torch.allclose(l1_loss(pred, gt), l1_loss(gt, pred))

    def test_returns_scalar(self):
        pred = torch.randn(3, 8, 8)
        gt   = torch.randn(3, 8, 8)
        out = l1_loss(pred, gt)
        assert out.shape == ()


# ── _gaussian_kernel ──────────────────────────────────────────────────────────

class TestGaussianKernel:
    def test_output_shape(self):
        k = _gaussian_kernel(window_size=11, sigma=1.5, channels=3)
        assert k.shape == (3, 1, 11, 11)

    def test_sums_to_one_per_channel(self):
        """Each channel's 2-D kernel must sum to 1 (normalized Gaussian)."""
        k = _gaussian_kernel(window_size=11, sigma=1.5, channels=4)
        for c in range(4):
            assert abs(k[c, 0].sum().item() - 1.0) < 1e-5

    def test_symmetric(self):
        """Kernel should be symmetric along both axes."""
        k = _gaussian_kernel(window_size=11, sigma=1.5, channels=1)
        kernel_2d = k[0, 0]
        assert torch.allclose(kernel_2d, kernel_2d.T, atol=1e-6)
        assert torch.allclose(kernel_2d, kernel_2d.flip(0), atol=1e-6)
        assert torch.allclose(kernel_2d, kernel_2d.flip(1), atol=1e-6)

    def test_center_is_max(self):
        """The center pixel should have the highest value."""
        k = _gaussian_kernel(window_size=11, sigma=1.5, channels=1)
        kernel_2d = k[0, 0]
        center = kernel_2d[5, 5]
        assert center == kernel_2d.max()

    def test_nonnegative(self):
        k = _gaussian_kernel(window_size=11, sigma=1.5, channels=2)
        assert (k >= 0).all()

    def test_single_channel(self):
        k = _gaussian_kernel(window_size=5, sigma=1.0, channels=1)
        assert k.shape == (1, 1, 5, 5)

    def test_channels_share_same_kernel(self):
        """All channels should contain the same 2-D kernel values."""
        k = _gaussian_kernel(window_size=7, sigma=1.5, channels=3)
        assert torch.allclose(k[0], k[1]) and torch.allclose(k[1], k[2])


# ── ssim ──────────────────────────────────────────────────────────────────────

class TestSSIM:
    def _img(self, val, C=3, H=64, W=64):
        return torch.full((C, H, W), val)

    def test_identical_inputs_near_one(self):
        """SSIM of an image with itself should be close to 1."""
        x = torch.rand(3, 64, 64)
        val = ssim(x, x).item()
        assert val > 0.99

    def test_range(self):
        """Result should lie in [-1, 1]."""
        pred = torch.rand(3, 64, 64)
        gt   = torch.rand(3, 64, 64)
        val  = ssim(pred, gt).item()
        assert -1.0 <= val <= 1.0

    def test_returns_scalar(self):
        pred = torch.rand(3, 32, 32)
        gt   = torch.rand(3, 32, 32)
        assert ssim(pred, gt).shape == ()

    def test_constant_images(self):
        """Two identical constant images → SSIM ≈ 1."""
        x = self._img(0.5)
        assert ssim(x, x).item() > 0.99

    def test_different_constants_lower_than_identical(self):
        """Dissimilar constant images should score lower than identical ones."""
        same = ssim(self._img(0.2), self._img(0.2)).item()
        diff = ssim(self._img(0.2), self._img(0.8)).item()
        assert same > diff

    def test_symmetric(self):
        """SSIM(pred, gt) == SSIM(gt, pred)."""
        pred = torch.rand(3, 64, 64)
        gt   = torch.rand(3, 64, 64)
        assert torch.allclose(ssim(pred, gt), ssim(gt, pred), atol=1e-6)

    def test_single_channel(self):
        pred = torch.rand(1, 64, 64)
        gt   = torch.rand(1, 64, 64)
        val  = ssim(pred, gt).item()
        assert -1.0 <= val <= 1.0

    def test_noise_reduces_ssim(self):
        """Adding noise to a clean image should lower SSIM vs the original."""
        clean = torch.rand(3, 64, 64)
        noisy = (clean + 0.3 * torch.randn_like(clean)).clamp(0, 1)
        assert ssim(clean, clean).item() > ssim(clean, noisy).item()
