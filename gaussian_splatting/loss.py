import torch
import torch.nn.functional as F


def l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - gt).mean()


def _gaussian_kernel(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    x = torch.arange(window_size).float() - window_size // 2
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
    gauss /= gauss.sum()
    kernel = gauss.outer(gauss)
    return kernel.expand(channels, 1, window_size, window_size).contiguous()


def ssim(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Structural similarity index. pred and gt are (C, H, W) in [0, 1]."""
    C = pred.shape[0]
    pred = pred.unsqueeze(0)  # (1, C, H, W)
    gt   = gt.unsqueeze(0)

    window = _gaussian_kernel(window_size, sigma=1.5, channels=C).to(pred.device)
    pad    = window_size // 2

    mu1 = F.conv2d(pred,      window, padding=pad, groups=C)
    mu2 = F.conv2d(gt,        window, padding=pad, groups=C)

    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(gt   * gt,   window, padding=pad, groups=C) - mu2_sq
    sigma12   = F.conv2d(pred * gt,   window, padding=pad, groups=C) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12   + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    return (numerator / denominator).mean()
