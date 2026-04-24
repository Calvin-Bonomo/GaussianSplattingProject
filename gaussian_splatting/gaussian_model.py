import torch
from torch import nn, optim, from_numpy
import numpy as np


class GaussianModel:
    def __init__(self, xyz: np.ndarray, rgb: np.ndarray):
        # Prepare data
        num_points = xyz.shape[0]
        rotations = torch.zeros(num_points, 4, dtype=torch.float32)
        rotations[:, 0] = 1.0

        # Setup the gaussian parameters
        self.mean = nn.Parameter(from_numpy(xyz).float().cuda().requires_grad_(True))
        self.scale = nn.Parameter(torch.ones(num_points, 3, dtype=torch.float32).float().cuda().requires_grad_(True))
        self.rotation = nn.Parameter(rotations.cuda().requires_grad_(True))
        self.color = nn.Parameter(from_numpy(rgb).float().cuda().requires_grad_(True))
        self.opacity = nn.Parameter(0.1 * torch.ones(num_points))

        # Setup the optimizer
        self.optimizer = optim.adam.Adam() # TODO: Setup learning rates
