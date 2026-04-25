import numpy as np
import torch
from torch import nn, from_numpy

from .math_utils import quat_to_mat


class Camera:
    def __init__(self, rotation: np.ndarray, translation: np.ndarray, focal_x: float, focal_y: float, width: int, height: int, image: np.ndarray):
        self.viewTransform = self._create_view_transform(rotation, translation)
        self.focal_x = focal_x
        self.focal_y = focal_y
        self.width = width
        self.height = height
        self.image = image

    def _create_view_transform(self, rotation: np.ndarray, translation: np.ndarray):
        rot_mat = quat_to_mat(rotation)
        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = translation
        return nn.Parameter(from_numpy(transform).float())

