import numpy as np

from math_utils import quat_to_mat


class Camera:
    def __init__(self, rotation: np.ndarray, translation: np.ndarray, focal: np.ndarray, width: int, height: int, image: np.ndarray):
        self.viewTransform = self._create_view_transform(rotation, translation)
        self.focal = focal
        self.width = width
        self.height = height
        self.image = image

    def _create_view_transform(self, rotation: np.ndarray, translation: np.ndarray):
        rot_mat = quat_to_mat(rotation)
        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = translation
        return transform

