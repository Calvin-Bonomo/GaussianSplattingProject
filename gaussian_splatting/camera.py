import numpy as np

from math_utils import quat_to_mat


class Camera:
    def __init__(self, rotation: np.ndarray, translation: np.ndarray, focal: np.ndarray, image: np.ndarray):
        self.transform_3D = self._create_T3D(rotation, translation, focal)
        self.image = image

    def _create_T3D(self, rotation: np.ndarray, translation: np.ndarray, focal: np.ndarray):
        # Create viewing transform (W)
        transform = np.identity(4)

        rot_mat = quat_to_mat(rotation)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = translation.T
        
        # Create projection matrix (J)
        
