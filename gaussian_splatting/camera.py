import numpy as np

from math_utils import quat_to_mat


class Camera:
    def __init__(self, rotation: np.ndarray, translation: np.ndarray, focal: np.ndarray, width: int, height: int, image: np.ndarray):
        self.transform_3D = self._create_T3D(rotation, translation, focal)
        self.focal = focal
        self.width = width
        self.height = height
        self.image = image

    def _create_T3D(self, rotation: np.ndarray, translation: np.ndarray, focal: np.ndarray):
        """
        Creates the 3D transformation described in the EWA splatting paper.

        Parameters:
            rotation: A quaternion in the form [x, y, z, w]
            translation: A 3-vector describing the position of the camera
            focal: A 2-vector describing the focal-length of the camera

        Returns:
            A 2x3 matrix
        """
        # Create viewing transform (W)
        rot_mat = quat_to_mat(rotation)
        
        # Create projection matrix (J)
        u2_sqrd = np.pow(translation[2], 2)
        projection_mat = [
                [focal[0] / translation[2], 0, -focal[0] * translation[0] / u2_sqrd],
                [0, focal[1] / translation[2], -focal[1] * translation[1] / u2_sqrd]]

        return projection_mat @ rot_mat
