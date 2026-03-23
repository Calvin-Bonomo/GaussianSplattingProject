import torch
import numpy as np
from gaussian_splatting.utils.graphics_utils import getWorld2View2, getProjectionMatrix


ZNEAR = 0.01
ZFAR = 100.0


class Camera:
    """Stores camera parameters and precomputes transforms used by the rasterizer."""

    def __init__(self, uid, R, T, FoVx, FoVy, image, image_name,
                 trans=np.zeros(3), scale=1.0):
        """
        Args:
            uid: unique integer identifier
            R: (3, 3) world-to-camera rotation matrix (COLMAP convention)
            T: (3,) camera translation in camera space (T = R @ camera_center_world * -1)
            FoVx, FoVy: horizontal / vertical field of view in radians
            image: (C, H, W) float32 torch.Tensor in [0, 1]
            image_name: filename stem used for logging and saving
            trans: optional (3,) scene-normalisation translation
            scale: optional scene-normalisation scale
        """
        self.uid = uid
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        self.original_image = image.clamp(0.0, 1.0)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # 4x4 world-to-camera matrix (row-major, for right-multiplying column vectors)
        self.world_view_transform = torch.tensor(
            getWorld2View2(R, T, trans, scale)
        ).transpose(0, 1).cuda()

        # 4x4 perspective projection matrix
        self.projection_matrix = (
            getProjectionMatrix(znear=ZNEAR, zfar=ZFAR, fovX=FoVx, fovY=FoVy)
            .transpose(0, 1)
            .cuda()
        )

        # Combined world-to-clip matrix
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

        # Camera origin in world space: -R^T @ T
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @property
    def tanfovx(self):
        import math
        return math.tan(self.FoVx * 0.5)

    @property
    def tanfovy(self):
        import math
        return math.tan(self.FoVy * 0.5)
