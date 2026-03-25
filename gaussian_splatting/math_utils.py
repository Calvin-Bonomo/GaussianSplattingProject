import numpy as np
import torch


def quat_to_rotation_np(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion (qw, qx, qy, qz) to a 3×3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def quat_to_rotation(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions (N, 4) [w, x, y, z] to rotation matrices (N, 3, 3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),   2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
