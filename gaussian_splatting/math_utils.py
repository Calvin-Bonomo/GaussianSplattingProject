import numpy as np


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion to a 3x3 rotation matrix.

    Parameters:
        q: Quaternion as [x, y, z, w].

    Returns:
        3x3 rotation matrix.
    """
    x, y, z, w = q / np.linalg.norm(q)

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])
