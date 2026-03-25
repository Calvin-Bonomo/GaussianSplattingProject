import os
import struct
from dataclasses import dataclass

import numpy as np
from PIL import Image


# ── COLMAP binary readers ──────────────────────────────────────────────────────

# Number of intrinsic parameters per COLMAP camera model ID
_CAMERA_PARAM_COUNTS = {
    0: 3,   # SIMPLE_PINHOLE: f, cx, cy
    1: 4,   # PINHOLE: fx, fy, cx, cy
    2: 4,   # SIMPLE_RADIAL: f, cx, cy, k
    3: 5,   # RADIAL: f, cx, cy, k1, k2
    4: 8,   # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
    5: 8,   # OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
    6: 12,  # FULL_OPENCV
    7: 5,   # FOV: fx, fy, cx, cy, omega
    8: 4,   # SIMPLE_RADIAL_FISHEYE: f, cx, cy, k
    9: 5,   # RADIAL_FISHEYE: f, cx, cy, k1, k2
    10: 12, # THIN_PRISM_FISHEYE
}

# Models that share a single focal length (f = fx = fy)
_SINGLE_FOCAL_MODELS = {0, 2, 3, 8, 9}


def _read_cameras(path: str) -> dict[int, dict]:
    cameras = {}
    with open(path, 'rb') as f:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            cam_id  = struct.unpack('<I', f.read(4))[0]
            model   = struct.unpack('<i', f.read(4))[0]
            width, height = struct.unpack('<QQ', f.read(16))
            n = _CAMERA_PARAM_COUNTS[model]
            params = struct.unpack(f'<{n}d', f.read(8 * n))
            cameras[cam_id] = {'model': model, 'width': width, 'height': height, 'params': params}
    return cameras


def _read_images(path: str) -> dict[int, dict]:
    images = {}
    with open(path, 'rb') as f:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            img_id = struct.unpack('<I', f.read(4))[0]
            qvec   = np.array(struct.unpack('<4d', f.read(32)))  # qw, qx, qy, qz
            tvec   = np.array(struct.unpack('<3d', f.read(24)))
            cam_id = struct.unpack('<I', f.read(4))[0]
            name = b''
            while (c := f.read(1)) != b'\x00':
                name += c
            num_pts2d = struct.unpack('<Q', f.read(8))[0]
            f.read(num_pts2d * 24)  # skip 2D points (x: f64, y: f64, point3D_id: i64)
            images[img_id] = {'qvec': qvec, 'tvec': tvec, 'cam_id': cam_id, 'name': name.decode()}
    return images


def _read_points3d(path: str) -> tuple[np.ndarray, np.ndarray]:
    positions, colors = [], []
    with open(path, 'rb') as f:
        for _ in range(struct.unpack('<Q', f.read(8))[0]):
            f.read(8)                                        # point3D_id (u64)
            positions.append(struct.unpack('<3d', f.read(24)))
            colors.append(struct.unpack('<3B', f.read(3)))
            f.read(8)                                        # reprojection error (f64)
            track_len = struct.unpack('<Q', f.read(8))[0]
            f.read(track_len * 8)                            # skip track (image_id: u32, point2D_idx: u32)
    return (np.array(positions, dtype=np.float64),
            np.array(colors,    dtype=np.float32) / 255.0)


def _quat_to_rotation(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion (qw, qx, qy, qz) to a 3×3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def _extract_intrinsics(cam: dict) -> tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) from a COLMAP camera entry."""
    params, model = cam['params'], cam['model']
    if model in _SINGLE_FOCAL_MODELS:
        fx = fy = params[0]
        cx, cy  = params[1], params[2]
    else:
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    return fx, fy, cx, cy


# ── Dataset types ──────────────────────────────────────────────────────────────

@dataclass
class Camera:
    uid: int
    image: np.ndarray   # (H, W, 3) uint8 RGB
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray       # (3, 3) world-to-camera rotation
    T: np.ndarray       # (3,)   world-to-camera translation


@dataclass
class Dataset:
    cameras: list[Camera]
    point_positions: np.ndarray   # (N, 3) float64
    point_colors: np.ndarray      # (N, 3) float32 in [0, 1]


# ── Public loader ──────────────────────────────────────────────────────────────

def load_dataset(src_dir: str) -> Dataset:
    sparse_dir = os.path.join(src_dir, 'sparse', '0')
    images_dir = os.path.join(src_dir, 'images')

    colmap_cameras = _read_cameras(os.path.join(sparse_dir, 'cameras.bin'))
    colmap_images  = _read_images(os.path.join(sparse_dir, 'images.bin'))
    point_positions, point_colors = _read_points3d(os.path.join(sparse_dir, 'points3D.bin'))

    cameras = []
    for uid, img in colmap_images.items():
        cam = colmap_cameras[img['cam_id']]
        fx, fy, cx, cy = _extract_intrinsics(cam)
        image = np.array(Image.open(os.path.join(images_dir, img['name'])).convert('RGB'))
        cameras.append(Camera(
            uid=uid,
            image=image,
            width=cam['width'],
            height=cam['height'],
            fx=fx, fy=fy, cx=cx, cy=cy,
            R=_quat_to_rotation(img['qvec']),
            T=img['tvec'],
        ))

    return Dataset(cameras=cameras, point_positions=point_positions, point_colors=point_colors)
