"""Loaders for COLMAP and Blender/NeRF-synthetic datasets."""

import os
import json
import numpy as np
from pathlib import Path
from typing import NamedTuple, List
from PIL import Image

from gaussian_splatting.utils.graphics_utils import (
    BasicPointCloud, focal2fov, fov2focal, getWorld2View2
)


class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray       # (3, 3) world-to-camera rotation
    T: np.ndarray       # (3,) translation (camera space)
    FoVy: float
    FoVx: float
    image: Image.Image
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: List[CameraInfo]
    test_cameras: List[CameraInfo]
    nerf_normalization: dict
    ply_path: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def getNerfppNorm(cam_info: List[CameraInfo]) -> dict:
    """Compute scene normalisation: centre and radius from camera positions."""
    def get_center_and_diag(cam_centers):
        cam_centers = np.stack(cam_centers, axis=1)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    return {"translate": translate, "radius": radius}


# ---------------------------------------------------------------------------
# COLMAP loader
# ---------------------------------------------------------------------------

def qvec2rotmat(qvec):
    """Convert COLMAP quaternion (w, x, y, z) to 3x3 rotation matrix."""
    return np.array([
        [
            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
        ],
        [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
        ],
        [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
        ],
    ])


def read_colmap_binary_cameras(path):
    """Read cameras.bin from a COLMAP sparse model. Returns dict of camera_id -> dict."""
    import struct
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<I", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            # Model params: PINHOLE has 4 (fx, fy, cx, cy); SIMPLE_PINHOLE has 3
            if model_id == 0:  # SIMPLE_PINHOLE
                params = struct.unpack("<3d", f.read(24))
            elif model_id == 1:  # PINHOLE
                params = struct.unpack("<4d", f.read(32))
            else:
                raise ValueError(f"Unsupported COLMAP camera model id {model_id}")
            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": list(params),
            }
    return cameras


def read_colmap_binary_images(path):
    """Read images.bin from a COLMAP sparse model."""
    import struct
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")
            # Skip 2D keypoint data
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2D * 24)  # each point2D: x(8) y(8) point3D_id(8)
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }
    return images


def read_colmap_binary_points3D(path):
    """Read points3D.bin from a COLMAP sparse model."""
    import struct
    points = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point3D_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            track_length = struct.unpack("<Q", f.read(8))[0]
            f.read(track_length * 8)  # each track element: image_id(4) + point2D_idx(4)
            points[point3D_id] = {
                "xyz": np.array(xyz),
                "rgb": np.array(rgb, dtype=np.float32) / 255.0,
            }
    return points


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for img_id, img_data in cam_extrinsics.items():
        R = qvec2rotmat(img_data["qvec"])
        T = np.array(img_data["tvec"])

        cam = cam_intrinsics[img_data["camera_id"]]
        if cam["model_id"] == 0:  # SIMPLE_PINHOLE: f, cx, cy
            focal_length = cam["params"][0]
            FoVy = focal2fov(focal_length, cam["height"])
            FoVx = focal2fov(focal_length, cam["width"])
        else:  # PINHOLE: fx, fy, cx, cy
            focal_length_x = cam["params"][0]
            focal_length_y = cam["params"][1]
            FoVy = focal2fov(focal_length_y, cam["height"])
            FoVx = focal2fov(focal_length_x, cam["width"])

        image_path = os.path.join(images_folder, img_data["name"])
        image_name = Path(image_path).stem

        if not os.path.exists(image_path):
            # Try common extensions
            for ext in [".png", ".jpg", ".jpeg", ".JPG", ".PNG"]:
                candidate = os.path.splitext(image_path)[0] + ext
                if os.path.exists(candidate):
                    image_path = candidate
                    break

        image = Image.open(image_path)

        cam_infos.append(
            CameraInfo(
                uid=img_id,
                R=R,
                T=T,
                FoVy=FoVy,
                FoVx=FoVx,
                image=image,
                image_path=image_path,
                image_name=image_name,
                width=cam["width"],
                height=cam["height"],
            )
        )
    return sorted(cam_infos, key=lambda x: x.image_name)


def readColmapSceneInfo(path, images="images", eval=False, llffhold=8):
    """Load a COLMAP sparse model from `path/sparse/0/`."""
    sparse_dir = os.path.join(path, "sparse", "0")
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")
    images_bin = os.path.join(sparse_dir, "images.bin")
    points_bin = os.path.join(sparse_dir, "points3D.bin")

    cam_intrinsics = read_colmap_binary_cameras(cameras_bin)
    cam_extrinsics = read_colmap_binary_images(images_bin)
    points3D = read_colmap_binary_points3D(points_bin)

    images_folder = os.path.join(path, images)
    cam_infos_unsorted = readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder)
    cam_infos = sorted(cam_infos_unsorted, key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse", "0", "points3D.ply")
    if not os.path.exists(ply_path):
        # Convert points3D.bin to ply if the ply doesn't exist yet
        xyz = np.array([v["xyz"] for v in points3D.values()], dtype=np.float32)
        rgb = np.array([v["rgb"] for v in points3D.values()], dtype=np.float32)
        normals = np.zeros_like(xyz)
        _storePly(ply_path, xyz, rgb * 255)

    pcd = _fetchPly(ply_path)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


# ---------------------------------------------------------------------------
# Blender / NeRF-synthetic loader
# ---------------------------------------------------------------------------

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # Blender stores camera-to-world; invert to get world-to-camera
            c2w = np.array(frame["transform_matrix"])
            # Convert Blender camera convention (Y up, -Z forward) to OpenCV (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, 3]

            image = Image.open(cam_name)
            if white_background and image.mode == "RGBA":
                bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
                bg.paste(image, mask=image.split()[3])
                image = bg.convert("RGB")
            else:
                image = image.convert("RGB")

            FoVx = fovx
            FoVy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FoVy=FoVy,
                    FoVx=FoVx,
                    image=image,
                    image_path=cam_name,
                    image_name=Path(cam_name).stem,
                    width=image.size[0],
                    height=image.size[1],
                )
            )
    return cam_infos


def readNerfSyntheticInfo(path, white_background=False, eval=False, extension=".png"):
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json",
                                                white_background, extension)
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json",
                                               white_background, extension)
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Create random point cloud initialisation for synthetic scenes
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts} points) for Blender scene...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        _storePly(ply_path, xyz, shs * 255)

    pcd = _fetchPly(ply_path)

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )


# ---------------------------------------------------------------------------
# PLY I/O helpers
# ---------------------------------------------------------------------------

def _fetchPly(path):
    from plyfile import PlyData
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float32)
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.float32) / 255.0
    try:
        normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T.astype(np.float32)
    except Exception:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def _storePly(path, xyz, rgb):
    from plyfile import PlyData, PlyElement
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)
