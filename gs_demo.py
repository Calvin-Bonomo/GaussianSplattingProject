import argparse
import os
import random

import pycolmap
import numpy as np
from PIL import Image

import gaussian_splatting.hyperparameters as params
from gaussian_splatting.camera import Camera
from gaussian_splatting.gaussian_model import GaussianModel


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src", required=True, help="Path to COLMAP data")
    parser.add_argument("-d", "--dst", required=True, help="Path to output directory")
    parser.add_argument("--no-viewer", action="store_true", help="Disable the built-in viewer")
    return parser.parse_args()

def load_colmap_data(src_dir: str) -> tuple[np.ndarray, np.ndarray, list[Camera]]:
    reconstruction = pycolmap.Reconstruction(os.path.join(src_dir, "sparse"))

    # Load in point-cloud data
    xyz_list = []
    rgb_list = []
    for _, point in reconstruction.points3D.items():
        xyz_list.append(point.xyz)
        rgb_list.append(point.rgb)

    xyz = np.ndarray(xyz_list)
    rgb = np.ndarray(rgb_list)

    # Load camera pose data
    cameras = []
    image_path = os.path.join(src_dir, "images")

    for _, image in reconstruction.images.items():
        if image.camera == None: # Ignore images without cameras
            continue
        pose = image.cam_from_world
        camera = image.camera
        image = np.array(Image.open(os.path.join(image_path, image.name)).convert("RGB"))
        cameras.append(Camera(
            pose.rotation.quat,
            pose.translation,
            camera.focal,
            camera.width,
            camera.height,
            image))
    return xyz, rgb, cameras
    
def run_gs_demo(xyz: np.ndarray, rgb: np.ndarray, cameras: list[Camera], no_viewer: bool, output_dir: str):
    # Initialize gaussians
    gaussians = GaussianModel(xyz, rgb)

    for iteration in range(params.MAX_ITERATIONS):
        camera = cameras[random.randrange(len(cameras))]
        # rasterize image
        # Calculate loss
        # backprop
        if not no_viewer:
            pass # rasterize viewer and send to window

        if iteration % params.REFINEMENT_ITERATION_PERIOD != 0:
            continue
        # prune gaussians
        # densify gaussians

if __name__ == "__main__":
    args = parse_arguments()
    xyz, rgb, cameras = load_colmap_data(args.src)
    run_gs_demo(xyz, rgb, cameras, args.no_viewer, args.dst)

