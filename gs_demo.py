import argparse
import os
import random

import cv2
import pycolmap
import numpy as np
from PIL import Image

import gaussian_splatting.hyperparameters as params
from gaussian_splatting.camera import Camera
from gaussian_splatting.gaussian_model import GaussianModel
from gaussian_splatting.render_settings import RenderSettings
import rasterizer.rasterizer as engine


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src", required=True, help="Path to COLMAP data")
    parser.add_argument("-d", "--dst", required=True, help="Path to output directory")
    parser.add_argument("--no-viewer", action="store_true", help="Disable the built-in viewer")
    parser.add_argument("--test-rasterizer", action="store_true", help="Test the rasterizer")
    return parser.parse_args()

def load_colmap_data(src_dir: str) -> tuple[np.ndarray, np.ndarray, list[Camera]]:
    reconstruction = pycolmap.Reconstruction(os.path.abspath(os.path.join(src_dir, "sparse/0/")))

    # Load in point-cloud data
    xyz_list = []
    rgb_list = []
    for _, point in reconstruction.points3D.items():
        xyz_list.extend(point.xyz.tolist())
        rgb_list.extend(point.color.tolist())

    xyz = np.array(xyz_list)
    rgb = np.array(rgb_list)

    # Load camera pose data
    cameras = []
    image_path = os.path.join(src_dir, "images")

    for _, image in reconstruction.images.items():
        if image.camera == None: # Ignore images without cameras
            continue
        pose = image.cam_from_world()
        camera = image.camera
        image = np.array(Image.open(os.path.join(image_path, image.name)).convert("RGB"))
        cameras.append(Camera(
            pose.rotation.quat,
            pose.translation,
            camera.focal_length_x,
            camera.focal_length_y,
            camera.width,
            camera.height,
            image))
    return xyz, rgb, cameras
    
def run_gs_demo(xyz: np.ndarray, rgb: np.ndarray, cameras: list[Camera], no_viewer: bool, test_rasterizer: bool, output_dir: str):
    # Initialize gaussians
    gaussians = GaussianModel(xyz, rgb)
    render_settings = RenderSettings()

    if test_rasterizer:
        frame_times_ms = []
        for iteration in range(params.MAX_ITERATIONS):
            if iteration % 500 == 0:
                print(f"Completed {iteration}/{params.MAX_ITERATIONS}")
            camera = cameras[random.randrange(len(cameras))]
            _, frame_time = engine.render_frame(gaussians, camera, render_settings)
            frame_times_ms.append(frame_time)
        frame_times_ms = np.array(frame_times_ms)
        print(f"Mean frame time (ms): {np.mean(frame_times_ms)}")
        print(f"Median frame time (ms): {np.median(frame_times_ms)}")
    else:
        for iteration in range(params.MAX_ITERATIONS):
            camera = cameras[random.randrange(len(cameras))]
            image, _ = engine.render_frame(gaussians, camera, render_settings)
            cpu_image = image.to('cpu').numpy()
            cpu_image.reshape(camera.height, camera.width, 3)
            cv2.imshow('render', cv2.cvtColor(cpu_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
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
    run_gs_demo(xyz, rgb, cameras, args.no_viewer, args.test_rasterizer, args.dst)

