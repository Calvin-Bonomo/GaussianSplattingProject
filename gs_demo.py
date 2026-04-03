import argparse
import pycolmap
import numpy as np

from camera import Camera

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src", required=True, help="Path to COLMAP data")
    parser.add_argument("-d", "--dst", required=True, help="Path to output directory")
    parser.add_argument("--no-viewer", action="store_true", help="Disable the built-in viewer")
    return parser.parse_args()

def load_colmap_data(src_dir: str) -> tuple[np.ndarray, np.ndarray, list[Camera], list[np.ndarray]]:
    reconstruction = pycolmap.Reconstruction(src_dir)

    # Load in point-cloud data
    xyz_list = []
    rgb_list = []
    for _, point in reconstruction.points3D.items():
        xyz_list.append(point.xyz)
        rgb_list.append(point.rgb)

    xyz = np.ndarray(xyz_list)
    rgb = np.ndarray(rgb_list)

    # Load camera pose data
    return xyz, rgb, [], []
    
def run_gs_demo(xyz: np.ndarray, rgb: np.ndarray, cameras: list[Camera], images: list[np.ndarray], output_dir: str):
    pass

if __name__ == "__main__":
    args = parse_arguments()
    xyz, rgb, cameras, images = load_colmap_data(args.src)
    run_gs_demo(xyz, rgb, cameras, images, args.dst)
