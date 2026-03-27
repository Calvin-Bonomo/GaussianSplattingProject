import argparse
import pycolmap
import numpy as np

from camera import Camera

def parse_arguments() -> argparse.Namespace:
    pass

def load_colmap_data(src_dir: str) -> tuple[np.ndarray, Camera]:
    reconstruction = pycolmap.Reconstruction(src_dir)
    
    for _, image in reconstruction.images.items():
        image
        

async def save_scene(output_dir: str):
    pass

def run_gs_demo(points: np.ndarray, cameras: Camera, output_dir: str):
    pass

if __name__ == "__main__":
    args = parse_arguments()
    sfm_points, cameras = load_colmap_data(args.src)
    run_gs_demo(sfm_points, cameras, args.dst)
