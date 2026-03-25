import argparse
import os
import random

import torch

from gaussian_splatting import config
from gaussian_splatting.loss import l1_loss, ssim
from gaussian_splatting.scene.dataset import Dataset, load_dataset
from gaussian_splatting.scene.gaussian_model import GaussianModel  # not yet implemented
from gaussian_splatting.renderer import render                      # not yet implemented



# ── Main functions ─────────────────────────────────────────────────────────────

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='3D Gaussian Splatting trainer')
    parser.add_argument('-s', '--source', required=True, help='Path to input dataset directory')
    parser.add_argument('-o', '--dst', required=True, help='Path to output directory')
    parser.add_argument('--no_viewer', action='store_true', help='Disable the built-in training viewer')
    return parser.parse_args()


def run_gs(data: Dataset) -> GaussianModel:
    gaussians = GaussianModel(sh_degree=config.SH_DEGREE)
    gaussians.create_from_pcd(data.point_positions, data.point_colors)

    for iteration in range(1, config.ITERATIONS + 1):
        camera = random.choice(data.cameras)

        # Forward pass
        render_pkg = render(camera, gaussians)
        image = render_pkg['render']
        gt    = torch.from_numpy(camera.image).float().permute(2, 0, 1) / 255.0

        loss = (1 - config.LAMBDA_DSSIM) * l1_loss(image, gt) \
                     + config.LAMBDA_DSSIM * (1 - ssim(image, gt))
        # Backward pass
        loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad()

        # Unlock next SH band every N iterations
        if iteration % config.SH_UPSCALE_EVERY == 0:
            gaussians.oneupSHdegree()

        # Adaptive density control
        if config.DENSIFY_FROM <= iteration <= config.DENSIFY_UNTIL and iteration % config.DENSIFY_INTERVAL == 0:
            gaussians.densify_and_prune(
                render_pkg['viewspace_points'],
                render_pkg['visibility_filter'],
                render_pkg['radii'],
            )

        if iteration % config.OPACITY_RESET_EVERY == 0:
            gaussians.reset_opacity()

    return gaussians


def save_data(dst_dir: str, data: GaussianModel) -> None:
    path = os.path.join(dst_dir, 'point_cloud', f'iteration_{config.ITERATIONS}', 'point_cloud.ply')
    data.save_ply(path)
    print(f"Saved {len(data._xyz)} Gaussians to {path}")


if __name__ == "__main__":
    assert torch.cuda.is_available()  # We need CUDA to do anything

    args = parse_arguments()
    dataset = load_dataset(args.source)
    trained_gaussians = run_gs(dataset)
    save_data(args.dst, trained_gaussians)
