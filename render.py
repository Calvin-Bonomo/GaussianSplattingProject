"""
Offline rendering script: loads a trained model and renders all cameras.

Usage:
    python render.py -m <model_path> [--iteration 30000] [--skip_train] [--skip_test]
"""

import os
import torch
import torchvision
from tqdm import tqdm
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.scene.scene import Scene
from gaussian_splatting.renderer import render
from gaussian_splatting.utils.general_utils import safe_state


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path    = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendered   = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt         = view.original_image[0:3, :, :]

        torchvision.utils.save_image(rendered, os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt,       os.path.join(gts_path,    f"{idx:05d}.png"))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter,
                       scene.getTrainCameras(), gaussians, pipeline, background)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter,
                       scene.getTestCameras(), gaussians, pipeline, background)


if __name__ == "__main__":
    parser = ArgumentParser(description="Render a trained 3DGS model")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration",  default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test",  action="store_true")
    parser.add_argument("--quiet",      action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args),
                args.skip_train, args.skip_test)
