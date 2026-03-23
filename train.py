"""
Main training script for 3D Gaussian Splatting.

Usage:
    python train.py -s <path_to_dataset> -m <output_path>

COLMAP datasets:
    python train.py -s data/garden -m output/garden --eval

Blender/NeRF-synthetic:
    python train.py -s data/lego -m output/lego --white_background
"""

import os
import sys
import uuid
import random
import torch
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.scene.scene import Scene
from gaussian_splatting.renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.utils.general_utils import safe_state


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]
    TENSORBOARD_FOUND = False


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # Background colour: white for synthetic, black for real
    bg_color = [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end   = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Exponential LR decay on position
        gaussians.update_learning_rate(iteration)

        # SH degree warmup: unlock one extra degree every 1000 iterations
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random training camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))

        # Enable debug kernels from a specified iteration onwards
        if iteration == debug_from:
            pipe.debug = True

        # ----------------------------------------------------------------
        # Forward pass
        # ----------------------------------------------------------------
        bg = torch.rand((3,), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image           = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii           = render_pkg["radii"]

        # ----------------------------------------------------------------
        # Loss
        # ----------------------------------------------------------------
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1  = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # EMA loss for display
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Logging
            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
                tb_writer.add_scalar('iter_time', iter_start.elapsed_time(iter_end), iteration)

            # ----------------------------------------------------------------
            # Densification
            # ----------------------------------------------------------------
            if iteration < opt.densify_until_iter:
                # Track 2D radii for pruning oversized splats
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if (iteration > opt.densify_from_iter and
                        iteration % opt.densification_interval == 0):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # ----------------------------------------------------------------
            # Optimiser step
            # ----------------------------------------------------------------
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # ----------------------------------------------------------------
            # Evaluation
            # ----------------------------------------------------------------
            if iteration in testing_iterations:
                _training_report(tb_writer, iteration, Ll1, loss,
                                 l1_loss, iter_start.elapsed_time(iter_end),
                                 testing_iterations, scene, render, pipe,
                                 background)

            # ----------------------------------------------------------------
            # Checkpoint / save
            # ----------------------------------------------------------------
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                           os.path.join(dataset.model_path,
                                        "chkpnt" + str(iteration) + ".pth"))


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND and SummaryWriter is not None:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def _training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
                     testing_iterations, scene, render_func, pipe, background):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Evaluate on test set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test',  'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                                          for idx in range(5, 30, 5)]},
        )
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = render_func(viewpoint, scene.gaussians, pipe, background)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt = viewpoint.original_image.to("cuda")
                    if tb_writer and idx < 5:
                        tb_writer.add_images(config['name'] + f"_view_{viewpoint.image_name}/render",
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + f"_view_{viewpoint.image_name}/gt",
                                                 gt[None], global_step=iteration)
                    from gaussian_splatting.utils.loss_utils import psnr
                    l1_test  += l1_loss(image, gt).mean().double()
                    psnr_test += psnr(image, gt).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test   /= len(config['cameras'])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.5f}  PSNR {psnr_test:.2f}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr',    psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram",
                                    scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="3D Gaussian Splatting Training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip',   type=str,  default="127.0.0.1")
    parser.add_argument('--port', type=int,  default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--test_iterations',  nargs="+", type=int,
                        default=[7000, 15000, 30000])
    parser.add_argument('--save_iterations',  nargs="+", type=int,
                        default=[7000, 30000])
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--checkpoint_iterations', nargs="+", type=int, default=[])
    parser.add_argument('--start_checkpoint', type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimising " + args.model_path)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations,
        args.checkpoint_iterations, args.start_checkpoint,
        args.debug_from,
    )

    print("\nTraining complete.")
