import torch
import sys
import os

import pybind11
sys.path.append(os.path.join(os.path.dirname(__file__), "build"))
import gs_rasterizer


class RenderPass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gaussian_input, camera, render_settings):
        image = gs_rasterizer.forward(
                    gaussian_input.mean,
                    gaussian_input.scale,
                    gaussian_input.rotation,
                    gaussian_input.color,
                    gaussian_input.opacity,
                    camera.viewTransform,
                    camera.focal_x,
                    camera.focal_y,
                    render_settings.zNear,
                    render_settings.zFar,
                    camera.width,
                    camera.height)
        return image

    @staticmethod
    def backward(ctx, grad_output):
        pass

def render_frame(gaussians, camera, render_settings):
    return RenderPass.apply(gaussians, camera, render_settings)


