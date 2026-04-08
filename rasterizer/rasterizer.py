import torch

from ..gaussian_splatting.gaussian_model import GaussianModel
from ..gaussian_splatting.camera import Camera
import build.GAUSSIAN_RENDERER as renderer


class RenderPass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass

def render_frame(gaussians: GaussianModel, camera: Camera):
    RenderPass.apply(gaussians, camera)


