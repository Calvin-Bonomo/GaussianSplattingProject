"""GaussianModel: the learnable 3D Gaussian scene representation."""

import torch
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import KDTree as _KDTree

from gaussian_splatting.utils.general_utils import (
    inverse_sigmoid, build_covariance_from_scaling_rotation, get_expon_lr_func
)
from gaussian_splatting.utils.sh_utils import RGB2SH


class GaussianModel:
    """Stores and optimises the 3D Gaussian parameters."""

    # --------------------------------------------------------------------------
    # Property accessors: return activated parameter values
    # --------------------------------------------------------------------------

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1.0):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    # --------------------------------------------------------------------------
    # Initialisation
    # --------------------------------------------------------------------------

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        # Learnable parameters (all nn.Parameter so autograd tracks them)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        # Activations
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        # Gradient accumulation for adaptive density control
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0.0
        self.spatial_lr_scale = 0.0

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd, spatial_lr_scale: float):
        """Initialise Gaussians from a point cloud."""
        self.spatial_lr_scale = spatial_lr_scale

        points = torch.tensor(pcd.points, dtype=torch.float, device="cuda")
        colors = torch.tensor(pcd.colors, dtype=torch.float, device="cuda")

        print(f"Number of points at initialisation: {points.shape[0]}")

        # Initialise scales from mean nearest-neighbour distance
        # Compute mean squared distance to the 3 nearest neighbours (on CPU via scipy)
        pts_np = pcd.points  # already float32 numpy array
        kd = _KDTree(pts_np)
        dists, _ = kd.query(pts_np, k=4)  # k=4: self (dist=0) + 3 neighbours
        mean_dist = dists[:, 1:].mean(axis=1).astype(np.float32)  # exclude self
        dist2 = torch.clamp_min(
            torch.tensor(mean_dist ** 2, device="cuda"), 0.0000001
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        # Identity rotation quaternion (w=1, x=y=z=0)
        rots = torch.zeros((points.shape[0], 4), device="cuda")
        rots[:, 0] = 1.0

        # Low initial opacity
        opacities = inverse_sigmoid(
            0.1 * torch.ones((points.shape[0], 1), dtype=torch.float, device="cuda")
        )

        # Convert RGB to SH DC coefficients
        features = torch.zeros((points.shape[0], 3, (self.max_sh_degree + 1) ** 2),
                               dtype=torch.float, device="cuda")
        features[:, :3, 0] = RGB2SH * colors  # DC term — store colour in DC coefficient
        features = features.transpose(1, 2)   # (N, (d+1)^2, 3)
        features_dc   = features[:, 0:1, :]   # (N, 1, 3)
        features_rest = features[:, 1:, :]    # (N, (d+1)^2 - 1, 3)

        self._xyz          = torch.nn.Parameter(points.requires_grad_(True))
        self._features_dc  = torch.nn.Parameter(features_dc.contiguous().requires_grad_(True))
        self._features_rest = torch.nn.Parameter(features_rest.contiguous().requires_grad_(True))
        self._scaling      = torch.nn.Parameter(scales.requires_grad_(True))
        self._rotation     = torch.nn.Parameter(rots.requires_grad_(True))
        self._opacity      = torch.nn.Parameter(opacities.requires_grad_(True))

        N = points.shape[0]
        self.max_radii2D       = torch.zeros(N, device="cuda")
        self.xyz_gradient_accum = torch.zeros((N, 1), device="cuda")
        self.denom             = torch.zeros((N, 1), device="cuda")

    # --------------------------------------------------------------------------
    # Optimizer setup
    # --------------------------------------------------------------------------

    def training_setup(self, training_args):
        assert self.optimizer is None, "training_setup called twice"
        self.percent_dense = training_args.percent_dense

        l = [
            {'params': [self._xyz],           'lr': training_args.position_lr_init * self.spatial_lr_scale,
             'name': 'xyz'},
            {'params': [self._features_dc],   'lr': training_args.feature_lr,    'name': 'f_dc'},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, 'name': 'f_rest'},
            {'params': [self._opacity],       'lr': training_args.opacity_lr,    'name': 'opacity'},
            {'params': [self._scaling],       'lr': training_args.scaling_lr,    'name': 'scaling'},
            {'params': [self._rotation],      'lr': training_args.rotation_lr,   'name': 'rotation'},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15, fused=True)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Decay position learning rate."""
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'xyz':
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    # --------------------------------------------------------------------------
    # Densification utilities
    # --------------------------------------------------------------------------

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """Accumulate the 2D positional gradient norm for visible Gaussians."""
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """Split large Gaussians with high gradient into N smaller children."""
        # Normalise accumulated gradient
        grads_norm = grads / (self.denom + 1e-7)
        grads_norm[grads_norm.isnan()] = 0.0

        # Mask: high gradient AND scale larger than percent_dense * scene_extent
        selected = torch.where(
            (grads_norm >= grad_threshold).squeeze() &
            (torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent),
            True, False
        )

        # Sample N children per selected Gaussian from the parent's distribution
        stds = self.get_scaling[selected].repeat(N, 1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = _build_rotation(self._rotation[selected]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected].repeat(N, 1)

        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected].repeat(N, 1) / (0.8 * N)
        )
        new_rotation     = self._rotation[selected].repeat(N, 1)
        new_features_dc  = self._features_dc[selected].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected].repeat(N, 1, 1)
        new_opacity      = self._opacity[selected].repeat(N, 1)

        self._densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacity, new_scaling, new_rotation
        )

        # Prune the selected (parent) Gaussians
        prune_filter = torch.cat((
            selected,
            torch.zeros(N * selected.sum(), device="cuda", dtype=bool)
        ))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """Clone under-reconstructed Gaussians (high gradient, small scale)."""
        grads_norm = grads / (self.denom + 1e-7)
        grads_norm[grads_norm.isnan()] = 0.0

        selected = torch.where(
            (grads_norm >= grad_threshold).squeeze() &
            (torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent),
            True, False
        )

        new_xyz          = self._xyz[selected]
        new_features_dc  = self._features_dc[selected]
        new_features_rest = self._features_rest[selected]
        new_opacities    = self._opacity[selected]
        new_scaling      = self._scaling[selected]
        new_rotation     = self._rotation[selected]

        self._densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacities, new_scaling, new_rotation
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """Full adaptive density control step."""
        grads = self.xyz_gradient_accum.clone()

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # Prune Gaussians that are too transparent or too large
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size is not None:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune_points(self, mask):
        """Remove Gaussians where mask is True.  Also prunes optimizer state."""
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz          = optimizable_tensors["xyz"]
        self._features_dc  = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity      = optimizable_tensors["opacity"]
        self._scaling      = optimizable_tensors["scaling"]
        self._rotation     = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom              = self.denom[valid_points_mask]
        self.max_radii2D        = self.max_radii2D[valid_points_mask]

    def reset_opacity(self):
        """Periodically reset opacities to prevent accumulation artefacts."""
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self._replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # --------------------------------------------------------------------------
    # Checkpoint I/O
    # --------------------------------------------------------------------------

    def save_ply(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz     = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc    = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        f_rest  = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale   = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        def construct_list_of_attributes():
            l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            for i in range(f_dc.shape[1]):     l.append(f'f_dc_{i}')
            for i in range(f_rest.shape[1]):   l.append(f'f_rest_{i}')
            l.append('opacity')
            for i in range(scale.shape[1]):    l.append(f'scale_{i}')
            for i in range(rotation.shape[1]): l.append(f'rot_{i}')
            return l

        dtype_full = [(attr, 'f4') for attr in construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz      = np.stack([np.asarray(plydata.elements[0][ax]) for ax in ['x', 'y', 'z']], axis=1)
        opacities = np.asarray(plydata.elements[0]['opacity'])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]['f_dc_0'])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]['f_dc_1'])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]['f_dc_2'])

        extra_f_names = [p.name for p in plydata.elements[0].properties
                         if p.name.startswith('f_rest_')]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = sorted([p.name for p in plydata.elements[0].properties
                               if p.name.startswith('scale_')], key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = sorted([p.name for p in plydata.elements[0].properties
                             if p.name.startswith('rot')], key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz          = torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc  = torch.nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = torch.nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity      = torch.nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling      = torch.nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation     = torch.nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    # --------------------------------------------------------------------------
    # Private helpers
    # --------------------------------------------------------------------------

    def _densification_postfix(self, new_xyz, new_f_dc, new_f_rest, new_opacities, new_scaling, new_rotation):
        """Concatenate new Gaussians and extend optimizer state."""
        d = {
            "xyz": new_xyz,
            "f_dc": new_f_dc,
            "f_rest": new_f_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self._cat_tensors_to_optimizer(d)

        self._xyz          = optimizable_tensors["xyz"]
        self._features_dc  = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity      = optimizable_tensors["opacity"]
        self._scaling      = optimizable_tensors["scaling"]
        self._rotation     = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom              = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D        = torch.zeros(self._xyz.shape[0], device="cuda")

    def _prune_optimizer(self, mask):
        assert self.optimizer is not None
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg']    = stored_state['exp_avg'][mask]
                stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]
                del self.optimizer.state[group['params'][0]]
                group['params'][0] = torch.nn.Parameter(group['params'][0][mask].requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
            else:
                group['params'][0] = torch.nn.Parameter(group['params'][0][mask].requires_grad_(True))
            optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def _cat_tensors_to_optimizer(self, tensors_dict):
        assert self.optimizer is not None
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group['params']) == 1
            extension_tensor = tensors_dict[group['name']]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg']    = torch.cat(
                    (stored_state['exp_avg'],    torch.zeros_like(extension_tensor)), dim=0)
                stored_state['exp_avg_sq'] = torch.cat(
                    (stored_state['exp_avg_sq'], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group['params'][0] = torch.nn.Parameter(
                    torch.cat((group['params'][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group['params'][0]] = stored_state
            else:
                group['params'][0] = torch.nn.Parameter(
                    torch.cat((group['params'][0], extension_tensor), dim=0).requires_grad_(True)
                )
            optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def _replace_tensor_to_optimizer(self, tensor, name):
        assert self.optimizer is not None
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state['exp_avg']    = torch.zeros_like(tensor)
                    stored_state['exp_avg_sq'] = torch.zeros_like(tensor)
                    del self.optimizer.state[group['params'][0]]
                    group['params'][0] = torch.nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state
                else:
                    group['params'][0] = torch.nn.Parameter(tensor.requires_grad_(True))
                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors


def _build_rotation(r):
    """Quaternion (w, x, y, z) -> (N, 3, 3) rotation matrices (normalised)."""
    norm = torch.sqrt(r[:, 0]**2 + r[:, 1]**2 + r[:, 2]**2 + r[:, 3]**2)
    q = r / norm[:, None]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.zeros((q.size(0), 3, 3), device=r.device)
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R
