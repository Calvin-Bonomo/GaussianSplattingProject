import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gaussian_splatting import config, sh_constants



def _nn_mean_dist(positions: np.ndarray, k: int = 3) -> torch.Tensor:
    """Mean distance to the k nearest neighbors for each point, used to initialize scale."""
    pts = torch.from_numpy(positions).float().cuda()
    chunk_size = 2048
    dists = []
    for i in range(0, len(pts), chunk_size):
        d = torch.cdist(pts[i:i + chunk_size], pts)
        d[d < 1e-10] = float('inf')                   # exclude self-distances
        knn = d.topk(k, dim=1, largest=False).values  # (chunk, k)
        dists.append(knn.mean(dim=1))
    return torch.cat(dists)


class GaussianModel:
    def __init__(self, sh_degree: int):
        self.max_sh_degree    = sh_degree
        self.active_sh_degree = 0

        self._xyz:           nn.Parameter = None
        self._sh_band0:      nn.Parameter = None
        self._sh_bands_rest: nn.Parameter = None
        self._scaling:       nn.Parameter = None
        self._rotation:      nn.Parameter = None
        self._opacity:       nn.Parameter = None

        self.optimizer: torch.optim.Adam = None

    def create_from_pcd(self, positions: np.ndarray, colors: np.ndarray) -> None:
        xyz = torch.tensor(positions, dtype=torch.float32).cuda()

        # Colors → SH DC coefficient via: color = sh * C0 + 0.5
        sh_band0 = ((torch.tensor(colors, dtype=torch.float32) - 0.5) / sh_constants.C0)
        sh_band0 = sh_band0.unsqueeze(1).cuda()   # (N, 1, 3)

        # Higher-order SH bands start at zero and are unlocked gradually via oneupSHdegree()
        num_sh_rest = (self.max_sh_degree + 1) ** 2 - 1   # 15 for degree 3
        sh_bands_rest = torch.zeros(len(xyz), num_sh_rest, 3).cuda()

        # Scale: log of mean nearest-neighbor distance, isotropic across all 3 axes
        mean_dist = _nn_mean_dist(positions)
        scaling = torch.log(torch.sqrt(torch.clamp(mean_dist, min=1e-10)))
        scaling = scaling.unsqueeze(1).repeat(1, 3)  # (N, 3)

        # Rotation: identity quaternion (w=1, x=0, y=0, z=0)
        rotation = torch.zeros(len(xyz), 4).cuda()
        rotation[:, 0] = 1.0

        # Opacity: logit(INIT_OPACITY) so sigmoid gives ≈INIT_OPACITY initial opacity
        opacity = torch.logit(torch.full((len(xyz), 1), config.INIT_OPACITY)).cuda()

        self._xyz           = nn.Parameter(xyz)
        self._sh_band0      = nn.Parameter(sh_band0)
        self._sh_bands_rest = nn.Parameter(sh_bands_rest)
        self._scaling       = nn.Parameter(scaling)
        self._rotation      = nn.Parameter(rotation)
        self._opacity       = nn.Parameter(opacity)

        self.optimizer = torch.optim.Adam([
            {'params': [self._xyz],           'lr': config.LR_XYZ,           'name': 'xyz'},
            {'params': [self._sh_band0],      'lr': config.LR_SH_BAND0,      'name': 'sh_band0'},
            {'params': [self._sh_bands_rest], 'lr': config.LR_SH_BANDS_REST, 'name': 'sh_bands_rest'},
            {'params': [self._opacity],       'lr': config.LR_OPACITY,       'name': 'opacity'},
            {'params': [self._scaling],       'lr': config.LR_SCALING,       'name': 'scaling'},
            {'params': [self._rotation],      'lr': config.LR_ROTATION,      'name': 'rotation'},
        ], eps=config.ADAM_EPS)

    # ── Optimizer helpers ──────────────────────────────────────────────────────

    def _prune_optimizer(self, keep: torch.Tensor) -> None:
        """Retain only Gaussians where keep is True, slicing all optimizer states."""
        for group in self.optimizer.param_groups:
            param = group['params'][0]
            state = self.optimizer.state.get(param, {})
            new_param = nn.Parameter(param.data[keep])
            if 'exp_avg' in state:
                self.optimizer.state[new_param] = {
                    'step':       state['step'],
                    'exp_avg':    state['exp_avg'][keep],
                    'exp_avg_sq': state['exp_avg_sq'][keep],
                }
                del self.optimizer.state[param]
            group['params'][0] = new_param
        self._sync_params()

    def _append_to_optimizer(self, new_tensors: dict) -> None:
        """Append new Gaussians to every param group with zeroed Adam state."""
        for group in self.optimizer.param_groups:
            ext   = new_tensors[group['name']].detach()
            param = group['params'][0]
            state = self.optimizer.state.get(param, {})
            new_param = nn.Parameter(torch.cat([param.data, ext], dim=0))
            if 'exp_avg' in state:
                self.optimizer.state[new_param] = {
                    'step':       state['step'],
                    'exp_avg':    torch.cat([state['exp_avg'],    torch.zeros_like(ext)], dim=0),
                    'exp_avg_sq': torch.cat([state['exp_avg_sq'], torch.zeros_like(ext)], dim=0),
                }
                del self.optimizer.state[param]
            group['params'][0] = new_param
        self._sync_params()

    def _sync_params(self) -> None:
        """Re-bind _xyz, _sh_band0, etc. to the current optimizer parameter objects."""
        for group in self.optimizer.param_groups:
            setattr(self, f"_{group['name']}", group['params'][0])

    # ── Training controls ──────────────────────────────────────────────────────

    def oneupSHdegree(self) -> None:
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def densify_and_prune(self, viewspace_points, visibility_filter, radii) -> None:
        if viewspace_points.grad is None:
            return

        original_N      = len(self._xyz)
        grad_threshold  = config.DENSIFY_GRAD_THRESHOLD
        min_opacity     = config.PRUNE_MIN_OPACITY
        percent_dense   = config.DENSIFY_PERCENT_DENSE
        max_screen_size = config.DENSIFY_MAX_SCREEN_SIZE

        # Per-Gaussian 2D gradient magnitude (zero for non-visible Gaussians)
        grads = torch.zeros(original_N, device=self._xyz.device)
        grads[visibility_filter] = viewspace_points.grad[visibility_filter].norm(dim=-1)
        grads = torch.nan_to_num(grads)

        # Scene extent: longest side of the point-cloud bounding box
        with torch.no_grad():
            extent = (self._xyz.max(dim=0).values - self._xyz.min(dim=0).values).max().item()
        scale_thresh = percent_dense * extent

        max_scale = torch.exp(self._scaling).max(dim=-1).values.detach()
        high_grad  = grads >= grad_threshold

        # Compute all masks on the original N Gaussians before any append
        clone_mask = high_grad & (max_scale <= scale_thresh)
        split_mask = high_grad & (max_scale >  scale_thresh)
        n_split    = split_mask.sum().item()

        # ── Clone: duplicate in place ──────────────────────────────────────────
        if clone_mask.any():
            self._append_to_optimizer({
                'xyz':           self._xyz.data[clone_mask],
                'sh_band0':      self._sh_band0.data[clone_mask],
                'sh_bands_rest': self._sh_bands_rest.data[clone_mask],
                'scaling':       self._scaling.data[clone_mask],
                'rotation':      self._rotation.data[clone_mask],
                'opacity':       self._opacity.data[clone_mask],
            })

        # ── Split: replace with 2 smaller children ─────────────────────────────
        if split_mask.any():
            # Access originals via [:original_N] in case clone already extended tensors
            orig_xyz      = self._xyz.data[:original_N][split_mask]
            orig_scaling  = self._scaling.data[:original_N][split_mask]
            orig_rotation = self._rotation.data[:original_N][split_mask]

            stds    = torch.exp(orig_scaling).repeat(2, 1)
            samples = torch.normal(torch.zeros_like(stds), stds)

            q = F.normalize(orig_rotation, dim=-1).repeat(2, 1)
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            R_mat = torch.stack([
                1 - 2*(y*y + z*z),  2*(x*y - w*z),     2*(x*z + w*y),
                2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
                2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
            ], dim=-1).reshape(-1, 3, 3)
            new_xyz = (R_mat @ samples.unsqueeze(-1)).squeeze(-1) + orig_xyz.repeat(2, 1)

            self._append_to_optimizer({
                'xyz':           new_xyz,
                'sh_band0':      self._sh_band0.data[:original_N][split_mask].repeat(2, 1, 1),
                'sh_bands_rest': self._sh_bands_rest.data[:original_N][split_mask].repeat(2, 1, 1),
                'scaling':       torch.log(torch.exp(orig_scaling) / 1.6).repeat(2, 1),
                'rotation':      orig_rotation.repeat(2, 1),
                'opacity':       self._opacity.data[:original_N][split_mask].repeat(2, 1),
            })

        # ── Prune: split parents + low opacity + oversized screen radius ───────
        current_N = len(self._xyz)
        prune = torch.zeros(current_N, dtype=torch.bool, device=self._xyz.device)

        if n_split > 0:
            prune[:original_N] |= split_mask                            # remove split parents

        prune |= torch.sigmoid(self._opacity.data).squeeze(-1) < min_opacity

        padded_radii = torch.zeros(current_N, dtype=radii.dtype, device=radii.device)
        padded_radii[:original_N] = radii
        prune |= padded_radii > max_screen_size

        if prune.any():
            self._prune_optimizer(~prune)

        torch.cuda.empty_cache()

    def reset_opacity(self) -> None:
        # Clamp all opacities to ≤ 0.01 to restart the pruning cycle,
        # then zero Adam moments so stale momentum doesn't re-inflate them.
        new_val = torch.logit(torch.clamp(torch.sigmoid(self._opacity.data), max=config.OPACITY_RESET_TARGET))
        for group in self.optimizer.param_groups:
            if group['name'] == 'opacity':
                param = group['params'][0]
                param.data.copy_(new_val)
                state = self.optimizer.state.get(param, {})
                if 'exp_avg' in state:
                    state['exp_avg'].zero_()
                    state['exp_avg_sq'].zero_()
                break

    # ── Serialization ──────────────────────────────────────────────────────────

    def save_ply(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        xyz      = self._xyz.detach().cpu().numpy()                                       # (N, 3)
        normals  = np.zeros_like(xyz)
        sh_dc    = self._sh_band0.detach().transpose(1, 2).flatten(1).cpu().numpy()      # (N, 3)
        sh_rest  = self._sh_bands_rest.detach().transpose(1, 2).flatten(1).cpu().numpy() # (N, 45)
        opacity  = self._opacity.detach().cpu().numpy()                                   # (N, 1)
        scaling  = self._scaling.detach().cpu().numpy()                                   # (N, 3)
        rotation = self._rotation.detach().cpu().numpy()                                  # (N, 4)

        n_rest = sh_rest.shape[1]
        dtype = np.dtype(
            [('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
             ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
             ('f_dc_0', '<f4'), ('f_dc_1', '<f4'), ('f_dc_2', '<f4')] +
            [(f'f_rest_{i}', '<f4') for i in range(n_rest)] +
            [('opacity',  '<f4'),
             ('scale_0',  '<f4'), ('scale_1', '<f4'), ('scale_2', '<f4'),
             ('rot_0',    '<f4'), ('rot_1',   '<f4'), ('rot_2',   '<f4'), ('rot_3', '<f4')]
        )

        N   = len(xyz)
        arr = np.empty(N, dtype=dtype)
        arr['x'],  arr['y'],  arr['z']  = xyz[:, 0],     xyz[:, 1],     xyz[:, 2]
        arr['nx'], arr['ny'], arr['nz'] = normals[:, 0], normals[:, 1], normals[:, 2]
        arr['f_dc_0'], arr['f_dc_1'], arr['f_dc_2'] = sh_dc[:, 0], sh_dc[:, 1], sh_dc[:, 2]
        for i in range(n_rest):
            arr[f'f_rest_{i}'] = sh_rest[:, i]
        arr['opacity'] = opacity[:, 0]
        arr['scale_0'], arr['scale_1'], arr['scale_2'] = scaling[:, 0],  scaling[:, 1],  scaling[:, 2]
        arr['rot_0'],   arr['rot_1'],   arr['rot_2'],   arr['rot_3'] = (
            rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3])

        with open(path, 'wb') as f:
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {N}\n"
            )
            for name in dtype.names or []:
                header += f"property float {name}\n"
            header += "end_header\n"
            f.write(header.encode('ascii'))
            f.write(arr.tobytes())
