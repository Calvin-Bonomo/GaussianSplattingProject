"""Scene: loads a dataset and manages cameras + GaussianModel initialisation."""

import os
import random
import json
import numpy as np
import torch

from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.scene.dataset_readers import (
    readColmapSceneInfo, readNerfSyntheticInfo
)
from gaussian_splatting.utils.graphics_utils import fov2focal


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), \
                     round(orig_h / (resolution_scale * args.resolution))
    else:
        if args.resolution == -1:
            if orig_w > 1600:
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = cam_info.image.resize(resolution)

    resized_image = torch.from_numpy(
        np.array(resized_image_rgb)
    ).float() / 255.0

    if len(resized_image.shape) == 3:
        resized_image = resized_image.permute(2, 0, 1)
    else:
        resized_image = resized_image.unsqueeze(0)

    return Camera(
        uid=id,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FoVx,
        FoVy=cam_info.FoVy,
        image=resized_image,
        image_name=cam_info.image_name,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))
    return camera_list


class Scene:
    def __init__(self, args, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=None):
        if resolution_scales is None:
            resolution_scales = [1.0]

        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # ----------------------------------------------------------------
        # Load scene info
        # ----------------------------------------------------------------
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = readColmapSceneInfo(args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json, loading Blender/NeRF-synthetic dataset.")
            scene_info = readNerfSyntheticInfo(args.source_path,
                                               args.white_background,
                                               args.eval)
        else:
            raise ValueError(
                f"Could not recognise scene type at {args.source_path}. "
                "Expected a COLMAP sparse/ directory or transforms_train.json."
            )

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, \
                 open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                dest_file.write(src_file.read())

            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append({
                    "id": id, "img_name": cam.image_name,
                    "width": cam.width, "height": cam.height,
                    "position": (-cam.R.T @ cam.T).tolist(),
                    "rotation": cam.R.tolist(),
                    "fy": fov2focal(cam.FoVy, cam.height),
                    "fx": fov2focal(cam.FoVx, cam.width),
                })
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as f:
                json.dump(json_cams, f)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # ----------------------------------------------------------------
        # Build camera lists for each resolution scale
        # ----------------------------------------------------------------
        self.train_cameras = {}
        self.test_cameras  = {}
        for resolution_scale in resolution_scales:
            print(f"Loading Training Cameras at scale {resolution_scale}")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args)
            print(f"Loading Test Cameras at scale {resolution_scale}")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args)

        # ----------------------------------------------------------------
        # Initialise or load Gaussians
        # ----------------------------------------------------------------
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                  "point_cloud",
                                                  f"iteration_{self.loaded_iter}",
                                                  "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path,
                                        "point_cloud",
                                        f"iteration_{iteration}")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
