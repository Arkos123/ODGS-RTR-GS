#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json

import torch
from arguments import ModelParams
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.cameras import Camera


class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel = None, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0], read_cam_only=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = None if read_cam_only else args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        source_path = args.source_path if read_cam_only else args.source_path

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        print(os.path.join(source_path, "transforms_train.json"))

        eval = True if read_cam_only else args.eval
        debug = False if read_cam_only else args.debug_cuda
        white_background = True if read_cam_only else args.white_background
        

        if os.path.exists(os.path.join(source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, None if read_cam_only else args.images, eval,
                                                          debug=debug, read_cam_only=read_cam_only)
        elif os.path.exists(os.path.join(source_path, "transforms_train.json")):
            
            if "stanford_orb" in source_path:
                print("Found keyword stanford_orb, assuming Stanford ORB data set!")
                scene_info = sceneLoadTypeCallbacks["StanfordORB"](source_path, white_background, eval, 
                                                                   debug=debug, read_cam_only=read_cam_only)
            elif "Synthetic4Relight" in source_path:
                print("Found transforms_train.json file, assuming Synthetic4Relight data set!")
                scene_info = sceneLoadTypeCallbacks["Synthetic4Relight"](source_path, white_background, eval,
                                                            debug=debug, read_cam_only=read_cam_only)
            else:
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](source_path, white_background, eval, 
                                                               debug=debug, read_cam_only=read_cam_only)
        elif os.path.exists(os.path.join(source_path, "inputs/sfm_scene.json")):
            print("Found sfm_scene.json file, assuming NeILF data set!")
            scene_info = sceneLoadTypeCallbacks["NeILF"](source_path, white_background, eval,
                                                         debug=debug, read_cam_only=read_cam_only)
        elif os.path.exists(os.path.join(source_path, "data_extrinsics.json")):
            print("Found data_extrinsics.json file, assuming OpenMVG equirectangular data set!")
            scene_info = sceneLoadTypeCallbacks["OpenMVG"](source_path, white_background, eval,
                                                           debug=debug, read_cam_only=read_cam_only)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter and not read_cam_only:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args, read_cam_only=read_cam_only)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args, read_cam_only=read_cam_only)

        self.scene_info = scene_info

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    

    def get_canonical_rays(self, scale: float = 1.0):
        """
        Get canonical rays for the scene.
        相机自身坐标系中，从原点出发、穿过每个像素的归一化射线方向。
        """
        # NOTE: some datasets do not share the same intrinsic (e.g. DTU)
        # get reference camera
        ref_camera: Camera = self.train_cameras[scale][0]
        return ref_camera.get_canonical_rays()

