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
os.environ["CC"] = "/ubc/cs/home/c/chunjins/chunjin_scratch/software/cuda/gcc/gcc_exe/bin/gcc-gcc-9.5.0"
os.environ["CXX"] = "/ubc/cs/home/c/chunjins/chunjin_scratch/software/cuda/gcc/gcc_exe/bin/g++-gcc-9.5.0"

import torch
import cv2
from scene import Scene
import time
import pickle
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, depth_to_normal, apply_depth_map, get_camera_motion_bullet, apply_depth_colormap
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from utils.image_utils import psnr
from utils.loss_utils import ssim
import lpips
import sys
import copy
import open3d as o3d
from utils.mesh_utils import GaussianExtractor, post_process_mesh
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))



def mesh_set(model_path, name, iteration, views, gaussians, pipeline, background, smpl_rot):
    mesh_path = os.path.join(model_path, name, "ours_{}".format(iteration), "meshes")
    makedirs(mesh_path, exist_ok=True)
    gaussExtractor = GaussianExtractor(gaussians, render, pipeline, background)
    gaussExtractor.gaussians.active_sh_degree = 0

    for id, view in enumerate(tqdm(views, desc="Rendering progress")):
        E = np.eye(4)
        E[:3, :3] = view.R
        E[:3, 3] = view.T + 0.5
        c2w = np.linalg.inv(E)
        c2ws_y = get_camera_motion_bullet(c2w, axis='y')
        c2ws_z = get_camera_motion_bullet(c2w, axis='z')
        c2ws_x = get_camera_motion_bullet(c2w, axis='x')
        c2ws = np.concatenate([c2ws_x, c2ws_y, c2ws_z], axis=0)
        mesh_views = []
        smpl_rot_name = name.replace('mesh_','').replace('ing','')
        transforms, translation = smpl_rot[smpl_rot_name][view.pose_id]['transforms'], smpl_rot[smpl_rot_name][view.pose_id]['translation']

        for i in range(1, c2ws.shape[0]):
            view_i = copy.deepcopy(view)
            Ei = np.linalg.inv(c2ws[i])
            view_i.setE(Ei)
            mesh_views.append(view_i)

            # render_output = render(view_i, gaussians, pipeline, background, transforms=transforms, translation=translation)
            # rendering = render_output["render"]
            # depth = render_output["render_depth"]
            # alpha = render_output["render_alpha"]

            # rendering.permute(1, 2, 0)[alpha[0] <= 0.] = 0 if background.sum().item() == 0 else 1

            # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
            # makedirs(render_path, exist_ok=True)
            # if "zju_mocap" in model_path:
            #     fn = f"camera_{int(views[id].cam_id) + 1:02d}_frame_{int(views[id].frame_id):06d}_{i:02d}.png"
            # else:
            #     fn = views[id].image_name + f'_{i:02d}.png'

            # img_save = torch.cat([rendering], dim=-1)
            # torchvision.utils.save_image(img_save, os.path.join(render_path, fn))

        gaussExtractor.reconstruction(render, mesh_views, gaussians, pipeline, background, transforms, translation)
        # mesh = gaussExtractor.extract_mesh_bounded()
        mesh = gaussExtractor.extract_mesh_unbounded(resolution=512)
        T = torch.eye(4)
        T[:3,:3] = mesh_views[0].smpl_param['R']
        T[:3,3] = mesh_views[0].smpl_param['Th']
        T = torch.linalg.inv(T).cpu().numpy()
        mesh.transform(T)

        mesh_post = post_process_mesh(mesh, cluster_to_keep=1)

        name_save = f"{view.image_name}.ply"
        o3d.io.write_triangle_mesh(os.path.join(mesh_path, name_save), mesh_post)
        print("mesh saved at {}".format(os.path.join(mesh_path, name_save)))
        # post-process the mesh and save, saving the largest N clusters
        # mesh_post = post_process_mesh(mesh, cluster_to_keep=1)
        # o3d.io.write_triangle_mesh(os.path.join(mesh_path, name_save.replace('.ply', '_post.ply')), mesh_post)
        # print("mesh post processed saved at {}".format(os.path.join(mesh_path, name_save.replace('.ply', '_post.ply'))))


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, smpl_rot):
    render_path = os.path.join(model_path, name+'_img_geo', "ours_{}".format(iteration), "renders")
    gs_mesh_path = os.path.join(model_path, name+'_gs_mesh', "ours_{}".format(iteration))
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gs_mesh_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)

    # # Load data (deserialize)
    # with open(model_path + '/smpl_rot/' + f'iteration_{iteration}/' + 'smpl_rot.pickle', 'rb') as handle:
    #     smpl_rot = pickle.load(handle)

    rgbs = []
    rgbs_gt = []
    elapsed_time = 0
    geometry_dict = {}
    for id, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :].cuda()
        bound_mask = view.bound_mask
        transforms, translation = smpl_rot[name][view.pose_id]['transforms'], smpl_rot[name][view.pose_id]['translation']

        # Start timer
        start_time = time.time() 
        render_output = render(view, gaussians, pipeline, background, transforms=transforms, translation=translation, return_gs_mesh=True)
        rendering = render_output["render"]
        depth = render_output["render_depth"]
        alpha = render_output["render_alpha"]
        gs_mesh = render_output["gs_mesh"]

        # end time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time += end_time - start_time

        rendering.permute(1,2,0)[alpha[0]<=0.] = 0 if background.sum().item() == 0 else 1

        # rendering.permute(1, 2, 0)[bound_mask[0] == 0] = 0 if background.sum().item() == 0 else 1

        depth_normal, _ = depth_to_normal(view, depth)

        geometry_map = torch.cat([alpha.permute(1, 2, 0), depth.permute(1, 2, 0), depth_normal], dim=-1)
        geometry_dict[view.image_name] = geometry_map.detach().cpu().numpy()

        depth_normal_image = (depth_normal + 1.) / 2.
        depth_normal_image = depth_normal_image * alpha.permute(1, 2, 0) + (1 - alpha.permute(1, 2, 0))
        depth_normal_image = depth_normal_image.permute(2, 0, 1)

        depth_map_image = apply_depth_colormap(depth.permute(1, 2, 0), alpha.permute(1, 2, 0), near_plane=None, far_plane=None)
        # depth_map = apply_depth_map(depth[0, :, :, None], alpha[0, :, :, None])
        depth_map_image = depth_map_image.permute(2, 0, 1)

        rgbs.append(rendering)
        rgbs_gt.append(gt)
        # depths.append(depth_map)
        # normals.append(depth_normal)

        if "zju_mocap" in model_path:
            fn = f"camera_{int(views[id].cam_id)+1:02d}_frame_{int(views[id].frame_id):06d}.png"
        else:
            fn = views[id].image_name+'.png'

        img_save = torch.cat([rendering, depth_map_image, depth_normal_image], dim=-1)
        torchvision.utils.save_image(img_save, os.path.join(render_path, fn))

        if gs_mesh is not None:
            gs_mesh.export(os.path.join(gs_mesh_path, fn.replace('png', 'ply')))

    np.savez(os.path.join(model_path, f'geometry_{name}.npz'), **geometry_dict)
    # Calculate elapsed time
    print("Elapsed time: ", elapsed_time, " FPS: ", len(views)/elapsed_time) 

    psnrs = 0.0
    ssims = 0.0
    lpipss = 0.0

    for id in range(len(views)):
        rendering = rgbs[id]
        gt = rgbs_gt[id]
        rendering = torch.clamp(rendering, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)

        # if "zju_mocap" in model_path:
        #     fn = f"camera_{int(views[id].cam_id)+1:02d}_frame_{int(views[id].frame_id):06d}.png"
        # else:
        #     fn = views[id].image_name+'.png'

        # torchvision.utils.save_image(rendering, os.path.join(render_path, fn))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, fn))

        # metrics
        # rendering_img = torch.tensor(cv2.imread(os.path.join(render_path, fn)) / 255.).cuda()
        # gt_img = torch.tensor(cv2.imread(os.path.join(gts_path, fn)) / 255.).cuda()

        rendering_img = rendering
        gt_img = gt

        # rendering_img = rendering_img.permute(2, 1, 0).float()
        # gt_img = gt_img.permute(2, 1, 0).float()

        psnrs += psnr(rendering_img, gt_img)
        ssims += ssim(rendering_img, gt_img)
        lpipss += loss_fn_vgg(rendering_img, gt_img, normalize=True).mean()

    psnrs /= len(views)
    ssims /= len(views)
    lpipss /= len(views)

    # evalution metrics
    print("\n[ITER {}] Evaluating {} #{}: PSNR {} SSIM {} LPIPS {}".format(iteration, name, len(views), psnrs, ssims, lpipss))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, split: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        validation_configs = (
            {'name': 'novel_pose', 'cameras' : scene.getNovelPoseCameras()},
            {'name': 'novel_view', 'cameras' : scene.getNovelViewCameras()},
            {'name': 'train', 'cameras': scene.getTrainCameras()},
        )

        smpl_rot = {}
        smpl_rot['train'], smpl_rot['novel_pose'], smpl_rot['novel_view'] = {}, {}, {}
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                for idx, viewpoint in enumerate(config['cameras']):
                    smpl_rot[config['name']][viewpoint.pose_id] = {}
                    render_output = render(viewpoint, scene.gaussians, pipeline, background, return_smpl_rot=True)
                    smpl_rot[config['name']][viewpoint.pose_id]['transforms'] = render_output['transforms']
                    smpl_rot[config['name']][viewpoint.pose_id]['translation'] = render_output['translation']

        # if not skip_train:
        #      render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        # if not skip_test:
        #      render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

        # if not skip_novel_pose:
        #      render_set(dataset.model_path, "novel_pose", scene.loaded_iter, scene.getNovelPoseCameras(), gaussians, pipeline, background, smpl_rot)
        #
        # if not skip_novel_view:
        #      render_set(dataset.model_path, "novel_view", scene.loaded_iter, scene.getNovelViewCameras(), gaussians, pipeline, background, smpl_rot)

        if split == 'train':
            render_set(dataset.model_path, split, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, smpl_rot)
        elif split == 'novel_view':
            render_set(dataset.model_path, split, scene.loaded_iter, scene.getNovelViewCameras(), gaussians, pipeline, background, smpl_rot)
        elif split == 'novel_pose':
            render_set(dataset.model_path, split, scene.loaded_iter, scene.getNovelPoseCameras(), gaussians, pipeline, background, smpl_rot)
        elif split == 'mesh_training':
            mesh_set(dataset.model_path, "mesh_training", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, smpl_rot)
            # render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, smpl_rot)
        elif split == 'mesh_novel_pose':
            mesh_set(dataset.model_path, "mesh_novel_pose", scene.loaded_iter, scene.getNovelPoseCameras(), gaussians, pipeline, background, smpl_rot)
        elif split == 'mesh_novel_view':
            mesh_set(dataset.model_path, "mesh_novel_view", scene.loaded_iter, scene.getNovelViewCameras(), gaussians, pipeline, background, smpl_rot)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_novel_pose", action="store_true")
    parser.add_argument("--skip_novel_view", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.split)