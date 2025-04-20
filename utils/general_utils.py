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

import torch
import sys
from datetime import datetime
import numpy as np
import random
import math

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling(s):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]
    return L

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float() + 0.5, torch.arange(H, device='cuda').float() + 0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T #@ c2w[:3,:3].T
    # rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d #+ rays_o
    return points

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    points[..., 1] = points[..., 1] * -1
    points[..., 2] = points[..., 2] * -1
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output, points

def apply_depth_map(
    depth_data,
    accumulation,
):
    mask = accumulation > 0.5
    depth_data[mask == 0] = 0
    depth_fg = depth_data[mask]  ## value in range [0, 1]

    if len(depth_fg) > 0:
        min_val, max_val = torch.min(depth_fg), torch.max(depth_fg)
        depth_normalized_foreground = 1 - (
                (depth_fg - min_val) / (max_val - min_val)
        )  ## for visualization, foreground is 1 (white), background is 0 (black)

        depth_data[mask] = depth_normalized_foreground

    depth_map = torch.cat((depth_data, depth_data, depth_data), axis=-1)
    return depth_map

def rotate_x(phi):
    cos = np.cos(phi)
    sin = np.sin(phi)
    return np.array([[1,   0,    0, 0],
                     [0, cos, -sin, 0],
                     [0, sin,  cos, 0],
                     [0,   0,    0, 1]], dtype=np.float32)

def rotate_z(psi):
    cos = np.cos(psi)
    sin = np.sin(psi)
    return np.array([[cos, -sin, 0, 0],
                     [sin,  cos, 0, 0],
                     [0,      0, 1, 0],
                     [0,      0, 0, 1]], dtype=np.float32)
def rotate_y(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos,   0,  sin, 0],
                     [0,     1,    0, 0],
                     [-sin,  0,  cos, 0],
                     [0,   0,      0, 1]], dtype=np.float32)

def get_camera_motion_bullet(c2w, n_bullet=15, axis='y'):
    if axis == 'y':
        rotate_fn = rotate_y
    elif axis == 'x':
        rotate_fn = rotate_x
    elif axis == 'z':
        rotate_fn = rotate_z
    else:
        raise NotImplementedError(f'rotate axis {axis} is not defined')

    y_angles = np.linspace(0, math.radians(360), n_bullet + 1)[:-1]
    c2ws = []
    for a in y_angles:
        c = rotate_fn(a) @ c2w
        c2ws.append(c)
    return np.array(c2ws)
