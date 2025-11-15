import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from plyfile import PlyData
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

import random
import numpy as np
random_seed = 0  # 你可以选择任何整数作为随机种子
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# 新增：PLY文件加载函数
def load_ply_as_initialization(ply_path, dataset, opt):
    """从PLY文件加载高斯模型参数"""
    gaussians = GaussianModel(dataset.sh_degree)

    gaussians.load_ply(ply_path)

    # # 读取PLY文件数据
    # plydata = PlyData.read(ply_path)
    # data = plydata.elements[0].data
    #
    # # 解析PLY属性（需与官方保存格式完全一致）
    # xyz = np.stack((data['x'], data['y'], data['z'])).transpose()
    # opacities = data['opacity'].reshape(-1, 1)
    #
    # # 解析球谐系数
    # features_dc = np.stack([data[f'f_dc_{i}'] for i in range(3)]).transpose().reshape(-1, 3, 1)
    # features_rest = np.stack([data[f'f_rest_{i}'] for i in range(45)]).transpose().reshape(-1, 3, 15)
    #
    # # 解析缩放和旋转参数
    # scales = np.stack([data[f'scale_{i}'] for i in range(3)]).transpose()
    # rots = np.stack([data[f'rot_{i}'] for i in range(4)]).transpose()
    #
    # # 转换为Tensor并设置可训练参数
    # gaussians._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    # gaussians._features_dc = nn.Parameter(
    #     torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    # gaussians._features_rest = nn.Parameter(
    #     torch.tensor(features_rest, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    # gaussians._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    # gaussians._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    # gaussians._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
    #
    # # 在load_ply_as_initialization函数末尾添加
    # gaussians.xyz_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
    # gaussians.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
    # gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")

    # 初始化优化器（关键！需与原始训练设置一致）
    gaussians.training_setup(opt)
    return gaussians


def training(dataset, opt, pipe, checkpoint_path=None, ply_path=None):
    """修改后的训练函数，支持从PLY文件加载"""

    first_iter = 0

    if "blender" in args.source_path:
        dataset.resolution = 1
    elif "LLFF" in args.source_path:
        dataset.resolution = 1

    # 初始化高斯模型
    if ply_path:  # 从PLY文件加载
        print(f"Loading initial Gaussians from {ply_path}")
        gaussians = load_ply_as_initialization(ply_path, dataset, opt)
        start_iter = 0  # 或从元数据读取历史迭代次数
    else:  # 正常初始化
        gaussians = GaussianModel(dataset.sh_degree)
        start_iter = 0

    scene = Scene(dataset, gaussians, load_iteration=start_iter)
    gaussians.spatial_lr_scale = scene.cameras_extent

    # 设置优化器学习率（关键！需与原始训练参数一致）
    lrs = {
        'position': opt.position_lr_init * gaussians.spatial_lr_scale,
        'feature': opt.feature_lr,
        'opacity': opt.opacity_lr,
        'scaling': opt.scaling_lr,
        'rotation': opt.rotation_lr
    }
    gaussians.optimizer = torch.optim.Adam([
        {'params': [gaussians._xyz], 'lr': lrs['position'], "name": "xyz"},
        {'params': [gaussians._features_dc], 'lr': lrs['feature'], "name": "f_dc"},
        {'params': [gaussians._features_rest], 'lr': lrs['feature'] / 20.0, "name": "f_rest"},
        {'params': [gaussians._opacity], 'lr': lrs['opacity'], "name": "opacity"},
        {'params': [gaussians._scaling], 'lr': lrs['scaling'], "name": "scaling"},
        {'params': [gaussians._rotation], 'lr': lrs['rotation'], "name": "rotation"}
    ])


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # 原有训练循环保持不变...
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))



        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        if iteration % 2000 == 0:
            plt.imshow(image.detach().cpu().numpy().transpose((1, 2, 0)))
            plt.show()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    print("1")
    gaussians.save_ply(save_path + str(iteration) + "/point_cloud.ply")



if __name__ == "__main__":
    # 修改参数解析器，添加PLY路径参数
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    save_path = "output/ship_reconstruct/point_cloud/iteration_"
    ifAttack = 'output/ship_attack/merged_gaussians'
    args = parser.parse_args([
        # '--source_path', 'data/blender/rendered_ship',
        '--source_path', 'data/blender/rendered_ship',
        '--model_path', ifAttack,
        '--sh_degree', '3'
    ])

    lp._model_path, lp.model_path = args.model_path, args.model_path


    resume_ply = ifAttack + "/point_cloud/iteration_2000/point_cloud.ply"

    # 启动训练时指定PLY路径
    training(lp.extract(args), op.extract(args), pp.extract(args), ply_path=resume_ply)



