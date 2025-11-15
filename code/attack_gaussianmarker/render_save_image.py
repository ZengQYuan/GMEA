
import matplotlib.pyplot as plt
import os
import sys
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch.nn as nn
from plyfile import PlyData


import random
import numpy as np
random_seed = 0  # 你可以选择任何整数作为随机种子
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_ply_as_initialization(ply_path, dataset, opt):
    """从PLY文件加载高斯模型参数"""
    gaussians = GaussianModel(dataset.sh_degree)

    # 读取PLY文件数据
    plydata = PlyData.read(ply_path)
    data = plydata.elements[0].data

    # 解析PLY属性（需与官方保存格式完全一致）
    xyz = np.stack((data['x'], data['y'], data['z'])).transpose()
    opacities = data['opacity'].reshape(-1, 1)

    # 解析球谐系数
    features_dc = np.stack([data[f'f_dc_{i}'] for i in range(3)]).transpose().reshape(-1, 3, 1)
    features_rest = np.stack([data[f'f_rest_{i}'] for i in range(45)]).transpose().reshape(-1, 3, 15)

    # 解析缩放和旋转参数
    scales = np.stack([data[f'scale_{i}'] for i in range(3)]).transpose()
    rots = np.stack([data[f'rot_{i}'] for i in range(4)]).transpose()

    # 转换为Tensor并设置可训练参数
    gaussians._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    gaussians._features_dc = nn.Parameter(
        torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    gaussians._features_rest = nn.Parameter(
        torch.tensor(features_rest, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    gaussians._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    gaussians._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    gaussians._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    # 在load_ply_as_initialization函数末尾添加
    gaussians.xyz_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
    gaussians.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")

    # 初始化优化器（关键！需与原始训练设置一致）
    gaussians.training_setup(opt)
    return gaussians

def save_rendered_views(scene, gaussians, pipe, dataset, output_dir="output/rendered_views"):
    """保存所有训练视角的渲染图"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置背景颜色（需与训练时一致）
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 获取所有训练相机（确保Scene初始化时shuffle=False）
    train_cams = scene.getTrainCameras()

    # 禁用梯度计算以节省显存
    with torch.no_grad():
        for idx, viewpoint_cam in tqdm(enumerate(train_cams), desc="渲染进度"):
            # 执行渲染
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            rendered_image = render_pkg["render"]  # Tensor形状为(3, H, W)

            # 转换为0-255的numpy数组
            img_np = rendered_image.clamp(0.0, 1.0).cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0)) * 255  # 转换为HWC格式
            img_np = img_np.astype(np.uint8)

            # 保存为PNG文件
            if "blender" in wm_source_path:
                Image.fromarray(img_np).save(os.path.join(output_dir, f"r_{idx}.png"))
            elif "LLFF" in wm_source_path:
                Image.fromarray(img_np).save(os.path.join(output_dir, f"image{idx:03d}.png"))

    print(f"渲染完成！结果已保存至：{os.path.abspath(output_dir)}")


if __name__ == "__main__":
    # 参数解析器配置
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    ifAttack = 'output/room'  # 'output/chair_wm'
    wm_source_path = "/sda1/qyzeng/GaussianMarker/data/LLFF/room"
    args = parser.parse_args([
        '--model_path', ifAttack,
        '--sh_degree', '3'
    ])

    lp._model_path, lp.model_path = args.model_path, args.model_path
    resume_ply = ifAttack + "/point_cloud/iteration_2000/point_cloud.ply"

    # 添加渲染保存参数
    # parser.add_argument("--render_output", type=str, default="data/blender/rendered_ship/train")
    parser.add_argument("--render_output", type=str, default="data/LLFF/rendered_room/images")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    args = parser.parse_args()

    # 初始化系统状态
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset = lp.extract(args)
    opt = op.extract(args)
    gaussians = load_ply_as_initialization(resume_ply, dataset, opt)
    dataset.model_path = ifAttack
    dataset.source_path = wm_source_path
    dataset.resolution = 1
    scene = Scene(dataset, gaussians, load_iteration=0, shuffle=False)

    # 执行渲染保存
    save_rendered_views(
        scene,
        gaussians,
        pp.extract(args),
        dataset,
        args.render_output
    )
