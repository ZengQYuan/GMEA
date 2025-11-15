import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from plyfile import PlyData
import torch.nn as nn

# 从您的项目中导入必要的模块
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, OptimizationParams, PipelineParams
from utils.general_utils import safe_state

# 导入用于计算指标的库 (torchmetrics 非常方便)
# 如果没有安装，请运行: pip install torchmetrics
import torchmetrics
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# --- 1. 辅助函数：从您的代码中复用PLY加载逻辑 ---
# 这个函数专门用于从 .ply 文件加载带有 'semantic_' 特征的3DGS模型
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

# --- 2. 主评估函数 ---
def calculate_rgb_metrics(dataset, opt, pipe, ply_path):
    """
    加载重建的3DGS模型，与原始模型的渲染图进行比较，并计算视觉质量指标。
    """
    print("\n[信息] 正在初始化场景和模型用于指标计算...")

    # --- 场景加载 ---
    # 我们加载原始xx_wm的场景，这样就能直接访问到它的渲染图(gt_image)
    scene = Scene(dataset, GaussianModel(dataset.sh_degree), shuffle=False)

    # --- 模型加载 ---
    # 加载你二次训练好的 xx_reconstruct 模型
    print(f"\n[信息] 正在从 PLY 文件加载重建的 3DGS 模型: {ply_path}")
    reconstructed_gaussians = load_ply_as_initialization(ply_path, dataset, opt)

    # --- 评估准备 ---
    # 获取测试集相机视角
    test_cameras = scene.getTrainCameras()  # haha

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 初始化指标记录变量
    total_ssim, total_psnr, total_mse = 0.0, 0.0, 0.0
    num_cameras = len(test_cameras)

    print(f"\n[信息] 开始在 {num_cameras} 个相机视角上进行评估...")

    # --- 循环评估 ---
    with torch.no_grad():
        for i, viewpoint_cam in enumerate(test_cameras):
            # 渲染 "重建图"
            render_pkg = render(viewpoint_cam, reconstructed_gaussians, pipe, background)
            rendered_image = torch.clamp(render_pkg["render"], 0.0, 1.0)

            # 获取 "基准真值图" (即 xx_wm 的渲染图)
            gt_image = torch.clamp(viewpoint_cam.original_image.to("cuda"), 0.0, 1.0)

            # 确保图像尺寸一致 (有时可能因数据处理有微小差异)
            if rendered_image.shape != gt_image.shape:
                gt_image = torch.nn.functional.interpolate(gt_image.unsqueeze(0),
                                                           size=rendered_image.shape[1:]).squeeze(0)

            # 为 torchmetrics 准备数据 (需要 batch 维度)
            rendered_batch = rendered_image.unsqueeze(0)
            gt_batch = gt_image.unsqueeze(0)

            # 计算指标
            ssim_val = torchmetrics.functional.structural_similarity_index_measure(rendered_batch, gt_batch,
                                                                                   data_range=1.0)
            psnr_val = torchmetrics.functional.peak_signal_noise_ratio(rendered_batch, gt_batch, data_range=1.0)
            mse_val = torch.nn.functional.mse_loss(rendered_batch, gt_batch)

            total_ssim += ssim_val.item()
            total_psnr += psnr_val.item()
            total_mse += mse_val.item()

            print(f"  评估视角 {i + 1}/{num_cameras}: PSNR={psnr_val.item():.2f}, SSIM={ssim_val.item():.4f}")

            # 可选：显示第一张图的对比
            if i % 50 == 0:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(gt_image.cpu().numpy().transpose(1, 2, 0))
                axes[0].set_title("Ground Truth Image (from xx_wm)")
                axes[0].axis('off')
                axes[1].imshow(rendered_image.cpu().numpy().transpose(1, 2, 0))
                axes[1].set_title("Reconstructed Image (from xx_reconstruct)")
                axes[1].axis('off')
                plt.show()

    # --- 计算并打印平均结果 ---
    avg_ssim = total_ssim / num_cameras
    avg_psnr = total_psnr / num_cameras
    avg_mse = total_mse / num_cameras

    print("\n--- 📊 最终平均视觉质量指标 ---")
    print(f"  - 平均 SSIM: {avg_ssim:.4f}  (越高越好, 1.0为完美)")
    print(f"  - 平均 PSNR: {avg_psnr:.2f} dB (越高越好)")
    print(f"  - 平均 MSE:  {avg_mse:.6f} (越低越好)")
    print("------------------------------------")


if __name__ == "__main__":
    # --- 参数解析 ---
    parser = ArgumentParser(description="评估二次训练后3DGS模型的RGB重建质量")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # 添加我们这个脚本需要的核心参数
    parser.add_argument("--ply_path", type=str, default="output/trex_attack/merged_gaussians/point_cloud/iteration_2000/point_cloud.ply",
                        help="指向 xx_reconstruct 模型的 .ply 文件路径")
    parser.add_argument("--quiet", action="store_true")

    # 解析命令行参数
    # 你可以像下面这样从命令行运行，或者直接在IDE中配置
    # python evaluate_reconstruction.py --source_path data/LLFF/room --model_path output/room_wm --ply_path output/room_reconstruct/point_cloud/iteration_20000/point_cloud.ply
    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet)

    # --- 执行评估函数 ---
    calculate_rgb_metrics(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        ply_path=args.ply_path
    )

    print("\n脚本执行完毕。✅")
