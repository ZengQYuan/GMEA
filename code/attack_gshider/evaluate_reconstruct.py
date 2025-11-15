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
from decoder import SimpleCNN  # [新增] 导入RGB解码器

# 导入用于计算指标的库
import torchmetrics
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# --- 1. 辅助函数：复用PLY加载逻辑 (根据您的代码做了微调以兼容两种特征) ---
def load_ply_as_initialization(ply_path, dataset, opt):
    gaussians = GaussianModel(dataset.sh_degree)
    try:
        plydata = PlyData.read(ply_path)
    except Exception as e:
        print(f"[错误] 无法读取 PLY 文件: {ply_path}。错误: {e}")
        sys.exit(1)

    data = plydata.elements[0].data

    xyz = np.stack((data['x'], data['y'], data['z'])).transpose()
    opacities = data['opacity'].reshape(-1, 1)
    scales = np.stack([data[f'scale_{i}'] for i in range(3)]).transpose()
    rots = np.stack([data[f'rot_{i}'] for i in range(4)]).transpose()

    # 兼容您的 semantic_feature (GS-Hider) 和 features_dc (GaussianMarker)
    if 'semantic_0' in data.dtype.names:
        print("[信息] 在 PLY 文件中检测到 'semantic_' 特征。")
        semantic_feature_names = [name for name in data.dtype.names if name.startswith('semantic_')]
        semantic_dim = len(semantic_feature_names)
        semantic_features = np.stack([data[name] for name in semantic_feature_names]).transpose()
        gaussians._semantic_feature = nn.Parameter(
            torch.tensor(semantic_features, dtype=torch.float, device="cuda").reshape(-1, 1,
                                                                                      semantic_dim).requires_grad_(
                True))
    elif 'f_dc_0' in data.dtype.names:
        print("[信息] 在 PLY 文件中检测到 'f_dc' 特征。")
        features_dc = np.stack([data[f'f_dc_{i}'] for i in range(3)]).transpose()
        features_rest = np.stack([data[f'f_rest_{i}'] for i in range(48)]).transpose()
        gaussians._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        gaussians._features_rest = nn.Parameter(
            torch.tensor(features_rest, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
    else:
        raise ValueError(f"错误: PLY 文件 '{ply_path}' 中找不到可识别的特征属性 ('semantic_' 或 'f_dc')。")

    gaussians._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    gaussians._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    gaussians._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    gaussians._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
    gaussians.training_setup(opt)
    return gaussians


# --- 2. 主评估函数 ---
def calculate_rgb_metrics(dataset, opt, pipe, ply_path, checkpoint_path):
    """
    加载重建的3DGS模型和RGB解码器，与原始模型的渲染图进行比较，并计算视觉质量指标。
    """
    print("\n[信息] 正在初始化场景和模型用于指标计算...")

    # --- 场景加载 (加载 xx_wm 的场景以获取 GT 图像和相机) ---
    scene = Scene(dataset, GaussianModel(dataset.sh_degree), shuffle=False)

    # --- 模型加载 ---
    # 1. 加载你二次训练好的 xx_reconstruct 模型
    print(f"\n[信息] 正在从 PLY 文件加载重建的 3DGS 模型: {ply_path}")
    reconstructed_gaussians = load_ply_as_initialization(ply_path, dataset, opt)

    # 2. [新增] 加载预训练的 RGB 解码器 (imagenet)
    imagenet = SimpleCNN().cuda()
    net_checkpoint_path = checkpoint_path.replace('.pth', '_net.pth')
    if os.path.exists(net_checkpoint_path):
        imagenet.load_state_dict(torch.load(net_checkpoint_path, map_location="cuda"))
        print(f"[信息] 成功从以下路径加载 RGB 解码器: {net_checkpoint_path}")
    else:
        print(f"[错误] RGB 解码器权重文件未找到: {net_checkpoint_path}")
        return
    imagenet.eval()  # 设为评估模式

    # --- 评估准备 ---
    test_cameras = scene.getTrainCameras()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    total_ssim, total_psnr, total_mse = 0.0, 0.0, 0.0
    num_cameras = len(test_cameras)

    print(f"\n[信息] 开始在 {num_cameras} 个相机视角上进行评估...")

    # --- 循环评估 ---
    with torch.no_grad():
        for i, viewpoint_cam in enumerate(test_cameras):
            # A. 渲染 "重建模型" 的特征图
            render_pkg = render(viewpoint_cam, reconstructed_gaussians, pipe, background)
            feature_map_reconstruct = render_pkg["render"]

            # B. [新增] 使用解码器将特征图解码为 "重建RGB图"
            rendered_rgb_image = imagenet(feature_map_reconstruct.unsqueeze(0)).squeeze(0)
            rendered_rgb_image = torch.clamp(rendered_rgb_image, 0.0, 1.0)

            # C. 获取 "基准真值图" (即 xx_wm 的最终渲染图)
            gt_image = torch.clamp(viewpoint_cam.original_image.to("cuda"), 0.0, 1.0)

            # 确保尺寸一致
            if rendered_rgb_image.shape != gt_image.shape:
                gt_image = torch.nn.functional.interpolate(gt_image.unsqueeze(0),
                                                           size=rendered_rgb_image.shape[1:]).squeeze(0)

            # 准备数据进行指标计算
            rendered_batch = rendered_rgb_image.unsqueeze(0)
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

            # 显示第一张图的对比
            if i % 50 == 0:
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(gt_image.cpu().numpy().transpose(1, 2, 0))
                axes[0].set_title("Ground Truth RGB (from xx_wm)")
                axes[0].axis('off')
                axes[1].imshow(rendered_rgb_image.cpu().numpy().transpose(1, 2, 0))
                axes[1].set_title("Reconstructed RGB (from xx_reconstruct)")
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
    parser = ArgumentParser(description="评估二次训练后3DGS模型的RGB重建质量")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # [修改] 添加了 --checkpoint_path 用于加载解码器
    parser.add_argument("--ply_path", type=str, default="output/trex_attack/merged_gaussians/point_cloud/iteration_2000/point_cloud.ply",
                        help="指向 xx_reconstruct 模型的 .ply 文件路径")
    parser.add_argument("--checkpoint_path", type=str, default="output/trex_wm/chkpnt10000.pth",
                        help="指向原始 xx_wm 模型的 .pth 检查点文件 (用于加载解码器权重)")
    parser.add_argument("--quiet", action="store_true")

    # 命令行运行示例:
    # python evaluate_reconstruction_v2.py --source_path data/LLFF/room --model_path output/room_wm --ply_path output/room_reconstruct/point_cloud/iteration_20000/point_cloud.ply --checkpoint_path output/room_wm/chkpnt20000.pth
    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet)

    calculate_rgb_metrics(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        ply_path=args.ply_path,
        checkpoint_path=args.checkpoint_path
    )

    print("\n脚本执行完毕。✅")