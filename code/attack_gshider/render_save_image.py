import numpy as np
import os
import torch
import torch.nn as nn
import sys
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image

# Ensure the project's root is in the python path to find custom modules
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, OptimizationParams
from decoder import SimpleCNN

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def render_and_save_images(dataset, opt, pipe, checkpoint, output_dir):
    """
    Loads models from a checkpoint, renders all training viewpoints,
    and saves the resulting RGB images to the specified directory.
    """
    # --- 1. Model and Scene Setup ---
    print("\n[INFO] Setting up models and scene...")
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    imagenet = SimpleCNN().cuda()

    # --- 2. Load Pre-trained Weights from Checkpoint ---
    if not checkpoint or not os.path.exists(checkpoint):
        print(f"[ERROR] Checkpoint not found at: {checkpoint}")
        return

    print(f"\n[INFO] Loading models from checkpoint: {checkpoint}")
    (model_params, _) = torch.load(checkpoint, map_location="cuda", weights_only=False)
    gaussians.restore(model_params, opt)

    net_checkpoint_path = checkpoint.replace('.pth', '_net.pth')
    if os.path.exists(net_checkpoint_path):
        print(f"[INFO] Loading Decoder from: {net_checkpoint_path}")
        imagenet.load_state_dict(torch.load(net_checkpoint_path, map_location="cuda", weights_only=False))
    else:
        print(f"[ERROR] Decoder checkpoint not found at: {net_checkpoint_path}")
        return

    # **IMPORTANT**: Set the network to evaluation mode
    imagenet.eval()

    # --- 3. Prepare for Rendering ---
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Rendered images will be saved to: {os.path.abspath(output_dir)}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Get all training cameras
    viewpoints = scene.getTrainCameras()

    # --- 4. Execute Rendering, Decoding, and Saving in a Loop ---
    # We use torch.no_grad() as we are not training.
    with torch.no_grad():
        # Use tqdm for a user-friendly progress bar
        for viewpoint_cam in tqdm(viewpoints, desc="Rendering and Saving Images"):
            # Step 4a: Render the scene to get the feature map
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            feature_image = render_pkg["render"]

            # Step 4b: Decode the feature map to get the final RGB image
            rgb_image = imagenet(feature_image.unsqueeze(0)).squeeze(0)

            # Step 4c: Convert the tensor to a saveable image format (uint8 numpy array)
            # Clamp values to [0, 1], permute from (C, H, W) to (H, W, C), and scale to [0, 255]
            rgb_image_np = (rgb_image.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # Step 4d: Save the image
            # The viewpoint_cam.image_name attribute holds the original filename (e.g., "r_1.png")
            save_path = os.path.join(output_dir, viewpoint_cam.image_name + ".png")
            Image.fromarray(rgb_image_np).save(save_path)

    print("\n[SUCCESS] All images have been rendered and saved.")


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = ArgumentParser(description="Rendering script based on a pre-trained model.")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # Add arguments for checkpoint path and output directory
    parser.add_argument("--start_checkpoint", type=str, default="output/room_wm/chkpnt10000.pth",  # "output/fern_wm/chkpnt10000.pth"
                        help="Path to the model checkpoint to load.")
    parser.add_argument("--output_dir", type=str, default="data/LLFF/rendered_room_reconstruct",
                        help="Directory to save the rendered images.")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # --- Execution ---
    render_and_save_images(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        checkpoint=args.start_checkpoint,
        output_dir=args.output_dir
    )

    print("\nRendering script finished.")
#
# from decoder import SimpleCNN, WatermarkCNN
# import matplotlib.pyplot as plt
# import os
# import sys
# import torch
# import random
# import argparse
# import numpy as np
# from tqdm import tqdm
# from PIL import Image
# from scene import Scene, GaussianModel
# from gaussian_renderer import render
# from utils.general_utils import safe_state
# from arguments import ModelParams, PipelineParams, OptimizationParams
# import torch.nn as nn
# from plyfile import PlyData
#
#
# import random
# import numpy as np
# random_seed = 0  # 你可以选择任何整数作为随机种子
# random.seed(random_seed)
# torch.manual_seed(random_seed)
# np.random.seed(random_seed)
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# def load_ply_as_initialization(ply_path, dataset, opt):
#     """从PLY文件加载高斯模型参数"""
#     gaussians = GaussianModel(dataset.sh_degree)
#
#     # 读取PLY文件数据
#     plydata = PlyData.read(ply_path)
#     data = plydata.elements[0].data
#
#     # 解析PLY属性（需与官方保存格式完全一致）
#     xyz = np.stack((data['x'], data['y'], data['z'])).transpose()
#     opacities = data['opacity'].reshape(-1, 1)
#
#     # 解析缩放和旋转参数
#     scales = np.stack([data[f'scale_{i}'] for i in range(3)]).transpose()
#     rots = np.stack([data[f'rot_{i}'] for i in range(4)]).transpose()
#
#     # 动态地从PLY文件中找出所有 'semantic_' 开头的字段名，以确定维度
#     semantic_feature_names = [name for name in data.dtype.names if name.startswith('semantic_')]
#     semantic_dim = len(semantic_feature_names)
#     print(f"[信息] 在PLY文件中找到 {semantic_dim} 维的语义特征。")
#
#     # 读取所有语义特征数据
#     semantic_features = np.stack([data[name] for name in semantic_feature_names]).transpose()
#     # 将其 reshape 成模型内部期望的格式 [N, 1, D]
#     semantic_features = semantic_features.reshape(-1, 1, semantic_dim)
#
#     # 转换为Tensor并设置可训练参数
#     gaussians._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
#     gaussians._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
#     gaussians._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
#     gaussians._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
#     gaussians._semantic_feature = nn.Parameter(torch.tensor(semantic_features, dtype=torch.float, device="cuda").requires_grad_(True))
#
#     # 在load_ply_as_initialization函数末尾添加
#     gaussians.xyz_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
#     gaussians.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
#     gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
#
#     # 初始化优化器（关键！需与原始训练设置一致）
#     gaussians.training_setup(opt)
#     return gaussians
#
# def save_rendered_views(scene, gaussians, pipe, dataset, output_dir="output/rendered_views"):
#     """保存所有训练视角的渲染图"""
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 设置背景颜色（需与训练时一致）
#     bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
#     background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
#
#     # 获取所有训练相机（确保Scene初始化时shuffle=False）
#     train_cams = scene.getTrainCameras()
#
#     # 禁用梯度计算以节省显存
#     with torch.no_grad():
#         for idx, viewpoint_cam in tqdm(enumerate(train_cams), desc="渲染进度"):
#             # 执行渲染
#             render_pkg = render(viewpoint_cam, gaussians, pipe, background)
#             feature_map = render_pkg["render"]  # Tensor形状为(3, H, W)
#             decoded_image = imagenet_decoder(feature_map.unsqueeze(0)).squeeze(0)
#
#             # 转换为0-255的numpy数组
#             img_np = decoded_image.clamp(0.0, 1.0).cpu().numpy()
#             img_np = np.transpose(img_np, (1, 2, 0)) * 255  # 转换为HWC格式
#             img_np = img_np.astype(np.uint8)
#
#             # 保存为PNG文件
#             if "blender" in wm_source_path:
#                 Image.fromarray(img_np).save(os.path.join(output_dir, f"r_{idx}.png"))
#             elif "LLFF" in wm_source_path:
#                 Image.fromarray(img_np).save(os.path.join(output_dir, f"{viewpoint_cam.image_name}.png"))
#
#     print(f"渲染完成！结果已保存至：{os.path.abspath(output_dir)}")
#
#
#
#
# if __name__ == "__main__":
#     # 参数解析器配置
#     parser = argparse.ArgumentParser()
#     lp = ModelParams(parser)
#     op = OptimizationParams(parser)
#     pp = PipelineParams(parser)
#
#     ifAttack = 'output/fortress_wm'  # 'output/chair_wm'
#     wm_source_path = "/sda1/qyzeng/GS-Hider/data/LLFF/fortress"
#     load_iteration = 30000
#     args = parser.parse_args([
#         '--model_path', ifAttack,
#         '--sh_degree', '0'
#     ])
#
#     lp._model_path, lp.model_path = args.model_path, args.model_path
#     resume_ply = os.path.join(ifAttack, "point_cloud", f"iteration_{load_iteration}", "point_cloud.ply")
#
#     # 添加渲染保存参数
#     # parser.add_argument("--render_output", type=str, default="data/blender/rendered_ship/train")
#     parser.add_argument("--render_output", type=str, default="data/LLFF/rendered_fortress/images")
#     parser.add_argument("--quiet", action="store_true")
#     parser.add_argument('--detect_anomaly', action='store_true', default=False)
#     args = parser.parse_args()
#
#     # 初始化系统状态
#     safe_state(args.quiet)
#     torch.autograd.set_detect_anomaly(args.detect_anomaly)
#
#     imagenet_decoder = SimpleCNN().cuda()
#     imagenet_checkpoint_path = os.path.join(ifAttack, f"chkpnt{load_iteration}_net.pth")
#     imagenet_decoder.load_state_dict(torch.load(imagenet_checkpoint_path, weights_only=False))
#     imagenet_decoder.eval()  # 设置为评估模式
#     print("[信息] 解码器加载成功。")
#
#     dataset = lp.extract(args)
#     opt = op.extract(args)
#     print(f"[信息] 正在从 {resume_ply} 加载高斯模型...")
#     gaussians = load_ply_as_initialization(resume_ply, dataset, opt)
#     dataset.model_path = ifAttack
#     dataset.source_path = wm_source_path
#     dataset.resolution = 1
#     scene = Scene(dataset, gaussians, load_iteration=0, shuffle=False)
#
#     # 执行渲染保存
#     save_rendered_views(
#         scene,
#         gaussians,
#         pp.extract(args),
#         dataset,
#         args.render_output
#     )
