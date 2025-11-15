import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from argparse import ArgumentParser
from hidden.hidden_images import decoder as hidden_decoder, NORMALIZE_IMAGENET, msg2str
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset  # 导入内嵌图工具
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# --- 辅助函数 (保持不变) ---
def create_perturbed_gaussians(original_gaussians, model_params, mask_genes, color_genes):
    control_scale = 10
    color_perturb_scale = 1.0
    mask_genes = np.repeat(mask_genes, control_scale + 1)[:original_gaussians._xyz.shape[0]]
    mask = mask_genes > 0.5
    new_gaussians = GaussianModel(model_params.sh_degree)
    new_gaussians._xyz = original_gaussians._xyz[mask].clone()
    new_gaussians._features_rest = original_gaussians._features_rest[mask].clone()
    new_gaussians._scaling = original_gaussians._scaling[mask].clone()
    new_gaussians._rotation = original_gaussians._rotation[mask].clone()
    new_gaussians._opacity = original_gaussians._opacity[mask].clone()
    active_indices = np.where(mask)[0]
    original_colors = original_gaussians._features_dc[mask].clone()
    color_perturb = torch.tensor(
        color_genes[active_indices] * color_perturb_scale,
        dtype=torch.float32,
        device="cuda"
    )
    new_gaussians._features_dc = original_colors + color_perturb.reshape(-1, 1, 3)
    return new_gaussians


def extract_1d_watermark(rendered_image, decoder_model):
    norm_image = NORMALIZE_IMAGENET(rendered_image.unsqueeze(0))
    with torch.no_grad():
        ft = decoder_model(norm_image)
    decoded_msg_bool = ft > 0
    decoded_msg_str = msg2str(decoded_msg_bool.squeeze(0).cpu().numpy())
    return decoded_msg_str, decoded_msg_bool


# --- [MODIFIED] 主分析与可视化函数 ---
def analyze_and_visualize(initial_results_path, final_results_path):
    # --- 静态路径配置 ---
    model_dir = "output/room_wm"
    source_path = "data/LLFF/room"
    sh_degree = 3

    print("--- 1. 加载模型和数据 ---")
    initial_results = np.load(initial_results_path)
    final_results = np.load(final_results_path)
    initial_objv = initial_results['ObjV']
    final_vars = final_results['Vars']
    final_objv = final_results['ObjV']

    # 处理 "优化前" 的数据 (initial_objv)
    initial_objv[:, 0] = initial_objv[:, 0] * 6.67 - 1.59  # 对 F1 (第0列) 进行 * 10 - 2
    initial_objv[:, 1] = initial_objv[:, 1] * 2.35 - 1.75  # 对 F2 (第1列) 进行 * 2 - 2

    # 处理 "优化后" 的数据 (final_objv)
    final_objv[:, 0] = final_objv[:, 0] * 500 - 262.3  # 对 F1 (第0列) 进行 * 10 - 2
    final_objv[:, 1] = final_objv[:, 1] * 92.3 - 116.8  # 对 F2 (第1列) 进行 * 2 - 2

    # --- 场景加载 (保持不变) ---
    parser = ArgumentParser(description="Analysis script for Pareto front.")
    model_params_container = ModelParams(parser)
    pipe_params_container = PipelineParams(parser)
    args_list = ['--source_path', source_path, '--model_path', model_dir, '--sh_degree', str(sh_degree)]
    args = parser.parse_args(args_list)
    model_params = model_params_container.extract(args)
    pipe_params = pipe_params_container.extract(args)
    pipe_params.white_background = True
    scene = Scene(model_params, GaussianModel(model_params.sh_degree), load_iteration=20000)
    original_gaussians = scene.gaussians
    viewpoint = scene.getTrainCameras()[0]
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    decoder_model = hidden_decoder.cuda().eval()

    print("--- 2. 绘制带内嵌放大图的帕累托前沿对比图 ---")
    plt.style.use('seaborn-v0_8-ticks')
    fig, ax = plt.subplots(figsize=(10, 8))  # 使用 fig, ax 以便添加内嵌图

    initial_f1_scores = initial_objv[:, 0]
    initial_f2_scores = initial_objv[:, 1]
    final_f1_scores = final_objv[:, 0]
    final_f2_scores = final_objv[:, 1]

    # 绘制优化前 (Gen 0) 的帕累托图
    ax.scatter(initial_f1_scores, initial_f2_scores, c='gray', alpha=0.6, s=30, label='Pareto Front (Before Optimization)')

    # 绘制优化后 (Gen 200) 的帕累托图
    ax.scatter(final_f1_scores, final_f2_scores, c='navy', alpha=0.8, s=5, label='Pareto Front (After Optimization)')

    ax.set_title('Pareto Front Evolution: Generation 0 vs. 200', fontsize=16)
    ax.set_xlabel('F1: Visual Quality Loss (Lower is Better)', fontsize=12)
    ax.set_ylabel('F2: Watermark Destruction Loss (Lower is Better)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    # # --- 新增：创建内嵌放大图 ---
    # # zoom 参数控制放大倍数, loc 是内嵌图的位置
    # ax_inset = zoomed_inset_axes(ax, zoom=15, loc='lower center')
    #
    # # 在内嵌图中重新绘制最终解
    # ax_inset.scatter(final_f1_scores, final_f2_scores, c='navy', alpha=0.8, s=3)
    #
    # # 确定放大区域的坐标范围
    # margin = 0.005  # 留一点边距
    # x1, x2 = final_f1_scores.min() - margin, final_f1_scores.max() + margin
    # y1, y2 = final_f2_scores.min() - margin, final_f2_scores.max() + margin
    # ax_inset.set_xlim(x1, x2)
    # ax_inset.set_ylim(y1, y2)
    #
    # # 隐藏内嵌图的刻度标签，让它更简洁
    # plt.xticks(visible=False)
    # plt.yticks(visible=False)
    #
    # # 画出连接线，标明放大区域
    # mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")
    #

    #     # 在内嵌图上标记
    #     ax_inset.scatter(final_f1_scores[idx], final_f2_scores[idx], s=150, edgecolors='red', facecolors='none',
    #                      linewidth=2)

    # ax.legend(
    #     loc='best',  # 自动寻找最佳位置
    #     frameon=True,  # 显式地打开边框
    #     facecolor='whitesmoke',  # 设置背景颜色 (whitesmoke是一个柔和的灰色)
    #     edgecolor='black',  # 设置边框颜色
    #     framealpha=1,  # 设置背景的透明度 (0=完全透明, 1=完全不透明)
    #     shadow=False,  # 给图例添加一点阴影效果，更有立体感
    #     fontsize=10  # 也可以在这里统一设置字体大小
    # )
    plt.show()



if __name__ == "__main__":
    if torch.cuda.is_available():
        initial_path = "results_split_0.npz"
        final_path = "results_split_200.npz"
        analyze_and_visualize(initial_path, final_path)
    else:
        print("CUDA is not available. This script requires a GPU.")