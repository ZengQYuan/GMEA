

import numpy as np
import geatpy as ea
import torch
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams

from torchvision import transforms
from hidden.hidden_images import decoder

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random

random_seed = 0  # 你可以选择任何整数作为随机种子
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

class GS_AttackProblem(ea.Problem):
    def __init__(self, original_gaussians, target_image, pipe_params, decoder):
        # 初始化问题参数
        self.decoder = decoder
        self.original_gaussians = original_gaussians
        self.target_image = target_image  # 原始渲染图像
        self.pipe = pipe_params
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._register_hook()  # 注册钩子捕获中间特征
        self.num = 0
        self.color_perturb_scale = 1  # 颜色扰动幅度控制

        self.obj1_history = []  # 记录每代平均目标1
        self.obj2_history = []  # 记录每代平均目标2
        self.best_obj1 = []  # 记录每代最优目标1
        self.best_obj2 = []  # 记录每代最优目标2
        self.obj3_history = []  # 记录每代平均目标3
        self.best_obj3 = []  # 记录每代最优目标3

        # 优化问题定义
        name = '3DGS_Attack'
        M = 3  # 双目标优化
        maxormins = [1, 1, 1]  # 两个目标都需要最小化 1是最小化 -1是最大化
        num_gaussians = original_gaussians.get_xyz.shape[0]  # 高斯球个数

        # 染色体编码：每个高斯球4个参数（1个mask + 3个颜色扰动）[10](@ref)
        Dim = num_gaussians * 4
        varTypes = [0] * Dim  # 1 二进制变量 0 实数型变量

        # 变量边界设置（0-1整数）
        lb = [0] * num_gaussians  # mask部分[0,1]
        lb += [-1] * (num_gaussians * 3)  # 颜色扰动[-1,1]
        ub = [1] * num_gaussians
        ub += [1] * (num_gaussians * 3)
        lbin = [1] * Dim
        ubin = [1] * Dim

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        # 解码种群获得掩膜矩阵 [NIND x Dim]
        genes = pop.Phen
        num_individuals = pop.sizes
        num_gaussians = self.original_gaussians.get_xyz.shape[0]

        ObjV = []

        for i in range(num_individuals):
            # 分离mask和颜色扰动基因
            mask_genes = genes[i, :num_gaussians]
            color_genes = genes[i, num_gaussians:].reshape(num_gaussians, 3)

            # 创建扰动后的高斯模型
            temp_gaussians = self.create_perturbed_gaussians(
                self.original_gaussians,
                mask_genes,
                color_genes
            )

            with torch.no_grad():
                temp_render = render(viewpoint_cam, temp_gaussians, pipe_params, background)
                temp_image = temp_render["render"]

            # 目标1：渲染质量损失
            with torch.no_grad():
                obj1 = self.calculate_quality_loss(temp_image, self.target_image)

                obj2 = self.calculate_feature_std(temp_image)

                active_gaussians = (mask_genes > 0.5).sum()
                obj3 = active_gaussians

            ObjV.append([obj1, obj2, obj3])

        print(self.num)
        self.num = self.num + 1

        pop.ObjV = np.array(ObjV)

        avg_obj1 = np.mean(pop.ObjV[:, 0])
        avg_obj2 = np.mean(pop.ObjV[:, 1])
        avg_obj3 = np.mean(pop.ObjV[:, 2])

        self.obj1_history.append(avg_obj1)
        self.obj2_history.append(avg_obj2)
        self.obj3_history.append(avg_obj3)
        # 非支配排序
        front_ranks = ea.ndsortESS(pop.ObjV, CV=pop.CV, needLevel=1)
        front = np.where(front_ranks == 1)[0]

        # 记录前沿解的目标值
        if len(front) == 0:
            self.best_obj1.append(np.inf)
            self.best_obj2.append(np.inf)
            self.best_obj3.append(np.inf)
        else:
            best_idx = np.argmin(pop.ObjV[front][:, 0] + pop.ObjV[front][:, 1])
            self.best_obj1.append(pop.ObjV[front][best_idx, 0])
            self.best_obj2.append(pop.ObjV[front][best_idx, 1])
            self.best_obj3.append(pop.ObjV[front][best_idx, 2])

    def plot_fitness_curves(self):
        plt.figure(figsize=(15, 10))

        # 1. 平均适应度曲线（三目标）
        plt.subplot(2, 2, 1)
        plt.plot(self.obj1_history, 'b-', label='Avg Obj1 (Quality Loss)', linewidth=2)
        plt.plot(self.obj2_history, 'r--', label='Avg Obj2 (Feature STD)', linewidth=2)
        plt.plot(self.obj3_history, 'g-.', label='Avg Obj3 (Active Gaussians)', linewidth=2)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness Value', fontsize=12)
        plt.title('Average Fitness Trends', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # 2. 最优适应度曲线（三目标）
        plt.subplot(2, 2, 2)
        plt.plot(self.best_obj1, 'b-', label='Best Obj1', linewidth=2)
        plt.plot(self.best_obj2, 'r--', label='Best Obj2', linewidth=2)
        plt.plot(self.best_obj3, 'g-.', label='Best Obj3', linewidth=2)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness Value', fontsize=12)
        plt.title('Elite Fitness Trends', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # 3. 目标1和目标2的关系
        plt.subplot(2, 2, 3)
        plt.scatter(self.best_obj1, self.best_obj2, c=range(len(self.best_obj1)),
                    cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(label='Generation')
        plt.xlabel('Objective 1 (Quality Loss)', fontsize=12)
        plt.ylabel('Objective 2 (Feature STD)', fontsize=12)
        plt.title('Trade-off between Obj1 and Obj2', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.5)

        # 4. 目标3的变化趋势（单独展示）
        plt.subplot(2, 2, 4)
        plt.plot(self.obj3_history, 'm-', label='Average Active Gaussians', linewidth=2)
        plt.plot(self.best_obj3, 'c--', label='Best Active Gaussians', linewidth=2)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Number of Active Gaussians', fontsize=12)
        plt.title('Gaussian Sparsity Trend', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)

        plt.tight_layout(pad=3.0)
        plt.savefig('multi_objective_fitness_curves.png', dpi=300)
        plt.show()

    def calculate_feature_std(self, rendered_image):
        """计算HiDDeN模型中间特征图的标准差"""
        with torch.no_grad():
            # 图像预处理（需与HiDDeN训练时一致）
            norm_image = self.normalize(rendered_image.unsqueeze(0))

            # 前向传播获取中间特征
            _ = self.decoder(norm_image)
            features = self.current_features

            # 计算通道间标准差（仿照Dispersion攻击）
            std_per_channel = features.std(dim=(2, 3))  # 计算各通道空间维度的标准差
            total_std = std_per_channel.mean()  # 取通道间平均

        return total_std.item()
    def _register_hook(self):
        """注册前向钩子捕获中间特征图"""

        def hook(module, input, output):
            self.current_features = output.detach()

        # 根据decoder结构选择目标层
        target_layer = self.decoder.layers[-2].layers[-3]
        target_layer.register_forward_hook(hook)

    def create_perturbed_gaussians(self, original_gaussians, mask_genes, color_genes):
        """创建应用了mask和颜色扰动的高斯模型"""
        mask = mask_genes > 0.5  # 二值化mask
        new_gaussians = GaussianModel(model_params.sh_degree)

        # 复制选中的高斯球属性
        new_gaussians._xyz = original_gaussians._xyz[mask].clone()
        new_gaussians._features_rest = original_gaussians._features_rest[mask].clone()
        new_gaussians._scaling = original_gaussians._scaling[mask].clone()
        new_gaussians._rotation = original_gaussians._rotation[mask].clone()
        new_gaussians._opacity = original_gaussians._opacity[mask].clone()

        # 应用颜色扰动[1](@ref)
        active_indices = np.where(mask)[0]
        original_colors = original_gaussians._features_dc[mask].clone()

        # 颜色扰动 = 扰动基因 * 幅度控制
        color_perturb = torch.tensor(
            color_genes[active_indices] * self.color_perturb_scale,
            dtype=torch.float32,
            device="cuda"
        )

        new_gaussians._features_dc = original_colors + color_perturb.reshape(-1, 1, 3)

        return new_gaussians

    def calculate_quality_loss(self, rendered_image, target_image):
        """计算渲染质量损失（多指标融合）"""
        l1 = l1_loss(rendered_image, target_image).item()
        ssim_loss = 1 - ssim(rendered_image, target_image).item()
        return 0.7 * l1 + 0.3 * ssim_loss  # 组合损失


# 使用示例
if __name__ == "__main__":
    parser = ArgumentParser()
    model_params = ModelParams(parser)
    args = parser.parse_args([
        '--source_path', 'data/blender/lego',
        '--model_path', 'output/lego_wm',
        '--sh_degree', '3'
    ])

    model_params.model_path, model_params._model_path = args.model_path, args.model_path
    model_params = model_params.extract(args)

    pipe_params = PipelineParams(ArgumentParser())
    pipe_params.convert_SHs_python = False
    pipe_params.compute_cov3D_python = False
    pipe_params.white_background = True  # 与训练时背景设置一致

    # 1. 加载原始3DGS模型
    original_scene = Scene(model_params,
                  GaussianModel(model_params.sh_degree),
                  load_iteration=2000)  # 指定最终迭代次数
    original_gaussians = original_scene.gaussians

    # 获取第一个训练视角的相机参数
    viewpoint_stack = original_scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(0)  # 取第一个训练视角
    # 背景色设置（需与训练时一致）
    bg_color = [1, 1, 1] if pipe_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 2. 渲染原始图像作为参考
    with torch.no_grad():
        original_render = render(original_scene.getTrainCameras()[0],
                                 original_gaussians, pipe_params, background)
        target_image = original_render["render"]

    # 3. 配置优化参数
    problem = GS_AttackProblem(original_gaussians, target_image,
                               {'viewpoint': viewpoint_cam, 'pipe': pipe_params, 'background': background}, decoder)

    # 4. 算法设置（NSGA-II）
    algorithm = ea.moea_NSGA2_templet(
        problem,
        ea.Population(Encoding='RI', NIND=50),
        MAXGEN=30,  # 最大进化代数
        logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。

    # 5. 运行优化
    res = ea.optimize(algorithm, verbose=True, drawLog=True, outputMsg=True, drawing=1)

    # 6. 结果处理
    problem.plot_fitness_curves()
    best_masks = res['Vars'].copy()  # 创建独立副本
    m = best_masks[0, :174263]
    c = best_masks[0, 174263:].reshape(174263, 3)
    best_gaussians = problem.create_perturbed_gaussians(original_gaussians, m, c)
    with torch.no_grad():
        temp_render = render(viewpoint_cam, best_gaussians, pipe_params, background)
        temp_image = temp_render["render"]
    plt.imshow(temp_image.cpu().detach().numpy().transpose(1, 2, 0))
    plt.show()


    print("1")

    save_dir = "output/lego_attack/best_gaussians"
    os.makedirs(save_dir, exist_ok=True)
    for i, gaussian in enumerate(best_gaussians):
        ply_path = os.path.join(save_dir, f"best_gaussian_{i}.ply")
        gaussian.save_ply(ply_path)


    # best_masks = res['Vars']
    # best_masks[:,50000:] = 0
    # best_gaussians = problem.create_masked_gaussians(original_gaussians, best_masks)
    # best_gaussians[0].save_ply("output/lego_attack/point_cloud/iteration_2000/point_cloud.ply")
