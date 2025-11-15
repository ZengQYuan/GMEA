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
        self.num = 0
        self.color_perturb_scale = 1  # 颜色扰动幅度控制
        # 优化问题定义（单目标）
        name = '3DGS_Attack_Single_Objective'

        num_gaussians = 17000

        M = 1  # 单目标
        maxormins = [1]  # 最大化目标（-1表示最大化）

        Dim = num_gaussians * 4
        varTypes = [0] * Dim  # 所有变量均为实数

        # 变量边界设置（0-1连续变量）
        lb = [0] * num_gaussians  # mask部分[0,1]
        lb += [-1] * (num_gaussians * 3)  # 颜色扰动[-1,1]
        ub = [1] * num_gaussians
        ub += [1] * (num_gaussians * 3)
        lbin = [1] * Dim
        ubin = [1] * Dim
        self.num_gaussians = num_gaussians

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        # 解码种群获得掩膜矩阵 [NIND x Dim]
        genes = pop.Phen[:, :self.num_gaussians]

        obj = np.sum(genes > 0.5, axis=1).reshape(-1, 1)
        pop.ObjV = obj

        print(f"Generation {self.num}")
        self.num += 1


    def plot_fitness_curves(self):
        plt.figure(figsize=(10, 6))

        # 绘制平均适应度曲线
        plt.plot(self.obj_history, 'b-', label='Average Active Gaussians', linewidth=2)
        plt.plot(self.best_obj, 'r--', label='Best Active Gaussians', linewidth=2)

        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Number of Active Gaussians', fontsize=12)
        plt.title('Gaussian Sparsity Optimization (DE Algorithm)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.tight_layout()
        plt.savefig('single_objective_fitness_curve.png', dpi=300)
        plt.show()

    def create_perturbed_gaussians(self, original_gaussians, mask_genes, color_genes):
        """创建应用了mask和颜色扰动的高斯模型"""
        mask = mask_genes > 0.5  # 二值化mask

        # 转换NumPy数组为PyTorch张量
        # mask_genes_tensor = torch.tensor(mask_genes, dtype=torch.float32, device="cuda")

        # 使用张量计算Sigmoid
        # mask = 1 / (1 + torch.exp(-10 * (mask_genes_tensor - 0.5)))

        new_gaussians = GaussianModel(model_params.sh_degree)

        # mask = mask.cpu().detach().numpy()

        # 复制选中的高斯球属性
        new_gaussians._xyz = original_gaussians._xyz[mask].clone()
        new_gaussians._features_rest = original_gaussians._features_rest[mask].clone()
        new_gaussians._scaling = original_gaussians._scaling[mask].clone()
        new_gaussians._rotation = original_gaussians._rotation[mask].clone()
        new_gaussians._opacity = original_gaussians._opacity[mask].clone()

        # 应用颜色扰动
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

    # 4. 算法设置（差分进化DE算法）
    algorithm = ea.soea_DE_rand_1_bin_templet(
        problem,
        ea.Population(Encoding='RI', NIND=50),
        MAXGEN=2500,  # 最大进化代数
        logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    algorithm.mutOper.F = 0.15  # 设置变异因子（默认0.5，范围0-2）[3,6](@ref)
    algorithm.recOper.XOVR = 0.15  # 设置交叉概率（默认0.5，范围0-1）[3,6](@ref)

    # 5. 运行优化
    res = ea.optimize(algorithm, verbose=True,
                                drawing=1,
                                outputMsg=False,
                                drawLog=True,
                                saveFlag=False)

    # 6. 结果处理
    problem.plot_fitness_curves()
    best_solution = res['Vars'].copy()  # 创建独立副本

    num_gaussians = original_gaussians.get_xyz.shape[0]
    m = best_solution[:num_gaussians]
    # c = best_solution[num_gaussians:].reshape(num_gaussians, 3)
    c = np.zeros((num_gaussians, 3))

    best_gaussians = problem.create_perturbed_gaussians(original_gaussians, m, c)
    with torch.no_grad():
        temp_render = render(viewpoint_cam, best_gaussians, pipe_params, background)
        temp_image = temp_render["render"]
    plt.imshow(temp_image.cpu().detach().numpy().transpose(1, 2, 0))
    plt.show()




# import numpy as np
# import geatpy as ea
# import torch
# from scene import Scene, GaussianModel
# from gaussian_renderer import render
# from utils.loss_utils import l1_loss, ssim
# import matplotlib.pyplot as plt
# from argparse import ArgumentParser, Namespace
# from arguments import ModelParams, PipelineParams
#
# from torchvision import transforms
# from hidden.hidden_images import decoder
#
# import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# import random
#
# random_seed = 0  # 你可以选择任何整数作为随机种子
# random.seed(random_seed)
# torch.manual_seed(random_seed)
# np.random.seed(random_seed)
#
#
# class GS_AttackProblem(ea.Problem):
#     def __init__(self, original_gaussians, target_image, pipe_params, decoder):
#         # 初始化问题参数
#         self.decoder = decoder
#         self.original_gaussians = original_gaussians
#         self.target_image = target_image  # 原始渲染图像
#         self.pipe = pipe_params
#         self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
#         self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         self.num = 0
#         self.color_perturb_scale = 1  # 颜色扰动幅度控制
#         # 优化问题定义（单目标）
#         name = '3DGS_Attack_Single_Objective'
#
#         DIM = 17000
#         M = 1  # 单目标
#         maxormins = [1]  # 最大化目标（-1表示最大化）
#         varTypes = [0] * DIM  # 连续型变量
#         lb = [0] * DIM  # 变量下界
#         ub = [1] * DIM  # 变量上界
#         lbin = [1] * DIM  # 包含下边界
#         ubin = [1] * DIM  # 包含上边界
#
#         ea.Problem.__init__(self, name, M, maxormins, DIM, varTypes, lb, ub, lbin, ubin)
#
#     def aimFunc(self, pop):
#         # 解码种群获得掩膜矩阵 [NIND x Dim]
#         genes = pop.Phen
#
#         obj = np.sum(genes > 0.5, axis=1).reshape(-1, 1)
#         pop.ObjV = obj
#
#         print(f"Generation {self.num}")
#         self.num += 1
#
#
#     def plot_fitness_curves(self):
#         plt.figure(figsize=(10, 6))
#
#         # 绘制平均适应度曲线
#         plt.plot(self.obj_history, 'b-', label='Average Active Gaussians', linewidth=2)
#         plt.plot(self.best_obj, 'r--', label='Best Active Gaussians', linewidth=2)
#
#         plt.xlabel('Generation', fontsize=12)
#         plt.ylabel('Number of Active Gaussians', fontsize=12)
#         plt.title('Gaussian Sparsity Optimization (DE Algorithm)', fontsize=14)
#         plt.grid(True, linestyle='--', alpha=0.7)
#         plt.legend(fontsize=10)
#         plt.xticks(fontsize=10)
#         plt.yticks(fontsize=10)
#
#         plt.tight_layout()
#         plt.savefig('single_objective_fitness_curve.png', dpi=300)
#         plt.show()
#
#     def create_perturbed_gaussians(self, original_gaussians, mask_genes, color_genes):
#         """创建应用了mask和颜色扰动的高斯模型"""
#         mask = mask_genes > 0.5  # 二值化mask
#
#         # 转换NumPy数组为PyTorch张量
#         # mask_genes_tensor = torch.tensor(mask_genes, dtype=torch.float32, device="cuda")
#
#         # 使用张量计算Sigmoid
#         # mask = 1 / (1 + torch.exp(-10 * (mask_genes_tensor - 0.5)))
#
#         new_gaussians = GaussianModel(model_params.sh_degree)
#
#         # mask = mask.cpu().detach().numpy()
#
#         # 复制选中的高斯球属性
#         new_gaussians._xyz = original_gaussians._xyz[mask].clone()
#         new_gaussians._features_rest = original_gaussians._features_rest[mask].clone()
#         new_gaussians._scaling = original_gaussians._scaling[mask].clone()
#         new_gaussians._rotation = original_gaussians._rotation[mask].clone()
#         new_gaussians._opacity = original_gaussians._opacity[mask].clone()
#
#         # 应用颜色扰动
#         active_indices = np.where(mask)[0]
#         original_colors = original_gaussians._features_dc[mask].clone()
#
#         # 颜色扰动 = 扰动基因 * 幅度控制
#         color_perturb = torch.tensor(
#             color_genes[active_indices] * self.color_perturb_scale,
#             dtype=torch.float32,
#             device="cuda"
#         )
#
#         new_gaussians._features_dc = original_colors + color_perturb.reshape(-1, 1, 3)
#
#         return new_gaussians
#
#
# # 使用示例
# if __name__ == "__main__":
#     parser = ArgumentParser()
#     model_params = ModelParams(parser)
#     args = parser.parse_args([
#         '--source_path', 'data/blender/lego',
#         '--model_path', 'output/lego_wm',
#         '--sh_degree', '3'
#     ])
#
#     model_params.model_path, model_params._model_path = args.model_path, args.model_path
#     model_params = model_params.extract(args)
#
#     pipe_params = PipelineParams(ArgumentParser())
#     pipe_params.convert_SHs_python = False
#     pipe_params.compute_cov3D_python = False
#     pipe_params.white_background = True  # 与训练时背景设置一致
#
#     # 1. 加载原始3DGS模型
#     original_scene = Scene(model_params,
#                            GaussianModel(model_params.sh_degree),
#                            load_iteration=2000)  # 指定最终迭代次数
#     original_gaussians = original_scene.gaussians
#
#     # 获取第一个训练视角的相机参数
#     viewpoint_stack = original_scene.getTrainCameras().copy()
#     viewpoint_cam = viewpoint_stack.pop(0)  # 取第一个训练视角
#     # 背景色设置（需与训练时一致）
#     bg_color = [1, 1, 1] if pipe_params.white_background else [0, 0, 0]
#     background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
#
#     # 2. 渲染原始图像作为参考
#     with torch.no_grad():
#         original_render = render(original_scene.getTrainCameras()[0],
#                                  original_gaussians, pipe_params, background)
#         target_image = original_render["render"]
#
#     # 3. 配置优化参数
#     problem = GS_AttackProblem(original_gaussians, target_image,
#                                {'viewpoint': viewpoint_cam, 'pipe': pipe_params, 'background': background}, decoder)
#
#     # 4. 算法设置（差分进化DE算法）
#     algorithm = ea.soea_DE_rand_1_bin_templet(
#         problem,
#         ea.Population(Encoding='RI', NIND=50),
#         MAXGEN=2500,  # 最大进化代数
#         logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。
#     algorithm.mutOper.F = 0.15  # 设置变异因子（默认0.5，范围0-2）[3,6](@ref)
#     algorithm.recOper.XOVR = 0.15  # 设置交叉概率（默认0.5，范围0-1）[3,6](@ref)
#
#     # 5. 运行优化
#     res = ea.optimize(algorithm, verbose=True,
#                                 drawing=1,
#                                 outputMsg=False,
#                                 drawLog=True,
#                                 saveFlag=False)
#
#     # 6. 结果处理
#     problem.plot_fitness_curves()
#     best_solution = res['Vars'].copy()  # 创建独立副本
#
#     num_gaussians = original_gaussians.get_xyz.shape[0]
#     m = best_solution[:num_gaussians]
#     # c = best_solution[num_gaussians:].reshape(num_gaussians, 3)
#     c = np.zeros((num_gaussians, 3))
#
#     best_gaussians = problem.create_perturbed_gaussians(original_gaussians, m, c)
#     with torch.no_grad():
#         temp_render = render(viewpoint_cam, best_gaussians, pipe_params, background)
#         temp_image = temp_render["render"]
#     plt.imshow(temp_image.cpu().detach().numpy().transpose(1, 2, 0))
#     plt.show()
