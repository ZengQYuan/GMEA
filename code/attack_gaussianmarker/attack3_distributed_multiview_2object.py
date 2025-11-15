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
from sklearn.cluster import KMeans
import time  # [ADDED] 导入 time 模块
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random

random_seed = 0  # 固定随机种子保证可复现性
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)


class GS_AttackProblem(ea.Problem):
    def __init__(self, original_gaussians, target_images, pipe_params, decoder):
        # 初始化问题参数
        self.decoder = decoder
        self.original_gaussians = original_gaussians
        self.target_images = target_images  # 多个视角的目标图像
        self.pipe = pipe_params
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._register_hook()  # 注册钩子捕获中间特征
        self.num = 0
        self.color_perturb_scale = 1  # 颜色扰动幅度控制
        self.control_scale = 10

        self.obj1_history = []  # 记录每代平均目标1
        self.obj2_history = []  # 记录每代平均目标2
        self.best_obj1 = []  # 记录每代最优目标1
        self.best_obj2 = []  # 记录每代最优目标2
        # [DELETED] 删除了 obj3_history 和 best_obj3 的定义
        # self.obj3_history = []
        # self.best_obj3 = []
        self.generation_times = []  # [ADDED] 用于记录每一代耗时的列表
        self.feature_maps = []

        # 优化问题定义
        name = '3DGS_Attack'
        M = 2  # [MODIFIED] 多目标优化（从3个目标改为2个）
        maxormins = [1, 1]  # [MODIFIED] 两个目标都需要最小化
        num_gaussians = original_gaussians._xyz.shape[0]  # 高斯球个数
        self.num_control_variables = num_gaussians // self.control_scale  # 每个控制变量管理10个高斯球

        # 染色体编码保持不变，因为mask基因仍然是必要的
        Dim = self.num_control_variables + num_gaussians * 3
        varTypes = [0] * Dim

        # 变量边界设置保持不变
        lb = [0] * self.num_control_variables
        lb += [-1] * (num_gaussians * 3)
        ub = [1] * self.num_control_variables
        ub += [1] * (num_gaussians * 3)
        lbin = [1] * Dim
        ubin = [1] * Dim

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        start_time = time.perf_counter()  # [ADDED] 记录该代开始时间
        genes = pop.Phen
        num_individuals = pop.sizes
        num_gaussians = self.original_gaussians._xyz.shape[0]

        ObjV = []

        for i in range(num_individuals):
            control_genes = genes[i, :self.num_control_variables]
            color_genes = genes[i, self.num_control_variables:].reshape(num_gaussians, 3)

            temp_gaussians = self.create_perturbed_gaussians(
                self.original_gaussians,
                control_genes,
                color_genes
            )

            total_obj1 = 0.0
            for view_idx in range(num_views):
                with torch.no_grad():
                    temp_render = render(self.pipe["viewpoints"][view_idx],
                                         temp_gaussians,
                                         self.pipe["pipe"],
                                         self.pipe["background"])
                    temp_image = temp_render["render"]
                current_obj1 = self.calculate_quality_loss(temp_image, self.target_images[view_idx])
                total_obj1 += current_obj1

            with torch.no_grad():
                obj2, feature_map = self.calculate_feature_std(temp_image)

                # [DELETED] 删除了计算 obj3 的逻辑
                # active_gaussians = (control_genes > 0.5).sum()
                # obj3 = active_gaussians

            # [MODIFIED] ObjV现在只包含两个目标
            ObjV.append([total_obj1, obj2])

        print("Generation： ", self.num)
        self.num = self.num + 1

        pop.ObjV = np.array(ObjV)

        avg_obj1 = np.mean(pop.ObjV[:, 0])
        avg_obj2 = np.mean(pop.ObjV[:, 1])
        # [DELETED] 删除了计算 avg_obj3 的逻辑
        # avg_obj3 = np.mean(pop.ObjV[:, 2])

        self.obj1_history.append(avg_obj1)
        self.obj2_history.append(avg_obj2)
        # [DELETED] 删除了记录 obj3_history 的逻辑
        # self.obj3_history.append(avg_obj3)

        front_ranks = ea.ndsortESS(pop.ObjV, CV=pop.CV, needLevel=1)
        front = np.where(front_ranks == 1)[0]

        if len(front) == 0:
            self.best_obj1.append(np.inf)
            self.best_obj2.append(np.inf)
            # [DELETED] 删除了记录 best_obj3 的逻辑
            # self.best_obj3.append(np.inf)
        else:
            best_idx = np.argmin(pop.ObjV[front][:, 0] + pop.ObjV[front][:, 1])
            self.best_obj1.append(pop.ObjV[front][best_idx, 0])
            self.best_obj2.append(pop.ObjV[front][best_idx, 1])
            # [DELETED] 删除了记录 best_obj3 的逻辑
            # self.best_obj3.append(pop.ObjV[front][best_idx, 2])

        end_time = time.perf_counter()  # [ADDED] 记录该代结束时间
        duration = end_time - start_time  # [ADDED] 计算耗时
        self.generation_times.append(duration)  # [ADDED] 保存耗时

        # [MODIFIED] 打印信息中加入耗时
        print(f"Generation: {self.num}, Time taken: {duration:.2f} seconds")
        self.num = self.num + 1

    def plot_fitness_curves(self):
        # [MODIFIED] 调整图表布局为 2x2，最后一个图留空或用于其他分析
        plt.figure(figsize=(15, 10))

        # 1. 平均适应度曲线（两目标）
        plt.subplot(2, 2, 1)
        plt.plot(self.obj1_history, 'b-', label='Avg Obj1 (Quality Loss)', linewidth=2)
        plt.plot(self.obj2_history, 'r--', label='Avg Obj2 (Feature STD)', linewidth=2)
        # [DELETED] 删除了绘制 obj3_history 的代码
        # plt.plot(self.obj3_history, 'g-.', label='Avg Obj3 (Active Gaussians)', linewidth=2)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness Value', fontsize=12)
        plt.title('Average Fitness Trends', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # 2. 最优适应度曲线（两目标）
        plt.subplot(2, 2, 2)
        plt.plot(self.best_obj1, 'b-', label='Best Obj1', linewidth=2)
        plt.plot(self.best_obj2, 'r--', label='Best Obj2', linewidth=2)
        # [DELETED] 删除了绘制 best_obj3 的代码
        # plt.plot(self.best_obj3, 'g-.', label='Best Obj3', linewidth=2)
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

        # [DELETED] 删除了第四个子图（关于obj3）的全部内容

        plt.tight_layout(pad=3.0)
        plt.savefig('multi_objective_fitness_curves_2obj.png', dpi=300)  # [MODIFIED] 修改保存文件名
        plt.show()

    # ... 其他GS_AttackProblem中的函数保持不变 ...
    def calculate_feature_std(self, rendered_image):
        """计算HiDDeN模型中间特征图的标准差"""
        with torch.no_grad():
            norm_image = self.normalize(rendered_image.unsqueeze(0))
            _ = self.decoder(norm_image)
            features = self.current_features
            std_per_channel = features.std(dim=(2, 3))
            total_std = std_per_channel.mean()
        return total_std.item(), features

    def _register_hook(self):
        """注册前向钩子捕获中间特征图"""

        def hook(module, input, output):
            self.current_features = output.detach()

        target_layer = self.decoder.layers[-2].layers[-3]
        target_layer.register_forward_hook(hook)

    def create_perturbed_gaussians(self, original_gaussians, mask_genes, color_genes):
        """创建应用了mask和颜色扰动的高斯模型"""
        mask_genes = np.repeat(mask_genes, self.control_scale + 1)[:self.original_gaussians._xyz.shape[0]]
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
        return 0.7 * l1 + 0.3 * ssim_loss


class SplitGS_AttackProblem:
    def __init__(self, original_gaussians, pipe_params, decoder, num_splits, num_views):
        self.num_splits = num_splits
        self.original_gaussians = original_gaussians
        self.pipe = pipe_params
        self.decoder = decoder
        self.num_views = num_views

        self.obj1_history = []
        self.obj2_history = []
        # [DELETED] 删除了 obj3_history 的定义
        # self.obj3_history = []

        # ... 其他 __init__ 代码保持不变 ...
        total_gaussians = original_gaussians._xyz.shape[0]
        self.chunk_size = total_gaussians // num_splits
        self.remaining = total_gaussians % num_splits
        self.split_indices = []
        start_idx = 0
        for i in range(num_splits):
            end_idx = start_idx + self.chunk_size + (1 if i < self.remaining else 0)
            self.split_indices.append((start_idx, end_idx))
            start_idx = end_idx

    def split_and_optimize(self):
        # ... split_gaussians_by_kmeans 和 generate_sub_target_images 函数保持不变 ...
        def split_gaussians_by_kmeans(original_gaussians, k):
            xyz = original_gaussians.get_xyz.cpu().detach().numpy()
            kmeans = KMeans(n_clusters=k, random_state=random_seed)
            cluster_labels = kmeans.fit_predict(xyz)
            sub_gaussians_list = []
            for cluster_idx in range(k):
                indices = np.where(cluster_labels == cluster_idx)[0]
                sub_gaussians = GaussianModel(model_params.sh_degree)
                sub_gaussians._xyz = original_gaussians._xyz[indices].clone()
                sub_gaussians._features_dc = original_gaussians._features_dc[indices].clone()
                sub_gaussians._features_rest = original_gaussians._features_rest[indices].clone()
                sub_gaussians._scaling = original_gaussians._scaling[indices].clone()
                sub_gaussians._rotation = original_gaussians._rotation[indices].clone()
                sub_gaussians._opacity = original_gaussians._opacity[indices].clone()
                sub_gaussians_list.append(sub_gaussians)
            return sub_gaussians_list

        def generate_sub_target_images(sub_gaussians_list):
            target_images = []
            for sub_gaussians in sub_gaussians_list:
                sub_target_images = []
                for view_idx in range(self.num_views):
                    with torch.no_grad():
                        sub_render = render(self.pipe["viewpoints"][view_idx],
                                            sub_gaussians,
                                            self.pipe["pipe"],
                                            self.pipe["background"])
                        sub_target_images.append(sub_render["render"])
                target_images.append(sub_target_images)
            return target_images

        all_results = {}
        sub_gaussians_list = split_gaussians_by_kmeans(self.original_gaussians, k=self.num_splits)
        sub_target_images_list = generate_sub_target_images(sub_gaussians_list)

        for split_idx in range(len(sub_gaussians_list)):
            problem = GS_AttackProblem(sub_gaussians_list[split_idx], sub_target_images_list[split_idx], self.pipe,
                                       self.decoder)
            algorithm = ea.moea_NSGA2_templet(
                problem,
                ea.Population(Encoding='RI', NIND=10),
                MAXGEN=200,
                logTras=0
            )
            res = ea.optimize(algorithm, verbose=False, drawLog=False, outputMsg=False, drawing=0)

            self.obj1_history.append(problem.obj1_history)
            self.obj2_history.append(problem.obj2_history)
            # [DELETED] 删除了记录 obj3_history 的逻辑
            # self.obj3_history.append(problem.obj3_history)

            all_results[split_idx] = {
                "best_masks": res['Vars'],
                "best_gaussians": problem.create_perturbed_gaussians(
                    sub_gaussians_list[split_idx],
                    res['Vars'][0, :problem.num_control_variables],
                    res['Vars'][0, problem.num_control_variables:].reshape(-1, 3)
                )
            }

            # 保存结果，现在只保存 Vars 和 ObjV
            # np.savez(
            #     f"results_split_{split_idx}.npz",
            #     Vars=res['Vars'],
            #     ObjV=res['ObjV']
            # )

        return all_results


if __name__ == "__main__":
    # ... __main__ 中的代码大部分保持不变 ...
    parser = ArgumentParser()
    model_params = ModelParams(parser)
    args = parser.parse_args([
        '--source_path', 'data/LLFF/room',
        '--model_path', 'output/room_wm',
        '--sh_degree', '3'
    ])
    save_dir = 'output/room_attack/merged_gaussians_'  # [MODIFIED] 修正了变量名

    model_params.model_path, model_params._model_path = args.model_path, args.model_path
    model_params = model_params.extract(args)
    if "blender" in args.source_path:
        model_params.resolution = 4
    elif "LLFF" in args.source_path:
        model_params.resolution = 2

    pipe_params = PipelineParams(ArgumentParser())
    pipe_params.convert_SHs_python = False
    pipe_params.compute_cov3D_python = False
    pipe_params.white_background = True

    original_scene = Scene(model_params,
                           GaussianModel(model_params.sh_degree),
                           load_iteration=20000)
    original_gaussians = original_scene.gaussians

    viewpoint_stack = original_scene.getTrainCameras().copy()
    num_views = 3
    selected_viewpoints = [viewpoint_stack[i] for i in range(min(num_views, len(viewpoint_stack)))]

    bg_color = [1, 1, 1] if pipe_params.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    split_problem = SplitGS_AttackProblem(
        original_gaussians,
        {'viewpoints': selected_viewpoints, 'pipe': pipe_params, 'background': background},
        decoder,
        num_splits=1,
        num_views=num_views
    )

    all_results = split_problem.split_and_optimize()

    # ... 合并与可视化的逻辑保持不变 ...
    total_optimized_gaussians = 0
    for split_idx in range(split_problem.num_splits):
        split_result = all_results[split_idx]
        split_gaussians = split_result["best_gaussians"]
        total_optimized_gaussians += split_gaussians._xyz.shape[0]

    merged_gaussians = GaussianModel(model_params.sh_degree)
    merged_gaussians._xyz = torch.zeros((total_optimized_gaussians, 3), device="cuda")
    merged_gaussians._features_dc = torch.zeros((total_optimized_gaussians, 1, 3), device="cuda")
    merged_gaussians._features_rest = torch.zeros((total_optimized_gaussians, 15, 3), device="cuda")
    merged_gaussians._scaling = torch.zeros((total_optimized_gaussians, 3), device="cuda")
    merged_gaussians._rotation = torch.zeros((total_optimized_gaussians, 4), device="cuda")
    merged_gaussians._opacity = torch.zeros((total_optimized_gaussians, 1), device="cuda")

    start_idx = 0
    for split_idx in range(split_problem.num_splits):
        split_result = all_results[split_idx]
        split_gaussians = split_result["best_gaussians"]
        split_size = split_gaussians._xyz.shape[0]
        if split_size == 0:
            continue
        end_idx = start_idx + split_size
        merged_gaussians._xyz[start_idx:end_idx] = split_gaussians._xyz
        merged_gaussians._features_dc[start_idx:end_idx] = split_gaussians._features_dc
        merged_gaussians._features_rest[start_idx:end_idx] = split_gaussians._features_rest
        merged_gaussians._scaling[start_idx:end_idx] = split_gaussians._scaling
        merged_gaussians._rotation[start_idx:end_idx] = split_gaussians._rotation
        merged_gaussians._opacity[start_idx:end_idx] = split_gaussians._opacity
        start_idx += split_size

    with torch.no_grad():
        merged_render = render(viewpoint_stack[0], merged_gaussians, pipe_params, background)
        merged_image = merged_render["render"]
    plt.imshow(merged_image.cpu().detach().numpy().transpose(1, 2, 0))
    plt.title("Merged Optimized 3DGS")
    plt.show()

    os.makedirs(save_dir, exist_ok=True)
    merged_gaussians.save_ply(os.path.join(save_dir, "point_cloud/iteration_2000/point_cloud.ply"))

    # [MODIFIED] 保存历史数据时，只保存 obj1 和 obj2
    np.savez(
        os.path.join(save_dir, "optimization_history_np10.npz"),  # np = 10
        obj1_history=np.array(split_problem.obj1_history),
        obj2_history=np.array(split_problem.obj2_history)
    )

    print("Optimization completed and results saved!")
