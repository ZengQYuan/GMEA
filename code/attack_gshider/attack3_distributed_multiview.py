import numpy as np
import geatpy as ea
import torch
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams
from decoder import SimpleCNN, WatermarkCNN
from torchvision import transforms
from hidden.hidden_images import decoder
import os
from sklearn.cluster import KMeans
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch.nn as nn
import random

random_seed = 0  # 固定随机种子保证可复现性
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)


class FeatureAdapter(nn.Module):
    """
    一个将 N 通道特征图转换为 3 通道特征图的适配器。
    """
    def __init__(self, in_channels):
        super(FeatureAdapter, self).__init__()
        # 使用 1x1 卷积将 N 通道映射到 3 通道
        self.adapter_layer = nn.Conv2d(in_channels, 3, kernel_size=1)

    def forward(self, x):
        return self.adapter_layer(x)

class GS_AttackProblem(ea.Problem):
    def __init__(self, original_gaussians, target_images, pipe_params, decoder, feature_channels_N=16):
        # 初始化问题参数
        self.decoder = decoder
        self.original_gaussians = original_gaussians
        self.target_images = target_images  # 多个视角的目标图像
        self.pipe = pipe_params
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._register_hook()  # 注册钩子捕获中间特征
        self.num = 0
        self.feature_perturb_scale = 1  # 特征扰动幅度控制
        self.control_scale = 10
        self.feature_channels = feature_channels_N
        self.adapter = FeatureAdapter(in_channels=self.feature_channels).to("cuda")

        self.obj1_history = []  # 记录每代平均目标1
        self.obj2_history = []  # 记录每代平均目标2
        self.best_obj1 = []  # 记录每代最优目标1
        self.best_obj2 = []  # 记录每代最优目标2
        self.obj3_history = []  # 记录每代平均目标3
        self.best_obj3 = []  # 记录每代最优目标3

        # 优化问题定义
        name = '3DGS_Attack'
        M = 3  # 多目标优化（三个目标）
        maxormins = [1, 1, 1]  # 三个目标都需要最小化
        num_gaussians = original_gaussians._xyz.shape[0]  # 高斯球个数
        self.num_control_variables = num_gaussians // self.control_scale  # 每个控制变量管理10个高斯球


        # 染色体编码：每个高斯球4个参数（1个mask + semantic_dim个通道的扰动）
        self.semantic_dim = original_gaussians._semantic_feature.shape[2]
        Dim = self.num_control_variables + num_gaussians * self.semantic_dim
        varTypes = [0] * Dim  # 所有变量均为实数

        # 变量边界设置（0-1连续变量）
        lb = [0] * self.num_control_variables  # mask部分[0,1]
        lb += [-1] * (num_gaussians * self.semantic_dim)  # 颜色扰动[-1,1]
        ub = [1] * self.num_control_variables
        ub += [1] * (num_gaussians * self.semantic_dim)
        lbin = [1] * Dim
        ubin = [1] * Dim

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        # 解码种群获得掩膜矩阵 [NIND x Dim]
        genes = pop.Phen
        num_individuals = pop.sizes
        num_gaussians = self.original_gaussians._xyz.shape[0]

        ObjV = []

        for i in range(num_individuals):
            # 分离控制变量和颜色扰动基因
            control_genes = genes[i, :self.num_control_variables]
            feature_genes = genes[i, self.num_control_variables:].reshape(num_gaussians, self.semantic_dim)

            # 创建扰动后的高斯模型
            temp_gaussians = self.create_perturbed_gaussians(
                self.original_gaussians,
                control_genes,
                feature_genes  # 传递特征基因
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


            # 目标1：渲染质量损失
            with torch.no_grad():
                obj2, feature_map = self.calculate_feature_std(temp_image)  # haha

                active_gaussians = (control_genes > 0.5).sum()
                obj3 = active_gaussians

            ObjV.append([total_obj1, obj2, obj3]) # haha ObjV.append([total_obj1, obj2, obj3])

        print("Generation： ", self.num)
        self.num = self.num + 1

        if self.num % 50 == 0:  # haha
            mean_activation_map = feature_map.squeeze(0).mean(dim=0).cpu().detach().numpy()
            plt.imshow(mean_activation_map, cmap='viridis', vmax=2, vmin=-2)
            plt.show()

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


    def calculate_feature_std(self, rendered_feature_map):
        """计算HiDDeN模型中间特征图的标准差"""
        with torch.no_grad():
            adapted_image = self.adapter(rendered_feature_map.unsqueeze(0))  # [1, N, H, W] -> [1, 3, H, W]

            # 图像预处理（需与HiDDeN训练时一致）
            norm_image = self.normalize(adapted_image)

            # 前向传播获取中间特征
            _ = self.decoder(norm_image)
            features = self.current_features

            # 计算通道间标准差（仿照Dispersion攻击）
            std_per_channel = features.std(dim=(2, 3))  # 计算各通道空间维度的标准差
            total_std = std_per_channel.mean()  # 取通道间平均


        return total_std.item(), features  # haha

    def _register_hook(self):
        """注册前向钩子捕获中间特征图"""

        def hook(module, input, output):
            self.current_features = output.detach()

        # 根据decoder结构选择目标层
        target_layer = self.decoder.layers[-2].layers[-3]
        target_layer.register_forward_hook(hook)

    def create_perturbed_gaussians(self, original_gaussians, mask_genes, feature_genes):
        """创建应用了mask和颜色扰动的高斯模型"""

        # 生成完整的掩膜（每个控制变量重复10次）
        mask_genes = np.repeat(mask_genes, self.control_scale + 1)[:self.original_gaussians._xyz.shape[0]]
        mask = mask_genes > 0.5  # 二值化mask

        new_gaussians = GaussianModel(model_params.sh_degree)

        # 复制选中的高斯球属性
        new_gaussians._xyz = original_gaussians._xyz[mask].clone()
        new_gaussians._scaling = original_gaussians._scaling[mask].clone()
        new_gaussians._rotation = original_gaussians._rotation[mask].clone()
        new_gaussians._opacity = original_gaussians._opacity[mask].clone()

        # 应用颜色扰动
        active_indices = np.where(mask)[0]

        # 读取原始的语义特征
        original_features = original_gaussians._semantic_feature[mask].clone()

        # 颜色扰动 = 扰动基因 * 幅度控制
        feature_perturb = torch.tensor(
            feature_genes[active_indices] * self.feature_perturb_scale,
            dtype=torch.float32,
            device="cuda"
        )

        new_gaussians._semantic_feature = original_features + feature_perturb.reshape(-1, 1, self.semantic_dim)

        return new_gaussians

    def calculate_quality_loss(self, rendered_image, target_image):
        """计算渲染质量损失（多指标融合）"""
        l1 = l1_loss(rendered_image, target_image).item()
        ssim_loss = 1 - ssim(rendered_image, target_image).item()
        return 0.7 * l1 + 0.3 * ssim_loss  # 组合损失


class SplitGS_AttackProblem:
    def __init__(self, original_gaussians,  pipe_params, decoder, num_splits, num_views):
        self.num_splits = num_splits
        self.original_gaussians = original_gaussians
        self.pipe = pipe_params
        self.decoder = decoder
        self.num_views = num_views  # 视角数量

        self.obj1_history = []
        self.obj2_history = []
        self.obj3_history = []

        # 将高斯球均匀分割
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
        def split_gaussians_by_kmeans(original_gaussians, k):
            xyz = original_gaussians.get_xyz.cpu().detach().numpy()
            kmeans = KMeans(n_clusters=k, random_state=random_seed)
            cluster_labels = kmeans.fit_predict(xyz)
            sub_gaussians_list = []
            for cluster_idx in range(k):
                # 找出属于当前聚类的高斯球索引
                indices = np.where(cluster_labels == cluster_idx)[0]

                # 创建子高斯球模型
                sub_gaussians = GaussianModel(model_params.sh_degree)
                sub_gaussians._xyz = original_gaussians._xyz[indices].clone()
                sub_gaussians._semantic_feature = original_gaussians._semantic_feature[indices].clone() # haha
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
            problem = GS_AttackProblem(sub_gaussians_list[split_idx], sub_target_images_list[split_idx], self.pipe, self.decoder)
            algorithm = ea.moea_NSGA2_templet(
                problem,
                ea.Population(Encoding='RI', NIND=20),  # haha
                MAXGEN=2000,  # 每个分割部分的最大进化代数
                logTras=0
            )
            res = ea.optimize(algorithm, verbose=False, drawLog=False, outputMsg=False, drawing=0)

            self.obj1_history.append(problem.obj1_history)
            self.obj2_history.append(problem.obj2_history)
            self.obj3_history.append(problem.obj3_history)

            all_results[split_idx] = {
                "best_masks": res['Vars'],
                "best_gaussians": problem.create_perturbed_gaussians(
                    sub_gaussians_list[split_idx],
                    res['Vars'][0, :problem.num_control_variables],
                    res['Vars'][0, problem.num_control_variables:].reshape(-1, problem.semantic_dim)
                )
            }

        return all_results

    #
    # def split_and_optimize(self):
    #     """将原始3DGS拆解并独立优化"""
    #     all_results = {}
    #
    #     for split_idx, (start, end) in enumerate(self.split_indices):
    #         print(f"Optimizing split {split_idx + 1}/{self.num_splits} ({start}-{end})")
    #
    #         # 提取当前分割部分的高斯球
    #         split_gaussians = GaussianModel(model_params.sh_degree)
    #         split_gaussians._xyz = self.original_gaussians._xyz[start:end].clone()
    #         split_gaussians._features_dc = self.original_gaussians._features_dc[start:end].clone()
    #         split_gaussians._features_rest = self.original_gaussians._features_rest[start:end].clone()
    #         split_gaussians._scaling = self.original_gaussians._scaling[start:end].clone()
    #         split_gaussians._rotation = self.original_gaussians._rotation[start:end].clone()
    #         split_gaussians._opacity = self.original_gaussians._opacity[start:end].clone()
    #
    #         # 配置优化问题
    #         problem = GS_AttackProblem(split_gaussians, self.target_image, self.pipe, self.decoder)
    #
    #         # 算法设置（NSGA-II）
    #         algorithm = ea.moea_NSGA2_templet(
    #             problem,
    #             ea.Population(Encoding='RI', NIND=50),
    #             MAXGEN=2,  # 每个分割部分的最大进化代数
    #             logTras=0
    #         )
    #
    #         # 运行优化
    #         res = ea.optimize(algorithm, verbose=False, drawLog=False, outputMsg=False, drawing=0)
    #
    #         self.obj1_history.append(problem.obj1_history)
    #         self.obj2_history.append(problem.obj2_history)
    #         self.obj3_history.append(problem.obj3_history)
    #         print("haha")
    #         # 记录结果
    #         all_results[split_idx] = {
    #             "best_masks": res['Vars'],
    #             "best_gaussians": problem.create_perturbed_gaussians(
    #                 split_gaussians,
    #                 res['Vars'][0, :problem.num_control_variables],
    #                 res['Vars'][0, problem.num_control_variables:].reshape(-1, 3)
    #             )
    #         }
    #
    #     return all_results


if __name__ == "__main__":
    parser = ArgumentParser()
    model_params = ModelParams(parser)
    args = parser.parse_args([
        '--source_path', 'data/LLFF/room',
        '--model_path', 'output/room_wm',
        '--sh_degree', '3'
    ])
    save_dir = 'output/room_attack/merged_gaussians'

    model_params.model_path, model_params._model_path = args.model_path, args.model_path
    model_params = model_params.extract(args)
    if "blender" in args.source_path:
        model_params.resolution = 1
    elif "LLFF" in args.source_path:
        model_params.resolution = 1

    pipe_params = PipelineParams(ArgumentParser())
    pipe_params.convert_SHs_python = False
    pipe_params.compute_cov3D_python = False
    pipe_params.white_background = True  # 与训练时背景设置一致

    load_iteration = 10000
    imagenet = SimpleCNN().cuda()
    waternet = WatermarkCNN().cuda()
    imagenet_checkpoint_path = os.path.join(args.model_path, "chkpnt" + str(load_iteration) + "_net.pth")
    waternet_checkpoint_path = os.path.join(args.model_path, "chkpnt" + str(load_iteration) + "_waternet.pth")
    print(f"[INFO] Loading ImageNet decoder from: {imagenet_checkpoint_path}")
    imagenet.load_state_dict(torch.load(imagenet_checkpoint_path, weights_only=False))
    imagenet.eval()  # 设置为评估模式
    print(f"[INFO] Loading WatermarkNet decoder from: {waternet_checkpoint_path}")
    waternet.load_state_dict(torch.load(waternet_checkpoint_path, weights_only=False))
    waternet.eval()  # 设置为评估模式


    # 1. 加载原始3DGS模型
    original_scene = Scene(model_params,
                           GaussianModel(model_params.sh_degree),
                           load_iteration=load_iteration)  # 指定最终迭代次数
    original_gaussians = original_scene.gaussians

    # 获取第一个训练视角的相机参数
    viewpoint_stack = original_scene.getTrainCameras().copy()
    num_views = 1  # 使用5个视角 haha
    selected_viewpoints = [viewpoint_stack[i] for i in range(min(num_views, len(viewpoint_stack)))]

    # 背景色设置（需与训练时一致）
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 3. 配置优化问题（将原始问题拆分为10个子问题）
    split_problem = SplitGS_AttackProblem(
        original_gaussians,
        {'viewpoints': selected_viewpoints, 'pipe': pipe_params, 'background': background},
        decoder,
        num_splits=1,  # haha
        num_views=num_views
    )

    # 4. 执行拆分后的独立优化
    all_results = split_problem.split_and_optimize()

    # 5. 合并结果并可视化
    # 先计算所有分割后的高斯球总数
    total_optimized_gaussians = 0
    for split_idx in range(split_problem.num_splits):
        split_result = all_results[split_idx]
        split_gaussians = split_result["best_gaussians"]
        total_optimized_gaussians += split_gaussians._xyz.shape[0]

    # 根据优化后的总数创建合并高斯球模型
    merged_gaussians = GaussianModel(model_params.sh_degree)
    merged_gaussians._xyz = torch.zeros((total_optimized_gaussians, 3), device="cuda")
    merged_gaussians._semantic_feature = torch.zeros((total_optimized_gaussians, 1, original_gaussians._semantic_feature.shape[2]), device="cuda")
    merged_gaussians._scaling = torch.zeros((total_optimized_gaussians, 3), device="cuda")
    merged_gaussians._rotation = torch.zeros((total_optimized_gaussians, 4), device="cuda")
    merged_gaussians._opacity = torch.zeros((total_optimized_gaussians, 1), device="cuda")

    start_idx = 0
    for split_idx in range(split_problem.num_splits):
        split_result = all_results[split_idx]
        split_gaussians = split_result["best_gaussians"]
        split_size = split_gaussians._xyz.shape[0]

        if split_size == 0:
            continue  # 跳过空的分割部分

        end_idx = start_idx + split_size

        merged_gaussians._xyz[start_idx:end_idx] = split_gaussians._xyz
        merged_gaussians._semantic_feature[start_idx:end_idx] = split_gaussians._semantic_feature
        merged_gaussians._scaling[start_idx:end_idx] = split_gaussians._scaling
        merged_gaussians._rotation[start_idx:end_idx] = split_gaussians._rotation
        merged_gaussians._opacity[start_idx:end_idx] = split_gaussians._opacity

        start_idx += split_size

    # 渲染合并后的结果
    with torch.no_grad():
        merged_render = render(viewpoint_stack[0], merged_gaussians, pipe_params, background)
        merged_feature_map = merged_render["render"]
        decoded_rgb_image = imagenet(merged_feature_map)
        decoded_watermark_image = waternet(merged_feature_map)


    plt.imshow(decoded_rgb_image.cpu().detach().numpy().transpose(1, 2, 0))
    plt.title("Merged Optimized 3DGS decoded_rgb_image")
    plt.show()

    plt.imshow(decoded_watermark_image.cpu().detach().numpy().transpose(1, 2, 0))
    plt.title("Merged Optimized 3DGS decoded_watermark_image")
    plt.show()

    # 保存结果
    os.makedirs(save_dir, exist_ok=True)
    merged_gaussians.save_ply(os.path.join(save_dir, "point_cloud/iteration_2000/point_cloud.ply"))

    # 保存历史数据为 npz 格式
    np.savez(
        os.path.join(save_dir, "optimization_history.npz"),
        obj1_history=np.array(split_problem.obj1_history),
        obj2_history=np.array(split_problem.obj2_history),
        obj3_history=np.array(split_problem.obj3_history)
    )

    print("Optimization completed and results saved!")

