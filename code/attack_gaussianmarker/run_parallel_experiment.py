import numpy as np
import geatpy as ea
import torch
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from torchvision import transforms
from hidden.hidden_images import decoder
import os
from sklearn.cluster import KMeans
import gc  # 引入垃圾回收模块
import random

# --- 环境设置 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
random_seed = 0
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)


# --- 复用您原来的 GS_AttackProblem 类 (无需修改) ---
class GS_AttackProblem(ea.Problem):
    def __init__(self, original_gaussians, target_images, pipe_params, decoder, model_params):
        self.decoder = decoder
        self.original_gaussians = original_gaussians
        self.target_images = target_images
        self.pipe = pipe_params
        self.model_params = model_params
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._register_hook()
        self.color_perturb_scale = 1
        self.control_scale = 10
        name = '3DGS_Attack'
        M = 2
        maxormins = [1, 1]
        num_gaussians = original_gaussians._xyz.shape[0]
        self.num_control_variables = max(1, num_gaussians // self.control_scale)
        Dim = self.num_control_variables + num_gaussians * 3
        varTypes = [0] * Dim
        lb = [0] * self.num_control_variables + [-1] * (num_gaussians * 3)
        ub = [1] * self.num_control_variables + [1] * (num_gaussians * 3)
        lbin = [1] * Dim
        ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):
        genes = pop.Phen
        num_individuals = pop.sizes
        num_gaussians = self.original_gaussians._xyz.shape[0]
        ObjV = []
        num_views = len(self.target_images)

        for i in range(num_individuals):
            control_genes = genes[i, :self.num_control_variables]
            color_genes = genes[i, self.num_control_variables:].reshape(num_gaussians, 3)
            temp_gaussians = self.create_perturbed_gaussians(self.original_gaussians, control_genes, color_genes)

            total_obj1 = 0.0
            for view_idx in range(num_views):
                with torch.no_grad():
                    temp_render = render(self.pipe["viewpoints"][view_idx], temp_gaussians, self.pipe["pipe"],
                                         self.pipe["background"])
                    temp_image = temp_render["render"]
                current_obj1 = self.calculate_quality_loss(temp_image, self.target_images[view_idx])
                total_obj1 += current_obj1

            with torch.no_grad():
                obj2, _ = self.calculate_feature_std(temp_image)

            ObjV.append([total_obj1 / num_views, obj2])
        pop.ObjV = np.array(ObjV)

    def _register_hook(self):
        def hook(module, input, output):
            self.current_features = output.detach()

        target_layer = self.decoder.layers[-2].layers[-3]
        target_layer.register_forward_hook(hook)

    def calculate_feature_std(self, rendered_image):
        with torch.no_grad():
            norm_image = self.normalize(rendered_image.unsqueeze(0))
            _ = self.decoder(norm_image)
            features = self.current_features
            std_per_channel = features.std(dim=(2, 3))
            total_std = std_per_channel.mean()
        return total_std.item(), features

    def create_perturbed_gaussians(self, original_gaussians, mask_genes, color_genes):
        mask_genes = np.repeat(mask_genes, self.control_scale + 1)[:self.original_gaussians._xyz.shape[0]]
        mask = mask_genes > 0.5
        new_gaussians = GaussianModel(self.model_params.sh_degree)
        new_gaussians._xyz = original_gaussians._xyz[mask].clone()
        new_gaussians._features_rest = original_gaussians._features_rest[mask].clone()
        new_gaussians._scaling = original_gaussians._scaling[mask].clone()
        new_gaussians._rotation = original_gaussians._rotation[mask].clone()
        new_gaussians._opacity = original_gaussians._opacity[mask].clone()
        active_indices = np.where(mask)[0]
        original_colors = original_gaussians._features_dc[mask].clone()
        color_perturb = torch.tensor(color_genes[active_indices] * self.color_perturb_scale, dtype=torch.float32,
                                     device="cuda")
        new_gaussians._features_dc = original_colors + color_perturb.reshape(-1, 1, 3)
        return new_gaussians

    def calculate_quality_loss(self, rendered_image, target_image):
        l1 = l1_loss(rendered_image, target_image).item()
        ssim_loss = 1 - ssim(rendered_image, target_image).item()
        return 0.7 * l1 + 0.3 * ssim_loss


# --- 辅助函数 ---
def split_gaussians_by_kmeans(original_gaussians, k, model_params):
    xyz = original_gaussians.get_xyz.cpu().detach().numpy()
    kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
    cluster_labels = kmeans.fit_predict(xyz)
    sub_gaussians_list = []
    for cluster_idx in range(k):
        indices = np.where(cluster_labels == cluster_idx)[0]
        if len(indices) == 0: continue
        sub_gaussians = GaussianModel(model_params.sh_degree)
        sub_gaussians._xyz = original_gaussians._xyz[indices].clone()
        sub_gaussians._features_dc = original_gaussians._features_dc[indices].clone()
        sub_gaussians._features_rest = original_gaussians._features_rest[indices].clone()
        sub_gaussians._scaling = original_gaussians._scaling[indices].clone()
        sub_gaussians._rotation = original_gaussians._rotation[indices].clone()
        sub_gaussians._opacity = original_gaussians._opacity[indices].clone()
        sub_gaussians_list.append(sub_gaussians)
    return sub_gaussians_list


def generate_target_images_for_sub_model(sub_gaussians, pipe, num_views):
    sub_target_images = []
    for view_idx in range(num_views):
        with torch.no_grad():
            sub_render = render(pipe["viewpoints"][view_idx], sub_gaussians, pipe["pipe"], pipe["background"])
            sub_target_images.append(sub_render["render"])
    return sub_target_images


if __name__ == "__main__":
    # --- 1. 场景设置 (与您代码相同) ---
    parser = ArgumentParser()
    model_params = ModelParams(parser)
    args = parser.parse_args(['--source_path', 'data/LLFF/flower', '--model_path', 'output/flower_wm', '--sh_degree', '3'])

    model_params.model_path = args.model_path
    model_params = model_params.extract(args)

    if "blender" in args.source_path:
        model_params.resolution = 6
    elif "LLFF" in args.source_path:
        model_params.resolution = 2

    pipe_params = PipelineParams(ArgumentParser())
    pipe_params.convert_SHs_python = False
    pipe_params.compute_cov3D_python = False
    pipe_params.white_background = True

    # --- 2. 加载完整模型 ---
    print("Step 1: Loading the full 3DGS model...")
    torch.cuda.reset_peak_memory_stats()
    original_scene = Scene(model_params, GaussianModel(model_params.sh_degree), load_iteration=20000)
    original_gaussians = original_scene.gaussians

    viewpoint_stack = original_scene.getTrainCameras().copy()
    num_views = 10
    selected_viewpoints = [viewpoint_stack[i] for i in range(min(num_views, len(viewpoint_stack)))]
    background = torch.tensor([1, 1, 1] if pipe_params.white_background else [0, 0, 0], dtype=torch.float32,
                              device="cuda")
    pipe_config = {'viewpoints': selected_viewpoints, 'pipe': pipe_params, 'background': background}

    memory_after_load = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Peak memory after loading full model: {memory_after_load:.2f} GB")

    # --- 3. 拆分模型 ---
    num_splits = 50  # 设定要拆分的份数
    print(f"\nStep 2: Splitting model into {num_splits} parts...")
    sub_gaussians_list = split_gaussians_by_kmeans(original_gaussians, num_splits, model_params)

    # --- 4. 隔离子模型并释放显存 ---
    print("\nStep 3: Isolating one sub-model and releasing memory...")

    # 我们选择第一个子模型进行优化
    target_sub_gaussians = sub_gaussians_list[0]
    print(f"Selected sub-model has {target_sub_gaussians.get_xyz.shape[0]} Gaussians.")

    # 显式删除不再需要的、占用大量显存的变量
    del original_gaussians
    del original_scene
    del sub_gaussians_list

    # 强制Python进行垃圾回收，并清空PyTorch未使用的CUDA缓存
    gc.collect()
    torch.cuda.empty_cache()

    print("Memory released.")

    # --- 5. 为隔离的子模型准备并执行优化 ---
    print("\nStep 4: Preparing and running optimization for the single sub-model...")

    # 为子模型生成对应的目标视图
    target_images = generate_target_images_for_sub_model(target_sub_gaussians, pipe_config, num_views)

    # 重置峰值显存统计，以便精确测量优化过程的开销
    torch.cuda.reset_peak_memory_stats()

    # 创建优化问题
    problem = GS_AttackProblem(target_sub_gaussians, target_images, pipe_config, decoder.cuda().eval(), model_params)
    algorithm = ea.moea_NSGA2_templet(
        problem,
        ea.Population(Encoding='RI', NIND=50),
        MAXGEN=2,  # 可以减少代数以快速测试
        logTras=0
    )

    # 运行优化
    res = ea.optimize(algorithm, verbose=True, drawLog=False, outputMsg=True, drawing=0)

    # 获取优化过程中的峰值显存
    peak_memory_optimization = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # --- 6. 报告结果 ---
    print("\n\n======================================================")
    print("              Single Split Memory Usage Report")
    print("======================================================")
    print(f"Number of Splits during setup: {num_splits}")
    print(f"Gaussians in the optimized sub-model: {target_sub_gaussians.get_xyz.shape[0]}")
    print(f"Peak GPU Memory for optimizing ONE sub-problem: {peak_memory_optimization:.2f} GB")
    print("======================================================")