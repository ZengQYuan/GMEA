import numpy as np
import matplotlib.pyplot as plt
import os


def plot_convergence_comparison(base_dir, population_sizes, time_per_gen_map):
    """
    读取多个.npz文件，并在两个独立的子图上绘制每个目标函数
    在不同种群规模下的收敛曲线，横坐标为累计时间。

    Args:
        base_dir (str): 存储.npz文件的目录路径。
        population_sizes (list): 要绘制的种群规模列表 (例如 [10, 30, 50, 70])。
        time_per_gen_map (dict): 映射种群规模到每代耗时的字典。
    """
    # --- 1. 初始化图表布局 ---
    # 创建一个包含2行1列的图表
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # 为不同曲线定义清晰的颜色
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(population_sizes)))

    # --- 2. 循环读取文件并绘图 ---
    for i, size in enumerate(population_sizes):
        file_name = f"optimization_history_np{size}.npz"
        file_path = os.path.join(base_dir, file_name)

        if not os.path.exists(file_path):
            print(f"警告: 文件 '{file_path}' 不存在，已跳过。")
            continue

        print(f"正在加载: {file_path}")
        data = np.load(file_path)

        # 提取历史数据
        obj1_history = data['obj1_history'].flatten()
        obj2_history = data['obj2_history'].flatten()

        # --- [MODIFIED] 计算时间轴 ---
        # 获取当前种群规模下每代的耗时
        time_per_generation = time_per_gen_map.get(size, 1)  # 如果找不到，默认每代1秒
        # 计算每个generation结束时的累计时间
        num_generations = len(obj1_history)
        # x轴从第一个generation结束时开始，即 t=time_per_generation
        time_axis = np.arange(1, num_generations + 1) * time_per_generation

        # --- 在第一个子图上绘制目标1的收敛曲线 ---
        ax1.plot(time_axis, obj1_history * 5,
                 color=colors[i],
                 label=f'NIND = {size}',
                 linewidth=2.5,
                 alpha=0.9)

        # --- 在第二个子图上绘制目标2的收敛曲线 ---
        ax2.plot(time_axis, obj2_history,
                 color=colors[i],
                 label=f'NIND = {size}',
                 linewidth=2.5,
                 alpha=0.9)

    # --- 3. 美化图表 ---
    # 子图1: 目标1 (Quality Loss)
    ax1.set_title('Comparison of Convergence Curves vs. Time (Quality Loss)', fontsize=16, weight='bold')
    ax1.set_ylabel('Fitness Value (Quality Loss)', fontsize=12)
    ax1.legend(title="Population Size (NIND)", fontsize=14)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 子图2: 目标2 (Feature STD)
    ax2.set_title('Comparison of Convergence Curves vs. Time (Watermark Destruction Loss)', fontsize=16, weight='bold')
    # [MODIFIED] 修改横坐标标签
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Fitness Value (Watermark Destruction Loss)', fontsize=12)
    ax2.legend(title="Population Size (NIND)", fontsize=14)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 统一设置X轴和Y轴刻度标签的大小
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    plt.tight_layout(pad=2.0)

    # --- 4. 保存并显示图表 ---
    output_filename = 'convergence_curves_vs_time_comparison.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至 '{output_filename}'")
    plt.show()


if __name__ == '__main__':
    # --- 用户配置区 ---
    # [ADDED] 定义不同NIND下的每代耗时 (单位: 秒)
    time_per_generation_mapping = {
        70: 5.5,
        50: 4.5,
        30: 2.6,
        10: 0.9
    }

    # !!! 重要: 请将此路径修改为您保存 .npz 文件的实际目录。
    results_directory = 'output/room_attack/merged_gaussians_'  # <--- 修改这里

    # 定义要比较的种群规模
    population_sizes_to_plot = [10, 30, 50, 70]

    # 运行绘图函数
    if not os.path.isdir(results_directory):
        print(f"错误: 目录 '{results_directory}' 不存在。")
        print("请确保路径正确，或者将您的 .npz 文件移动到该路径下。")
    else:
        plot_convergence_comparison(results_directory, population_sizes_to_plot, time_per_generation_mapping)


# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
#
# def plot_convergence_comparison(base_dir, population_sizes):
#     """
#     读取多个.npz文件，并在两个独立的子图上绘制每个目标函数
#     在不同种群规模下的收敛曲线。
#
#     Args:
#         base_dir (str): 存储.npz文件的目录路径。
#         population_sizes (list): 要绘制的种群规模列表 (例如 [10, 30, 50, 70])。
#     """
#     # --- 1. 初始化图表布局 ---
#     # 创建一个包含2行1列的图表，共享x轴
#     plt.style.use('seaborn-v0_8-whitegrid')
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
#
#     # 为不同曲线定义清晰的颜色
#     colors = plt.cm.plasma(np.linspace(0, 0.8, len(population_sizes)))
#
#     # --- 2. 循环读取文件并绘图 ---
#     for i, size in enumerate(population_sizes):
#         file_name = f"optimization_history_np{size}.npz"
#         file_path = os.path.join(base_dir, file_name)
#
#         if not os.path.exists(file_path):
#             print(f"警告: 文件 '{file_path}' 不存在，已跳过。")
#             continue
#
#         print(f"正在加载: {file_path}")
#         data = np.load(file_path)
#
#         # 提取历史数据，并处理可能的 (1, num_generations) 形状
#         obj1_history = data['obj1_history'].flatten()
#         obj2_history = data['obj2_history'].flatten()
#
#         # 创建x轴 (迭代次数)
#         generations = np.arange(len(obj1_history))
#
#         # --- 在第一个子图上绘制目标1的收敛曲线 ---
#         ax1.plot(generations, obj1_history*5,
#                  color=colors[i],
#                  label=f'NIND = {size}',
#                  linewidth=2,
#                  alpha=0.9)
#
#         # --- 在第二个子图上绘制目标2的收敛曲线 ---
#         ax2.plot(generations, obj2_history,
#                  color=colors[i],
#                  label=f'NIND = {size}',
#                  linewidth=2,
#                  alpha=0.9)
#
#     # --- 3. 美化图表 ---
#     # 子图1: 目标1 (Quality Loss)
#     ax1.set_title('Comparison of convergence curves (Quality Loss)', fontsize=16, weight='bold')
#     ax1.set_ylabel('Fitness Value (Quality Loss)', fontsize=12)
#     ax2.set_xlabel('Generation', fontsize=12)
#     ax1.legend(title="Population Size (NIND)", fontsize=18)
#     ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
#
#     # 子图2: 目标2 (Feature STD)
#     ax2.set_title('Comparison of convergence curves (Watermark Destruction Loss)', fontsize=16, weight='bold')
#     ax2.set_xlabel('Generation', fontsize=12)
#     ax2.set_ylabel('Fitness Value (Watermark Destruction Loss)', fontsize=12)
#     ax2.legend(title="Population Size (NIND)", fontsize=18)
#     ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
#
#     plt.tight_layout(pad=2.0)
#
#     # --- 4. 保存并显示图表 ---
#     # output_filename = 'convergence_curves_comparison.png'
#     # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
#     # print(f"\n图表已保存至 '{output_filename}'")
#     plt.show()
#
#
# if __name__ == '__main__':
#     # --- 用户配置区 ---
#     # !!! 重要: 请将此路径修改为您保存 .npz 文件的实际目录。
#     # 根据您的脚本，它可能是 'output/room_attack/merged_gaussians_'
#     results_directory = 'output/room_attack/merged_gaussians_'  # <--- 修改这里
#
#     # 定义要比较的种群规模
#     population_sizes_to_plot = [10, 30, 50, 70]
#
#     # 运行绘图函数
#     if not os.path.isdir(results_directory):
#         print(f"错误: 目录 '{results_directory}' 不存在。")
#         print("请确保路径正确，或者将您的 .npz 文件移动到该路径下。")
#     else:
#         plot_convergence_comparison(results_directory, population_sizes_to_plot)