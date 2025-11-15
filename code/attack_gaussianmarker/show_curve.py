import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. 加载并准备你的两条曲线数据 ---
data = np.load("output/room_attack/merged_gaussians/optimization_history.npz")
a = data['obj1_history']
b = data['obj2_history']
c = data['obj3_history']

# 根据你的论文，a是obj1_history(视觉质量损失), b是obj2_history(特征标准差)
# 我们将y_data1和y_data2的命名与F1, F2对应起来，让代码更清晰
# F1: Visual Quality Loss (from a)
y_F1_raw = 10 * a[2] - 2
# F2: Feature STD (from b)
y_F2_raw = 2 * b[2] - 2


# --- 2. 分别对两条曲线进行平滑处理 ---
window_size = 10

# 平滑 F1
y_series_F1 = pd.Series(y_F1_raw)
y_F1_smoothed = y_series_F1.rolling(window=window_size, center=True, min_periods=1).mean()

# 平滑 F2
y_series_F2 = pd.Series(y_F2_raw)
y_F2_smoothed = y_series_F2.rolling(window=window_size, center=True, min_periods=1).mean()


# --- 3. 在同一张图中绘制所有曲线（使用更新后的信息） ---
plt.style.use('seaborn-v0_8-ticks')
# 建议使用稍宽的尺寸，更适合展示收敛曲线
plt.figure(figsize=(7, 7))

# --- 绘制 F1: Visual Quality Loss ---
plt.plot(y_F1_raw,
         color='lightcoral',
         linestyle='None',
         marker='.',
         markersize=4,
         label='F1: Visual Quality Loss (Raw)') # [MODIFIED] 更新标签

plt.plot(y_F1_smoothed,
         color='crimson',
         linewidth=2.5,
         label=f'F1: Visual Quality Loss (Smoothed)') # [MODIFIED] 更新标签

# --- 绘制 F2: Feature STD ---
plt.plot(y_F2_raw,
         color='green',
         linestyle='None',
         marker='.',
         markersize=4,
         label='F2: Watermark Destruction Loss (Raw)') # [MODIFIED] 更新标签

plt.plot(y_F2_smoothed,
         color='green',
         linewidth=2.5,
         label=f'F2: Watermark Destruction Loss (Smoothed)') # [MODIFIED] 更新标签


# --- 4. 设置图表的整体信息 ---
plt.title('Convergence Curves for Objective Functions F1 & F2', fontsize=16) # [MODIFIED] 更新标题
plt.xlabel('Generation', fontsize=16) # [MODIFIED] 更新X轴标签
plt.ylabel('Fitness Value (Lower is Better)', fontsize=16) # [MODIFIED] 更新Y轴标签
plt.legend(loc='best', frameon=True, fontsize=13) # 优化图例显示
plt.grid(True, linestyle='--', alpha=0.7)

# [ADDED] 自动调整布局，防止标签重叠或被截断
plt.tight_layout()

# 显示最终的图表
plt.show()