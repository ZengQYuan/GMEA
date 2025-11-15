import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os

# 1. 定义疏松布局的高斯球参数
data = {
    'x': [0.0, 1.2, -0.8, 1.5, -1.5, 0.2],
    'y': [0.0, 0.5, 1.2, -1.0, -0.8, -1.5],
    'z': [0.0, 0.8, -0.5, -0.2, 0.6, 1.0],
    'qw': [0.866, 0.707, 0.966, 0.5, 1.0, 0.707],
    'qx': [0.5, 0.0, 0.0, 0.5, 0.0, 0.707],
    'qy': [0.0, 0.707, 0.0, 0.5, 0.0, 0.0],
    'qz': [0.0, 0.0, 0.259, 0.5, 0.0, 0.0],
    's_x': [1.0, 0.9, 0.8, 1.2, 0.7, 0.4],
    's_y': [0.8, 0.9, 0.3, 0.5, 0.7, 1.0],
    's_z': [0.2, 0.3, 1.1, 0.5, 0.7, 0.6],
    'r': [230, 46, 52, 155, 241, 26],
    'g': [126, 204, 152, 89, 196, 188],
    'b': [34, 113, 219, 182, 15, 156],
    'a': [0.6, 0.65, 0.6, 0.55, 0.65, 0.5]
}
df = pd.DataFrame(data)

# 辅助函数：将四元数转换为旋转矩阵 (与之前相同)
def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q
    r11 = 1 - 2 * (qy**2 + qz**2)
    r12 = 2 * (qx * qy - qw * qz)
    r13 = 2 * (qx * qz + qw * qy)
    r21 = 2 * (qx * qy + qw * qz)
    r22 = 1 - 2 * (qx**2 + qz**2)
    r23 = 2 * (qy * qz - qw * qx)
    r31 = 2 * (qx * qz - qw * qy)
    r32 = 2 * (qy * qz + qw * qx)
    r33 = 1 - 2 * (qx**2 + qy**2)
    return np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

# 创建3D图形对象
fig = go.Figure()

# 创建椭球体网格点
u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 40)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

# 循环遍历每个高斯球并添加到fig中 (与之前相同)
for i, row in df.iterrows():
    points = np.stack([row['s_x'] * x_sphere.flatten(), row['s_y'] * y_sphere.flatten(), row['s_z'] * z_sphere.flatten()])
    quat = [row['qw'], row['qx'], row['qy'], row['qz']]
    rot_matrix = quaternion_to_rotation_matrix(quat)
    rotated_points = rot_matrix @ points
    x_final = rotated_points[0, :].reshape(x_sphere.shape) + row['x']
    y_final = rotated_points[1, :].reshape(y_sphere.shape) + row['y']
    z_final = rotated_points[2, :].reshape(z_sphere.shape) + row['z']
    color_str = f'rgb({row["r"]}, {row["g"]}, {row["b"]})'
    fig.add_trace(go.Surface(x=x_final, y=y_final, z=z_final, opacity=row['a'], colorscale=[[0, color_str], [1, color_str]], showscale=False))

# --- 主要修改部分：设置固定的相机视角和布局 ---
camera_eye = dict(x=2.0, y=2.0, z=1.5) # 设置相机位置
fig.update_layout(
    title_text='3D Gaussian Splats - 2D静态图输出',
    scene=dict(
        xaxis_title='X轴', yaxis_title='Y轴', zaxis_title='Z轴',
        xaxis=dict(showbackground=False, showticklabels=False, title=''),
        yaxis=dict(showbackground=False, showticklabels=False, title=''),
        zaxis=dict(showbackground=False, showticklabels=False, title=''),
        aspectratio=dict(x=1, y=1, z=1)
    ),
    margin=dict(l=0, r=0, b=0, t=40),
    # 固定相机视角，确保每次生成的图片都一样
    scene_camera=dict(
        eye=camera_eye,
        up=dict(x=0, y=0, z=1), # Z轴为上方向
        center=dict(x=0, y=0, z=0)
    )
)

# --- 主要修改部分：将图形保存为静态图片 ---
output_filename = "gaussian_splats_2d.png"
fig.write_image(output_filename, width=800, height=600, scale=2) # scale=2 提高分辨率

print(f"图片已成功保存为 '{os.path.abspath(output_filename)}'")