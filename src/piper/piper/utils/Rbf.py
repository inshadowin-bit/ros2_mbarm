import numpy as np
from scipy.interpolate import Rbf
import collections
from show_traj import show_trajectory3d_comp_web

# 设定窗口大小 N
N = 15  # 减小窗口大小，减少滞后
window = collections.deque(maxlen=N)

def exponential_moving_average(data, alpha=0.3):
    """ 使用指数加权平均（EMA）平滑数据 """
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    return smoothed

def add_point(x, y, z):
    """ 添加新点到窗口 """
    window.append((x, y, z))

def get_smoothed_point():
    """ 使用 RBF 对窗口内的点进行拟合，优化当前点 """
    if len(window) < N:
        return window[-1]  

    # 提取窗口内的 X, Y, Z
    data = np.array(window)
    t = np.arange(len(data))  
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    # 先进行指数加权平均平滑
    # x, y, z = exponential_moving_average(x), exponential_moving_average(y), exponential_moving_average(z)

    # 使用 RBF 拟合
    rbf_x = Rbf(t, x, function='gaussian', smooth=0.5)  
    rbf_y = Rbf(t, y, function='gaussian', smooth=0.5)  
    rbf_z = Rbf(t, z, function='gaussian', smooth=0.5)  

    smoothed_x = rbf_x(len(data) - 1)
    smoothed_y = rbf_y(len(data) - 1)
    smoothed_z = rbf_z(len(data) - 1)

    return smoothed_x, smoothed_y, smoothed_z

if __name__ == '__main__':
    np.random.seed(42)

    # 生成带噪声的 3D 曲线
    t_values = np.linspace(0, 10, 50)
    x_values = np.sin(t_values) + np.random.normal(scale=0.1, size=len(t_values))
    y_values = np.cos(t_values) + np.random.normal(scale=0.1, size=len(t_values))
    z_values = t_values / 10 + np.random.normal(scale=0.05, size=len(t_values))

    smoothed_x_values, smoothed_y_values, smoothed_z_values = [], [], []

    for i in range(len(t_values)):
        add_point(x_values[i], y_values[i], z_values[i])
        sm_x, sm_y, sm_z = get_smoothed_point() #因为每次获取的平滑点都不是从一个窗口中获取的，所以并不具有平滑特性
        smoothed_x_values.append(sm_x)
        smoothed_y_values.append(sm_y)
        smoothed_z_values.append(sm_z)

    # 数据格式转换
    xyz = np.array([x_values, y_values, z_values]).T
    xyz_pro = np.array([smoothed_x_values, smoothed_y_values, smoothed_z_values]).T

    print("xyz.shape:", xyz.shape)
    print("xyz_pro.shape:", xyz_pro.shape)

    show_trajectory3d_comp_web([xyz, xyz_pro], 7026)
