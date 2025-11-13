import os
import random
import numpy as np
import torch
from matplotlib import pyplot as plt
def get_upper_boundary_mask(computational_points):
    eta = computational_points[:, 1]
    eta_max = eta.max()
    return eta == eta_max

def get_lower_boundary_mask(computational_points):
    eta = computational_points[:, 1]
    eta_min = eta.min()
    return eta == eta_min

def get_left_boundary_mask(computational_points):
    xi = computational_points[:, 0]
    xi_min = xi.min()
    return xi == xi_min
def get_right_boundary_mask(computational_points):
    xi = computational_points[:, 0]
    xi_max = xi.max()
    return xi == xi_max


def get_corner_mask(computational_points, grid_size=101, distance=4):
    """
    提取四个角及其附近的多个点，横纵坐标差在指定距离内的点都认为是角点。
    distance: 提取角点区域的范围。例如，distance=4 表示距离角点4以内的点都算作角点。
    grid_size: 网格的边长（假设网格是方形的）。
    """
    # 创建空的掩码
    corner_mask = torch.zeros(computational_points.shape[0], dtype=torch.bool)

    # 获取左下角（0,0）、左上角（0,grid_size-1）、右下角（grid_size-1,0）、右上角（grid_size-1, grid_size-1）
    for i in range(grid_size):
        for j in range(grid_size):
            # 左下角区域 (0, 0)
            if i <= distance and j <= distance:
                corner_mask[i * grid_size + j] = True
            # 左上角区域 (0, grid_size-1)
            elif i <= distance and j >= grid_size - 1 - distance:
                corner_mask[i * grid_size + j] = True
            # 右下角区域 (grid_size-1, 0)
            elif i >= grid_size - 1 - distance and j <= distance:
                corner_mask[i * grid_size + j] = True
            # 右上角区域 (grid_size-1, grid_size-1)
            elif i >= grid_size - 1 - distance and j >= grid_size - 1 - distance:
                corner_mask[i * grid_size + j] = True

    return corner_mask




def visualize_results(predicted_grid, true_grid, test_idx):
    """
    可视化并保存测试集中每个样本的真实网格与预测网格，分别用子图展示，并显示样本的 test_idx
    :param predicted_grid: 预测的网格结果 (N, 10201, 2)
    :param true_grid: 真实的网格结果 (N, 10201, 2)
    :param fold: 当前折的编号
    :param test_idx: 当前折中每个样本的测试集索引列表
    """
    num_samples = len(predicted_grid)  # 测试样本的数量
    # 设置图像布局，2列，行数为样本数
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))

    if num_samples == 1:  # 只有一个测试样本时，axes 不是二维数组
        axes = np.array([axes])

    # 遍历每个样本，分别绘制真实网格和预测网格
    for i in range(num_samples):
        pred_sample = predicted_grid[i].reshape(-1, 2)  # 展平为(10201, 2)形式
        true_sample = true_grid[i].reshape(-1, 2)  # 展平为(10201, 2)形式

        # 真实网格的可视化
        axes[i][0].scatter(true_sample[:, 0], true_sample[:, 1], c='blue', s=0.3, label='True Grid')
        axes[i][0].set_title(f'True Grid - Sample {test_idx[i]}')
        axes[i][0].set_xlabel('x')
        axes[i][0].set_ylabel('y')
        axes[i][0].legend()

        # 预测网格的可视化
        axes[i][1].scatter(pred_sample[:, 0], pred_sample[:, 1], c='red', s=0.3, label='Predicted Grid')
        axes[i][1].set_title(f'Predicted Grid - Sample {test_idx[i]}')
        axes[i][1].set_xlabel('x')
        axes[i][1].set_ylabel('y')
        axes[i][1].legend()

    # 调整布局，防止重叠
    plt.tight_layout()

    # 定义文件夹路径
    folder_path = 'figure'

    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 保存图像到指定文件夹，文件名中加入 fold 和 test_idx
    idx_str = '_'.join(map(str, test_idx))  # 将 test_idx 列表转换为字符串
    plt.savefig(f'{folder_path}/grid_visualization_samples_{idx_str}.png')
    plt.close()


def extract_boundary_points(pred):
    """
    从网格点的预测值中提取边界点的坐标。

    参数:
    pred: PyTorch 张量，形状为 (N, 10201, 2)，表示 N 个样本，每个样本的网格为 101 x 101 的点坐标。

    返回:
    boundary_points: PyTorch 张量，形状为 (N, 边界点数量, 2)，包含提取出的边界点的坐标。
    """
    N, num_points, _ = pred.shape
    n = int(torch.sqrt(torch.tensor(num_points, dtype=torch.float32)).item())  # 计算网格的边长 n

    # 用于存放所有样本的边界点
    all_boundary_points = []

    for i in range(N):
        # 提取每个样本的预测网格
        sample_pred = pred[i]

        # 提取下边界（第一行）
        boundary_points = sample_pred[:n]

        # 提取上边界（最后一行）
        boundary_points = torch.cat([boundary_points, sample_pred[-n:]], dim=0)

        # 提取左边界（每一行的第一个点）
        left_boundary = torch.stack([sample_pred[j * n] for j in range(n)], dim=0)
        boundary_points = torch.cat([boundary_points, left_boundary], dim=0)

        # 提取右边界（每一行的最后一个点）
        right_boundary = torch.stack([sample_pred[(j + 1) * n - 1] for j in range(n)], dim=0)
        boundary_points = torch.cat([boundary_points, right_boundary], dim=0)

        # 将每个样本的边界点加入总的列表中
        all_boundary_points.append(boundary_points)

    # 将边界点列表转换为张量，形状为 (N, 边界点数量, 2)
    return torch.stack(all_boundary_points)


def replace_boundary_points(pred, real_boundary):
    """
    将提取出来的网格边界点替换为真实的边界点坐标。

    参数:
    pred: PyTorch 张量，形状为 (N, 10201, 2)，表示 N 个 101 x 101 网格的预测点坐标。
    real_boundary: PyTorch 张量，形状为 (N, m, 2)，包含每个样本真实的边界点坐标。

    返回:
    pred: PyTorch 张量，形状为 (N, 10201, 2)，替换后的预测点坐标。
    """
    N, num_points, _ = pred.shape
    n = int(torch.sqrt(torch.tensor(num_points, dtype=torch.float32)).item())  # 假设是 n x n 的网格

    # 遍历每个样本，分别替换边界点
    for i in range(N):
        sample_pred = pred[i]
        sample_real_boundary = real_boundary[i]

        # 1. 替换下边界
        sample_pred[:n] = sample_real_boundary[:n]

        # 2. 替换上边界
        sample_pred[-n:] = sample_real_boundary[n:2 * n]

        # 3. 替换左边界
        for j in range(n):
            sample_pred[j * n] = sample_real_boundary[2 * n + j]

        # 4. 替换右边界
        for j in range(n):
            sample_pred[(j + 1) * n - 1] = sample_real_boundary[3 * n + j]

    return pred


def save_grid_as_x_format(grid, filename_prefix):
    """
    将预测网格保存为 .x 格式。网格可以是 101x101 或 50x50。
    第一行为 1，第二行为对应的网格大小，之后的行先保存所有 x 坐标，再保存所有 y 坐标，
    每个数格式为 0.0000000000000000，每行四个数之间有四个空格。

    :param grid: 预测的网格 (N, num_points, 2)，其中 num_points 可以是 10201 (101x101) 或 2500 (50x50)
    :param filename_prefix: 文件名前缀，每个样本会生成一个单独的文件
    """
    N, num_points, _ = grid.shape  # 获取样本数和点数

    if num_points == 10201:
        grid_size = (101, 101)
    elif num_points == 2500:
        grid_size = (50, 50)
    else:
        raise ValueError("Unsupported grid size. Only 101x101 or 50x50 grids are supported.")

    for sample_idx in range(N):
        filename = f"{filename_prefix}_sample_{sample_idx + 1}.x"  # 为每个样本生成不同的文件
        with open(filename, 'w') as f:
            f.write("1\n")  # 第一行写 1
            f.write(f"{grid_size[0]} {grid_size[1]}\n")  # 第二行写网格大小

            # 提取当前样本的坐标
            sample_grid = grid[sample_idx]

            # 将所有的 x 坐标写入文件，每行 4 个值，使用 4 个空格分隔
            x_coords = sample_grid[:, 0]  # 提取 x 坐标
            for i in range(0, len(x_coords), 4):
                f.write("    ".join(f"{x:.16f}" for x in x_coords[i:i + 4]) + "\n")

            # 将所有的 y 坐标写入文件，每行 4 个值，使用 4 个空格分隔
            y_coords = sample_grid[:, 1]  # 提取 y 坐标
            for i in range(0, len(y_coords), 4):
                f.write("    ".join(f"{y:.16f}" for y in y_coords[i:i + 4]) + "\n")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # 确保使用确定性算法


# 定义 worker_init_fn 来确保每个线程的随机性是固定的
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def generate_grid(x_points=101, y_points=101):
    x = np.linspace(0, 1, x_points)
    y = np.linspace(0, 1, y_points)

    # 使用 meshgrid 生成二维网格
    X, Y = np.meshgrid(x, y)

    # 将 X 和 Y 展开成一维数组
    x_grid = X.flatten()
    y_grid = Y.flatten()

    # 将 x 和 y 坐标组合成 (N, 2) 形状的数组
    grid_points = np.vstack((x_grid, y_grid)).T

    return grid_points, X, Y

def find_index_in_grid(point, grid, atol=1e-6):
    grid_shape = int(np.sqrt(len(grid)))  # 假设计算域是方形的

    x_coords = np.unique(grid[:, 0])  # 计算域中唯一的x坐标
    y_coords = np.unique(grid[:, 1])  # 计算域中唯一的y坐标

    # 找到点对应的x, y坐标在网格中的索引
    row_indices = np.where(np.isclose(y_coords, point[1], atol=atol))[0]
    col_indices = np.where(np.isclose(x_coords, point[0], atol=atol))[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        raise ValueError(f"Point {point} not found in grid.")

    row = row_indices[0]
    col = col_indices[0]

    return row, col

def map_to_physical_grid(point_index, physical_grid, grid_shape):
    """
    将计算域坐标的索引 (row, col) 直接映射到物理域坐标。
    """
    row, col = point_index
    index = row * grid_shape[1] + col  # grid_shape[1] 是每行的点数

    # 检查 index 是否在有效范围内
    if index >= len(physical_grid):
        raise IndexError(f"Index {index} out of bounds for physical grid with length {len(physical_grid)}.")

    return physical_grid[index]


def map_corners_to_physical(corner_points, physical_grid, grid_shape):
    """
    将角点附近的所有计算域索引映射到物理域坐标，并存储为一个张量。
    """
    physical_coords = []

    for point in corner_points:
        # 直接将索引映射到物理域坐标
        physical_coord = map_to_physical_grid(point, physical_grid, grid_shape)
        physical_coords.append(physical_coord)

    # 将所有物理域坐标合并为一个张量
    return torch.tensor(physical_coords)


import torch

def get_corner_mask(computational_points, grid_size=101, distance=4):
    """
    提取左上角和右上角及其附近的多个点，横纵坐标差在指定距离内的点都认为是角点。
    distance: 提取角点区域的范围。例如，distance=4 表示距离角点4以内的点都算作角点。
    grid_size: 网格的边长（假设网格是方形的）。
    """
    # 创建空的掩码
    corner_mask = torch.zeros(computational_points.shape[0], dtype=torch.bool)

    # 获取左上角和右上角
    for i in range(grid_size):
        for j in range(grid_size):
            # 左上角区域 (grid_size-1, 0)
            if i >= grid_size - 1 - distance and j <= distance:
                corner_mask[i * grid_size + j] = True
            # 右上角区域 (grid_size-1, grid_size-1)
            elif i >= grid_size - 1 - distance and j >= grid_size - 1 - distance:
                corner_mask[i * grid_size + j] = True

    return corner_mask


