import random
from utils import *
import numpy as np
from torch.utils.data import Dataset
import torch
from grid_reader import *


class BoundaryGridDataset(Dataset):
    def __init__(self, boundary_inputs, full_xi_eta_points, labels):
        """
        :param boundary_inputs: 边界点坐标的列表，包含上边界的点
        :param full_xi_eta_points: 计算域网格坐标
        :param labels: 物理域的标签（真实坐标）
        """
        self.boundary_inputs = boundary_inputs
        self.full_xi_eta_points = full_xi_eta_points
        self.labels = labels

    def __len__(self):
        """返回数据集的大小"""
        return len(self.boundary_inputs)

    def __getitem__(self, idx):
        """
        获取数据集中的第 idx 个样本，组织形式为 {边界点坐标, 计算域坐标, 物理域标签}
        """
        boundary_points = self.boundary_inputs[idx]  # 获取上边界点坐标
        full_xi_eta_points = self.full_xi_eta_points[idx]  # 获取计算域网格坐标
        labels = self.labels[idx]  # 获取物理域标签（真实坐标）

        return {"boundary_points": boundary_points, "xi_eta_points": full_xi_eta_points, "labels": labels}


def load_datasets(filepaths):
    """加载并处理所有数据集"""
    upper_xy_boundary_points_list = []
    all_lists = []

    for filepath in filepaths:
        all_list, x_list, y_list, z_list, x_num_list = readFile(filepath, flg=False)
        upper_xy_boundary_points = get_upper_boundary_points(x_list, y_list, x_num_list)
        upper_xy_boundary_points_list.append(torch.tensor(upper_xy_boundary_points, dtype=torch.float32))
        all_lists.append(all_list)

    return upper_xy_boundary_points_list, all_lists


def save_dataset(dataset, filename):
    """将数据集保存到文件"""
    torch.save(dataset, filename)
    print(f"数据集已保存至 {filename}")


def main():
    # 设置随机种子
    set_seed(12345)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 准备数据集文件路径
    filepaths = [rf'D:\\AI 4 PDE\\Mesh Grid Generation\\DeepONet网格生成\\data\\{i}new.x' for i in range(1, 44)]

    # 加载所有数据集，提取上边界点和物理域的标签（真实坐标）
    upper_xy_boundary_points_list, all_lists = load_datasets(filepaths)

    # 生成完整的计算域网格，根据不同文件设置不同的大小
    full_xi_eta_points_list = []

    for i in range(1, 44):
        if 38 <= i <= 43:  # 对于文件 38-44new.x 使用 50x50 的网格
            full_xi_eta_points, X, Y = generate_grid(x_points=50, y_points=50)
        else:  # 对于其他文件使用 101x101 的网格
            full_xi_eta_points, X, Y = generate_grid(x_points=101, y_points=101)

        full_xi_eta_points = torch.tensor(full_xi_eta_points, dtype=torch.float32)
        full_xi_eta_points_list.append(full_xi_eta_points)

    # 构建边界点输入，包含上边界
    boundary_inputs = [
        upper_xy_boundary_points_list[i].clone().detach().float()
        for i in range(len(upper_xy_boundary_points_list))
    ]

    # 组织成数据集
    full_dataset = BoundaryGridDataset(
        boundary_inputs=boundary_inputs,
        full_xi_eta_points=full_xi_eta_points_list,
        labels=[torch.tensor(all_lists[i], dtype=torch.float32) for i in range(len(all_lists))]
    )

    # 保存数据集到文件
    save_dataset(full_dataset, "boundary_grid_dataset.pt")


if __name__ == "__main__":
    main()
