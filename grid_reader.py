import time
import os
from numpy import *
import numpy as np
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块

def extractHeadInfo(f, row):  # 坐标开始前面几行的读取
    x_num_list = []
    for i in range(row):
        t_str = f.readline().strip()
        t_str = t_str.replace(' ', ',')
        t_list = t_str.split(',')
        # print(t_list)
        for index in t_list:
            if index != '':
                x_num_list.append(int(index))
        # x_num = int(t_list[0])  # x坐标数量
        # x_num_list.append(x_num)
        # x_num = int(t_list[7])  # y坐标数量
        # x_num_list.append(x_num)
        # print(x_num_list)
    return x_num_list


def readFile(filepath, flg):
    with open(filepath, 'r') as f:
        t_str = f.readline().strip()  # 读第一行
        # print(t_str)
        t_str = t_str.replace(' ', '')  # 取出头部文件有多少行,第一行的数字
        row = int(t_str)  # 转换为整数
        # print(row)
        # 处理头部信息
        x_num_list = extractHeadInfo(f, row)

        x_list = []
        y_list = []
        z_list = []
        all_list = []

        x_row = 0
        # for x in range(x_num_list[0]*x_num_list[1]-1):
        x = x_num_list[0] * x_num_list[1]
        if x % 4 == 0:
            x_row = int(x / 4)
        else:
            x_row = int(x / 4) + 1
        # print(x_row)
        # 先读x坐标
        for i in range(x_row):
            t_str = f.readline().strip()
            if not t_str:
                break
            t_str = t_str.replace('\n', '')
            t_str = t_str.replace(' ', ',')
            templist = t_str.split(',')
            for index in templist:
                if index != '':
                    x_list.append(float(index))
        # 读y坐标
        for i in range(x_row):
            t_str = f.readline().strip()
            if not t_str:
                break
            t_str = t_str.replace('\n', '')
            t_str = t_str.replace(' ', ',')
            templist = t_str.split(',')
            for index in templist:
                if index != '':
                    y_list.append(float(index))
        # 读z坐标
        if flg:
            for i in range(x_row):
                t_str = f.readline().strip()
                if not t_str:
                    break
                t_str = t_str.replace('\n', '')
                t_str = t_str.replace(' ', ',')
                templist = t_str.split(',')
                for index in templist:
                    if index != '':
                        z_list.append(float(index))

        if flg:
            for i in range(len(x_list)):
                temp_list = [(x_list[i]), (y_list[i]), (z_list[i])]
                # temp_list = [(x_list[i]), (y_list[i])]
                all_list.append(temp_list)
        else:
            for i in range(len(x_list)):
                # temp_list=[(x_list[i]),(y_list[i]),(z_list[i])]
                temp_list = [(x_list[i]), (y_list[i])]
                all_list.append(temp_list)
        return all_list, x_list, y_list, z_list, x_num_list




# 生成物理域的边界点和内部点
def get_internal_points(x_list, y_list, x_num_list):
    x_number = x_num_list[0]
    y_number = x_num_list[1]

    interior_points = []

    for i in range(y_number):
        for j in range(x_number):
            if i == 0 or i == y_number - 1 or j == 0 or j == x_number - 1:
                # 如果是边界点，跳过
                continue
            else:
                # 如果是内部点
                interior_points.append((x_list[i * x_number + j], y_list[i * x_number + j]))

    # 将内部点转换为 NumPy 数组
    interior_points_tensor = np.array(interior_points)

    return interior_points_tensor



def get_upper_boundary_points(x_list, y_list, x_num_list):
    """
    提取网格的上边界点坐标。

    参数:
    x_list (list): 所有点的 x 坐标列表。
    y_list (list): 所有点的 y 坐标列表。
    x_num_list (list): 包含 x 和 y 方向上的点数的列表。

    返回:
    upper_boundary_points (list): 上边界点的 (x, y) 坐标列表。
    """
    x_number = x_num_list[0]
    y_number = x_num_list[1]

    upper_boundary_points = []

    # 提取上边界点
    for j in range(x_number):
        # 上边界点对应 y = 1 的点
        upper_boundary_points.append((x_list[(y_number - 1) * x_number + j], y_list[(y_number - 1) * x_number + j]))

    return upper_boundary_points


def get_boundary_points_excluding_top(x_list, y_list, x_num_list):
    """
    提取网格的左边、下边、右边的边界点坐标，不包括上边界。

    参数:
    x_list (list): 所有点的 x 坐标列表。
    y_list (list): 所有点的 y 坐标列表。
    x_num_list (list): 包含 x 和 y 方向上的点数的列表。

    返回:
    left_boundary (list): 左边界的 (x, y) 边界点坐标列表。
    bottom_boundary (list): 下边界的 (x, y) 边界点坐标列表。
    right_boundary (list): 右边界的 (x, y) 边界点坐标列表。
    """
    x_number = x_num_list[0]
    y_number = x_num_list[1]

    left_boundary = []
    bottom_boundary = []
    right_boundary = []

    # 提取左边界点 (对应 x = 0 的点)
    for i in range(y_number):
        left_boundary.append((x_list[i * x_number], y_list[i * x_number]))

    # 提取下边界点 (对应 y = 0 的点)
    for j in range(x_number):
        bottom_boundary.append((x_list[j], y_list[j]))

    # 提取右边界点 (对应 x = 1 的点)
    for i in range(y_number):
        right_boundary.append((x_list[i * x_number + (x_number - 1)], y_list[i * x_number + (x_number - 1)]))

    return left_boundary, bottom_boundary, right_boundary


'''
def main():
    # 假设你要读取的文件路径如下
    filepath = r'D:\data\deeponet-fno-main\pythonProject\6.x'

    # 调用 readFile 函数读取 2D 文件，注意 flg 参数设为 False
    all_list, x_list, y_list, z_list, x_num_list = readFile(filepath, flg=False)

    # 打印读取的 x 和 y 值，确认读取是否成功
    print("X coordinates:", x_list)
    print("Y coordinates:", y_list)
    print("Grid dimensions (x, y):", x_num_list)

    # 获取边界点和内部点张量
    boundary_points_tensor, interior_points_tensor = get_boundary_and_interior_points(x_list, y_list, x_num_list)

    # 打印边界点张量和内部点张量
    print("Boundary points tensor (shape: {}):".format(boundary_points_tensor.shape))
    print(boundary_points_tensor)

    print("Interior points tensor (shape: {}):".format(interior_points_tensor.shape))
    print(interior_points_tensor)

    # 使用 matplotlib 绘制 2D 网格点
    plt.scatter(x_list, y_list, label="All Points", alpha=0.3)

    # 绘制边界点
    plt.scatter(boundary_points_tensor[:, 0], boundary_points_tensor[:, 1], color='red', label="Boundary Points")

    # 绘制内部点
    plt.scatter(interior_points_tensor[:, 0], interior_points_tensor[:, 1], color='blue', label="Interior Points")

    plt.title("2D Grid Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
'''
