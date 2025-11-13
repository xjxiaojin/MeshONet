import torch
from torch.utils.data import DataLoader
from model import DeepONet
from grid_dataset import BoundaryGridDataset
from utils import *

def main():
    # 设置随机种子
    set_seed(12345)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_file = "boundary_grid_dataset.pt"
    full_dataset = torch.load(dataset_file,weights_only=False)
    print(f"Loaded dataset from {dataset_file}")

    # 将前30个文件作为训练集
    train_idx = list(range(30))  # 前30个文件的索引
    test_idx = [31, 32, 33, 34, 35, 36]  # 文件35, 36, 37的索引 (Python的索引是从0开始)#Nc用18

    # 分配训练集和测试集
    train_data = torch.utils.data.Subset(full_dataset, train_idx)
    test_data = torch.utils.data.Subset(full_dataset, test_idx)

    # 创建 DataLoader
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=6, pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)

    # 定义模型参数
    branch_layers = [50, 64, 128, 64, 50]
    trunk_layers =  [12, 50, 50, 50, 50]

    # 初始化 DeepONet 模型
    model = DeepONet(branch_layers, trunk_layers, device)
    print(f"Number of trainable parameters: {model.count_parameters()}")
    model.to(device)

    # 设置优化器
    model.set_optimizer(learning_rate=1e-3, decay_steps=1000, decay_rate=0.9)

    # 定义模型保存路径
    model_save_path = f"model/model_best_model.pth"

    # 训练模型，设置分组大小，并将最佳模型保存到指定路径
    model.train_model(train_loader, nIter=15000,  model_save_path=model_save_path)

    # 加载并测试最佳模型
    state_dict = torch.load(model_save_path)
    model.load_state_dict(state_dict)

    model.eval()
    predicted_grid = []
    true_grid = []

    # 将 test_idx 列表中的每个元素加1 (转换为人类可读的索引)
    test_idx = [i + 1 for i in test_idx]
    with torch.no_grad():
        total_mse_error = 0.0
        num_batches = 0
        predicted_grid = []
        true_grid = []

        for batch_idx, batch in enumerate(test_loader):
            branch_input_full = batch["boundary_points"]  # 获取上边界点坐标 (1, num_points, 2)
            trunk_input = batch["xi_eta_points"]  # (1, num_grid_points, 2)
            true_coordinates = batch["labels"]  # (1, num_grid_points, 2)

            # 获取当前的测试文件索引，用于区分不同的处理方式
            test_idx_value = test_idx[batch_idx]

            sampled_indices = np.linspace(0, branch_input_full.size(1) - 1, 50, dtype=int)
            branch_input = branch_input_full[:, sampled_indices, :]  # 采样后的 branch_input
           

            # 将所有边界点、内点、标签移动到设备
            branch_input = branch_input.to(device)  # (1, 50, 2) 或 (1, num_points, 2)
            trunk_input = trunk_input.to(device)  # (1, num_grid_points, 2)
            true_coordinates = true_coordinates.to(device)  # (1, num_grid_points, 2)
            branch_input = branch_input.squeeze(0)
            trunk_input = trunk_input.squeeze(0)
          
            pred = model.predict(branch_input, trunk_input)

            pred = pred.unsqueeze(0)
            lower_boundary_mask = get_lower_boundary_mask(trunk_input)
            left_boundary_mask = get_left_boundary_mask(trunk_input)
            right_boundary_mask = get_right_boundary_mask(trunk_input)
            upper_boundary_mask = get_upper_boundary_mask(trunk_input)

            pred[0][left_boundary_mask] = true_coordinates[0][left_boundary_mask]
            pred[0][right_boundary_mask] = true_coordinates[0][right_boundary_mask]
            pred[0][lower_boundary_mask] = true_coordinates[0][lower_boundary_mask]
            pred[0][upper_boundary_mask] = true_coordinates[0][upper_boundary_mask]

            # 将预测值和真实值扩展到总的列表中
            predicted_grid.append(pred.cpu().numpy())
            true_grid.append(true_coordinates.cpu().numpy())

            # 存为 .x 文件
            save_grid_as_x_format(pred.cpu().numpy(),
                                  f"data/test{test_idx[batch_idx]}.x")

            # 计算 MSE
            mse_error = model.loss_data(pred, true_coordinates)
            total_mse_error += mse_error.item()
            num_batches += 1

        mean_mse_error = total_mse_error / num_batches
        print(f'Mean of MSE error for all points: {mean_mse_error:.16e}')

        # 调用函数可视化
        visualize_results(predicted_grid, true_grid, test_idx)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
