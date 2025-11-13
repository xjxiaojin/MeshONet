import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import *

class MLP(nn.Module):
    def __init__(self, layers, activation=nn.Tanh(), final_activation=None):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = activation
        self.final_activation = final_activation  # 可选的最后一层激活函数

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        if self.final_activation:
            x = self.final_activation(x)  # 在最后一层加上激活函数
        return x


class LiftLayer(nn.Module):
    def __init__(self, input_dim, poly_order=2, trig=True):
        super(LiftLayer, self).__init__()
        self.input_dim = input_dim
        self.poly_order = poly_order
        self.trig = trig

    def forward(self, x):
        poly_features = [x]
        for i in range(2, self.poly_order + 1):
            poly_features.append(x ** i)

        if self.trig:
            trig_features = [torch.sin(x), torch.cos(x)]
            poly_features.extend(trig_features)

        return torch.cat(poly_features, dim=-1)


class DeepONet(nn.Module):
    def __init__(self, branch_layers, trunk_layers, device):
        super(DeepONet, self).__init__()
        self.device = device
        self.LiftLayer = LiftLayer(2, poly_order=4, trig=True)

        # Branch networks (Branch-x and Branch-y)
        self.branch_x = MLP(branch_layers, activation=nn.Tanh())
        self.branch_y = MLP(branch_layers, activation=nn.Tanh())

        # Trunk network for computational grid points (Nx10201, 2)，并在最后一层加上激活函数
        self.trunk = MLP(trunk_layers, activation=nn.Tanh(), final_activation=nn.Tanh())

        # 偏置参数
        self.bias_x = nn.Parameter(torch.zeros(1))
        self.bias_y = nn.Parameter(torch.zeros(1))

    def forward(self, branch_input, trunk_input):
        # 使用采样后的上边界点的 x 和 y 坐标分别作为 branch-x 和 branch-y 的输入
        branch_input_x = branch_input[:, 0].reshape(1, -1)  # 横坐标
        branch_input_y = branch_input[:, 1].reshape(1, -1)  # 纵坐标

        # 分别通过 branch-x 和 branch-y 网络
        branch_x_output = self.branch_x(branch_input_x)
        branch_y_output = self.branch_y(branch_input_y)

        # Trunk 网络
        trunk_input = self.LiftLayer(trunk_input)
        trunk_output = self.trunk(trunk_input)

        # Branch 和 Trunk 网络输出的点积并加上偏置
        x_output = torch.sum(branch_x_output * trunk_output, dim=-1, keepdim=True) + self.bias_x
        y_output = torch.sum(branch_y_output * trunk_output, dim=-1, keepdim=True) + self.bias_y

        output = torch.cat([x_output, y_output], dim=-1)
        return output

    def loss(self, pred_interior, pred_boundary, true_interior, true_boundary):
        # 计算内部点和边界点之间的损失
        interior_loss = nn.functional.mse_loss(pred_interior, true_interior)
        boundary_loss = nn.functional.mse_loss(pred_boundary, true_boundary)

        # 综合损失
        total_loss = 900 * interior_loss + 1000 * boundary_loss 
        
        return total_loss

    def loss_data(self, pred, true_coordinates):
        return nn.functional.mse_loss(pred, true_coordinates)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_optimizer(self, learning_rate=1e-3, decay_steps=1000, decay_rate=0.75):
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_steps, gamma=decay_rate)

    def train_model(self, dataloader, nIter=1000,  model_save_path="best_model.pth"):
        self.train()
        best_loss = float('inf')
        best_epoch = 0

        # 开始训练
        for it in tqdm(range(nIter), desc="Training Epochs"):
            epoch_loss = 0.0  # 记录每个 epoch 的损失

            for batch in dataloader:
                computational_points = batch["xi_eta_points"]  # 计算域中的所有点
                physical_points = batch["labels"]  # 物理域中的所有点 (真实值)
                upper_boundary_points = batch["boundary_points"]  # (N, num_upper_points, 2) 上边界点

                # 将输入数据移动到设备 (cuda 或 cpu)
                computational_points = computational_points.to(self.device)
                physical_points = physical_points.to(self.device)
                upper_boundary_points = upper_boundary_points.to(self.device)  # 真实上边界点

                N = computational_points.shape[0]

                for i in range(N):
                    # 对上边界点进行均匀采样，保留 50 个点用于 branch_input
                    sampled_indices = torch.linspace(0, upper_boundary_points.size(1) - 1, 50).long()  # 均匀采样 50 个点
                    sampled_upper_boundary_points = upper_boundary_points[i][sampled_indices]  # (50, 2)

                    # 提取其他方向的边界点
                    lower_boundary_mask = get_lower_boundary_mask(computational_points[i])
                    left_boundary_mask = get_left_boundary_mask(computational_points[i])
                    right_boundary_mask = get_right_boundary_mask(computational_points[i])
                    upper_boundary_mask = get_upper_boundary_mask(computational_points[i])

                    lower_boundary_points = computational_points[i][lower_boundary_mask]
                    left_boundary_points = computational_points[i][left_boundary_mask]
                    right_boundary_points = computational_points[i][right_boundary_mask]
                    all_upper_boundary_points = computational_points[i][upper_boundary_mask]  # 使用所有的上边界点

                    # 对应的真实物理边界点
                    true_lower_boundary_points = physical_points[i][lower_boundary_mask]
                    true_left_boundary_points = physical_points[i][left_boundary_mask]
                    true_right_boundary_points = physical_points[i][right_boundary_mask]
                    true_all_upper_boundary_points = physical_points[i][upper_boundary_mask]  # 所有的上边界点对应的真实值

                    # 拼接所有边界点（使用所有上边界点，而不是采样后的）
                    full_boundary_points = torch.cat(
                        [all_upper_boundary_points, lower_boundary_points, left_boundary_points, right_boundary_points],
                        dim=0)
                    true_full_boundary_points = torch.cat(
                        [true_all_upper_boundary_points, true_lower_boundary_points,
                        true_left_boundary_points, true_right_boundary_points], dim=0)

                    # 分离内点和边界点
                    boundary_mask = upper_boundary_mask | lower_boundary_mask | left_boundary_mask | right_boundary_mask
                    interior_mask = ~boundary_mask

                    interior_points = computational_points[i][interior_mask]
                    true_interior_points = physical_points[i][interior_mask]



                    # 模型预测，branch_input 使用均匀采样后的上边界点，边界点仍使用完整的边界点集
                    pred_interior_points = self.forward(sampled_upper_boundary_points, interior_points)  # 对随机采样的内点进行预测
                    pred_boundary_points = self.forward(sampled_upper_boundary_points, full_boundary_points)  # 对完整边界点集进行预测

                    # 计算损失
                    total_loss = self.loss(
                        pred_interior_points,
                        pred_boundary_points,
                        true_interior_points,
                        true_full_boundary_points
                    )
                    epoch_loss += total_loss.item()

                    # 反向传播和优化
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

            # 在所有批次结束后计算当前 epoch 的平均损失
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{it + 1}/{nIter}], Current Loss: {avg_epoch_loss:.6f}")

            # 保存最佳模型
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_epoch = it + 1
                torch.save(self.state_dict(), model_save_path)

            # 更新 scheduler
            self.scheduler.step()

        print(f"Best model saved with loss {best_loss:.6f} at epoch {best_epoch}")

    def predict(self, branch_input, trunk_input):
        self.eval()
        with torch.no_grad():
            return self.forward(branch_input, trunk_input)

