import torch
from torch.utils.data import DataLoader
from model import DeepONet
from grid_dataset import BoundaryGridDataset
from utils import *

def main():
    # Set random seed
    set_seed(12345)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_file = "boundary_grid_dataset.pt"
    full_dataset = torch.load(dataset_file, weights_only=False)
    print(f"Loaded dataset from {dataset_file}")

    # Use the first 30 files as the training set
    train_idx = list(range(30))  # Indices of the first 30 files
    test_idx = [31, 32, 33, 34, 35, 36]  # Indices of files 32-37 (Python index starts from 0)

    # Split dataset into training and testing sets
    train_data = torch.utils.data.Subset(full_dataset, train_idx)
    test_data = torch.utils.data.Subset(full_dataset, test_idx)

    # Create DataLoader
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=6, pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)

    # Define model parameters
    branch_layers = [50, 64, 128, 64, 50]
    trunk_layers = [12, 50, 50, 50, 50]

    # Initialize MeshONet model
    model = MeshONet(branch_layers, trunk_layers, device)
    print(f"Number of trainable parameters: {model.count_parameters()}")
    model.to(device)

    # Set optimizer
    model.set_optimizer(learning_rate=1e-3, decay_steps=1000, decay_rate=0.9)

    # Define model save path
    model_save_path = f"model/model_best_model.pth"

    # Train the model and save the best model to the specified path
    model.train_model(train_loader, nIter=15000, model_save_path=model_save_path)

    # Load and evaluate the best model
    state_dict = torch.load(model_save_path)
    model.load_state_dict(state_dict)

    model.eval()
    predicted_grid = []
    true_grid = []

    # Increment test_idx by 1 for human-readable file indexing
    test_idx = [i + 1 for i in test_idx]
    with torch.no_grad():
        total_mse_error = 0.0
        num_batches = 0
        predicted_grid = []
        true_grid = []

        for batch_idx, batch in enumerate(test_loader):
            branch_input_full = batch["boundary_points"]  # Get top boundary points coordinates (1, num_points, 2)
            trunk_input = batch["xi_eta_points"]  # Grid coordinates in computational domain (1, num_grid_points, 2)
            true_coordinates = batch["labels"]  # True physical coordinates (1, num_grid_points, 2)

            # Get current test file index to handle different files
            test_idx_value = test_idx[batch_idx]

            # Sample 50 points from the branch input
            sampled_indices = np.linspace(0, branch_input_full.size(1) - 1, 50, dtype=int)
            branch_input = branch_input_full[:, sampled_indices, :]  # Sampled branch input

            # Move branch input, trunk input, and labels to device
            branch_input = branch_input.to(device)  # (1, 50, 2) or (1, num_points, 2)
            trunk_input = trunk_input.to(device)  # (1, num_grid_points, 2)
            true_coordinates = true_coordinates.to(device)  # (1, num_grid_points, 2)
            branch_input = branch_input.squeeze(0)
            trunk_input = trunk_input.squeeze(0)

            # Predict grid points
            pred = model.predict(branch_input, trunk_input)
            pred = pred.unsqueeze(0)

            # Apply boundary masks to ensure predicted boundary matches true boundary
            lower_boundary_mask = get_lower_boundary_mask(trunk_input)
            left_boundary_mask = get_left_boundary_mask(trunk_input)
            right_boundary_mask = get_right_boundary_mask(trunk_input)
            upper_boundary_mask = get_upper_boundary_mask(trunk_input)

            pred[0][left_boundary_mask] = true_coordinates[0][left_boundary_mask]
            pred[0][right_boundary_mask] = true_coordinates[0][right_boundary_mask]
            pred[0][lower_boundary_mask] = true_coordinates[0][lower_boundary_mask]
            pred[0][upper_boundary_mask] = true_coordinates[0][upper_boundary_mask]

            # Append predicted and true coordinates to overall lists
            predicted_grid.append(pred.cpu().numpy())
            true_grid.append(true_coordinates.cpu().numpy())

            # Save predicted grid as .x file
            save_grid_as_x_format(pred.cpu().numpy(), f"data/test{test_idx[batch_idx]}.x")

            # Compute MSE error
            mse_error = model.loss_data(pred, true_coordinates)
            total_mse_error += mse_error.item()
            num_batches += 1

        mean_mse_error = total_mse_error / num_batches
        print(f'Mean MSE error for all points: {mean_mse_error:.16e}')

        # Visualize results
        visualize_results(predicted_grid, true_grid, test_idx)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
