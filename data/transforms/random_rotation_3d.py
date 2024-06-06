import random
import torch

class RandomRotation3D:
    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float
    ):
        self.x_min: float = torch.tensor(x_min)
        self.x_max: float = torch.tensor(x_max)
        self.y_min: float = torch.tensor(y_min)
        self.y_max: float = torch.tensor(y_max)
        self.z_min: float = torch.tensor(z_min)
        self.z_max: float = torch.tensor(z_max)
        
    def __call__(
        self,
        tensor
    ):
        x_angle = random.uniform(self.x_min, self.x_max)
        y_angle = random.uniform(self.y_min, self.y_max)
        z_angle = random.uniform(self.z_min, self.z_max)
        
        x_rad = torch.deg2rad(x_angle)
        y_rad = torch.deg2rad(y_angle)
        z_rad = torch.deg2rad(z_angle)
        
        # Rotation matrices for each axis
        x_rot_matrix = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(x_rad), -torch.sin(x_rad)],
            [0, torch.sin(x_rad), torch.cos(x_rad)]
        ])

        y_rot_matrix = torch.tensor([
            [torch.cos(y_rad), 0, torch.sin(y_rad)],
            [0, 1, 0],
            [-torch.sin(y_rad), 0, torch.cos(y_rad)]
        ])

        z_rot_matrix = torch.tensor([
            [torch.cos(z_rad), -torch.sin(z_rad), 0],
            [torch.sin(z_rad), torch.cos(z_rad), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix
        rot_matrix = torch.matmul(torch.matmul(z_rot_matrix, y_rot_matrix), x_rot_matrix)
        
        # Get the tensor dimensions
        x_dim, y_dim, z_dim = tensor.shape
        
        # Create a grid of coordinates
        x_coords = torch.arange(x_dim).float()
        y_coords = torch.arange(y_dim).float()
        z_coords = torch.arange(z_dim).float()

        x_grid, y_grid, z_grid = torch.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
        
        # Flatten the grids and stack them
        original_coords = torch.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()])
        
        # Center the coordinates
        center = torch.tensor([x_dim / 2, y_dim / 2, z_dim / 2])
        centered_coords = original_coords - center[:, None]

        # Apply rotation
        rotated_coords = torch.matmul(rot_matrix, centered_coords).to(torch.float32)

        # Un-center the coordinates
        uncentered_coords = rotated_coords + center[:, None]
        
        # Interpolation to get the rotated tensor values
        rotated_tensor = torch.zeros_like(tensor)
        for i in range(x_dim):
            for j in range(y_dim):
                for k in range(z_dim):
                    xi, yi, zi = uncentered_coords[:, i * y_dim * z_dim + j * z_dim + k]
                    xi, yi, zi = int(round(xi.item())), int(round(yi.item())), int(round(zi.item()))
                    
                    if 0 <= xi < x_dim and 0 <= yi < y_dim and 0 <= zi < z_dim:
                        rotated_tensor[xi, yi, zi] = tensor[i, j, k]
        
        return rotated_tensor

