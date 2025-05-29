import torch
import torch.nn as nn
import numpy as np

class MLPGenerator(nn.Module):
    def __init__(self, patch_height, patch_width, n_tile_types=10, latent_dim=32, hidden_dim=256):
        super(MLPGenerator, self).__init__()
        
        # Initialize parameters
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.n_tile_types = n_tile_types
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Calculate output size
        output_size = patch_height * patch_width * n_tile_types
        
        # Define the network architecture
        self.main = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 4, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        # Flatten input 
        if len(z.shape) > 2:
            z = z.view(z.size(0), -1)
        
        # Pass through network
        output = self.main(z)
        
        # Reshape to (batch_size, height, width, channels) for one-hot format
        output = output.view(-1, self.patch_height, self.patch_width, self.n_tile_types)
        
        return output
    
    def generate_patch(self, batch_size=1, device='cpu'):
        # Generate a single patch in one-hot encoding
        self.eval()
        with torch.no_grad():
            z = torch.randn(batch_size, self.latent_dim, device=device)
            generated_patch = self.forward(z)
        return generated_patch
    
    def generate_symbolic_patch(self, processor, batch_size=1, device='cpu'):
        # Generate a symbolic patch using one-hot encoding
        generated_patch = self.generate_patch(batch_size, device)
        
        # Convert from tensor to numpy
        patch_np = generated_patch.squeeze(0).cpu().numpy()
        
        # Convert to symbolic using backward mapping
        symbolic_patch = processor.backward_mapping_onehot(patch_np)
        
        if isinstance(symbolic_patch, tuple):
            symbolic_patch = symbolic_patch[1]
        
        return symbolic_patch
    
    def generate_whole_level(self, num_patches_width, processor, num_patches_height=1, device='cpu'):
        # Generate a complete level using one-hot encoding
        self.eval()
        symbolic_level = []
        
        print(f"Generating level: {num_patches_height} x {num_patches_width} patches using MLP")
        
        with torch.no_grad():
            for h in range(num_patches_height):
                row_patches = []
                for w in range(num_patches_width):
                    # Generate a random noise vector
                    z = torch.randn(1, self.latent_dim, device=device)
                    
                    # Generate a patch
                    generated_patch = self.forward(z)
                    
                    # Convert from tensor to numpy (MLP output is already in H,W,C format)
                    patch_np = generated_patch.squeeze(0).cpu().numpy()
                    
                    # Convert to symbolic
                    symbolic_patch = processor.backward_mapping_onehot(patch_np)
                    
                    if isinstance(symbolic_patch, tuple):
                        symbolic_patch = symbolic_patch[1]
                    
                    row_patches.append(symbolic_patch)
                
                # Combine patches horizontally for each row
                for i in range(len(row_patches[0])):
                    symbolic_row = ''.join([patch[i] for patch in row_patches])
                    symbolic_level.append(symbolic_row)
        
        return symbolic_level