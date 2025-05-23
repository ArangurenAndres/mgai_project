import torch
import torch.nn as nn

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim, n_tile_types, patch_height=14, patch_width=28):
        super(DCGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.n_tile_types = n_tile_types
        self.patch_height = patch_height
        self.patch_width = patch_width
        
        # Starting with input size (latent_dim, 1, 1)
        # We'll use a first layer to get to 4×7, then double twice to reach 14×28
        self.main = nn.Sequential(
            # Input: (latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=(4, 4), stride=1, padding=0, bias=False),
            # Output: (256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Input: (256, 4, 4)
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(1, 2), padding=1, bias=False),
            # Output: (128, 4, 7)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Input: (128, 4, 7)
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            # Output: (64, 7, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Input: (64, 7, 14)
            nn.ConvTranspose2d(64, n_tile_types, kernel_size=(4, 4), stride=2, padding=1, bias=False),
            # Output: (n_tile_types, 14, 28)
            nn.Sigmoid()
        )

    def forward(self, z):
        # Ensure z is in the right shape for ConvTranspose2d
        if len(z.shape) == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
        return self.main(z)  # Output shape: (batch_size, n_tile_types, 14, 28)


class DCGANDiscriminator(nn.Module):
    def __init__(self, n_tile_types, patch_height=14, patch_width=28):
        super(DCGANDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: (n_tile_types, 14, 28)
            nn.Conv2d(n_tile_types, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # Output: (64, 14, 28)
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),  # Halve dimensions to (64, 7, 14)
            
            # Input: (64, 7, 14)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # Output: (128, 7, 14)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),  # Halve dimensions to (128, 3, 7)
            
            # Input: (128, 3, 7)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=False),
            # Output: (256, 1, 5)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final classification layer
            nn.Conv2d(256, 1, kernel_size=(1, 5), stride=1, padding=0, bias=False),
            # Output: (1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).squeeze()