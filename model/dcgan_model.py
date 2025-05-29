import torch
import torch.nn as nn

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim, n_tile_types, patch_height=14, patch_width=28, hidden_dim=512):
        super(DCGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.n_tile_types = n_tile_types
        self.patch_height = patch_height
        self.patch_width = patch_width
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(True),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 2, 128 * 7 * 14),  # Increased channels
            nn.BatchNorm1d(128 * 7 * 14),
            nn.ReLU(True)
        )
        
        # Simple upsampling to get exact size
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, n_tile_types, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        # Flatten input if needed
        if len(z.shape) > 2:
            z = z.view(z.size(0), -1)
            
        # Pass through FC layers
        x = self.fc(z)
        
        # Reshpe to feature maps
        x = x.view(-1, 128, 7, 14)
        
        # Pass through conv layers
        x = self.conv(x)
        
        # Ensure output size
        if x.size(2) != self.patch_height or x.size(3) != self.patch_width:
            x = nn.functional.interpolate(x, size=(self.patch_height, self.patch_width), mode='bilinear', align_corners=False)
        
        return x
        
        # Starting with input size (latent_dim, 1, 1)
        # We'll use a first layer to get to 4×7, then double twice to reach 14×28
    #     self.main = nn.Sequential(
    #         # Input: (latent_dim, 1, 1)
    #         nn.ConvTranspose2d(latent_dim, 256, kernel_size=(4, 4), stride=1, padding=0, bias=False),
    #         # Output: (256, 4, 4)
    #         nn.BatchNorm2d(256),
    #         nn.ReLU(True),
            
    #         # Input: (256, 4, 4)
    #         nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(1, 2), padding=1, bias=False),
    #         # Output: (128, 4, 7)
    #         nn.BatchNorm2d(128),
    #         nn.ReLU(True),
            
    #         # Input: (128, 4, 7)
    #         nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=2, padding=1, bias=False),
    #         # Output: (64, 7, 14)
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(True),
            
    #         # Input: (64, 7, 14)
    #         nn.ConvTranspose2d(64, n_tile_types, kernel_size=(4, 4), stride=2, padding=1, bias=False),
    #         # Output: (n_tile_types, 14, 28)
    #         nn.Sigmoid()
    #     )

    # def forward(self, z):
    #     # Ensure z is in the right shape for ConvTranspose2d
    #     if len(z.shape) == 2:
    #         z = z.unsqueeze(-1).unsqueeze(-1)  # Add spatial dimensions
    #     return self.main(z)  # Output shape: (batch_size, n_tile_types, 14, 28)

class DCGANDiscriminator(nn.Module):
    def __init__(self, n_tile_types, patch_height=14, patch_width=28):
        super(DCGANDiscriminator, self).__init__()
        
        # Conv layers + FC
        self.conv = nn.Sequential(
            nn.Conv2d(n_tile_types, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 7, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        
        return x.squeeze()
            
            # # Input: (64, 7, 14)
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            # # Output: (128, 7, 14)
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.MaxPool2d(2),  # Halve dimensions to (128, 3, 7)
            
            # # Input: (128, 3, 7)
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, bias=False),
            # # Output: (256, 1, 5)
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.2, inplace=True),
            
            # # Final classification layer
            # nn.Conv2d(256, 1, kernel_size=(1, 5), stride=1, padding=0, bias=False),
            # # Output: (1, 1, 1)
            # nn.Sigmoid()

    # def forward(self, x):
    #     return self.main(x).squeeze()