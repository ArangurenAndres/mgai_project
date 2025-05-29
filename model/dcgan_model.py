import torch
import torch.nn as nn

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim, n_tile_types, patch_height=14, patch_width=28, hidden_dim=512):
        super(DCGANGenerator, self).__init__()
        
        # Intialize parameters
        self.latent_dim = latent_dim
        self.n_tile_types = n_tile_types
        self.patch_height = patch_height
        self.patch_width = patch_width
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(True),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 2, 128 * 7 * 14),  
            nn.BatchNorm1d(128 * 7 * 14),
            nn.ReLU(True)
        )
        
        # Convolutional layers
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
        
        # Reshape to feature maps
        x = x.view(-1, 128, 7, 14)
        
        # Pass through conv layers
        x = self.conv(x)
        
        # Ensure output size matches desired dimensions
        if x.size(2) != self.patch_height or x.size(3) != self.patch_width:
            x = nn.functional.interpolate(x, size=(self.patch_height, self.patch_width), mode='bilinear', align_corners=False)
        
        return x

class DCGANDiscriminator(nn.Module):
    def __init__(self, n_tile_types, patch_height=14, patch_width=28):
        super(DCGANDiscriminator, self).__init__()
        
        # Convolutional layers for feature extraction
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
        
        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 7, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Pass through convolutional layers
        x = self.conv(x)
        
        # Pass through fully connected layers
        x = self.fc(x)
        
        return x.squeeze()