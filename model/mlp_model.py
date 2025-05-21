import torch
import torch.nn as nn
import numpy as np

class MLPGenerator(nn.Module):
    def __init__(self, patch_height, patch_width, n_tile_types=10, latent_dim=32, hidden_dim=256):
        super(MLPGenerator, self).__init__()
        
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.n_tile_types = n_tile_types
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        output_size = patch_height * patch_width * n_tile_types
        
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
        return self.main(z).view(-1, self.patch_height, self.patch_width, self.n_tile_types)
    
    def generate_patch(self, batch_size=1, device='cpu'):
        # Generate random noise
        z = torch.randn(batch_size, self.latent_dim, device=device)
        
        # Generate level
        with torch.no_grad():
            patch = self(z) # Pass noise through the generator
            
        # Convert to numpy for processing with existing code
        return patch.cpu().numpy() # Convert to numpy array
    
    def generate_symbolic_patch(self, processor, batch_size=1, device='cpu'):
        # Generate level as one-hot encoding
        vector_patch = self.generate_patch(batch_size, device)[0]  # Get first level
        
        # Convert to identity representation
        id_patch = processor.convert_vector_to_id(vector_patch)
        
        # Convert to symbolic representation
        symbolic_patch = processor.convert_identity_to_symbolic(id_patch)
        
        return symbolic_patch
    
    def generate_whole_level(self, level_tile_width, processor, device='cpu'):
        # Generates a whole level by stitching together patches (might have seams between incoherent patches)
        patch_width = self.patch_width
        
        if level_tile_width % patch_width != 0:
            raise ValueError(f"Desired level width ({level_tile_width}) must be a multiple of patch width ({patch_width})")
        
        # Number of patches to generate to stitch into one level
        num_patches = level_tile_width // patch_width
        
        # Generate enough patches to cover the desired width level_tile_width
        generated_patches = []
        print(f"Generating {num_patches} patches to stitch into one level...")
        
        for i in range(num_patches):
            # Generate a patch
            print(f"Generating patch {i+1}/{num_patches}...")
            patch_vector = self.generate_patch(batch_size=1, device=device)[0] # Get the single patch
            generated_patches.append(patch_vector)
            
        # Stitch the patches horizontally
        print("Stitching patches...")
        
        # Initialize the full level array using the first patch
        full_level = np.concatenate(generated_patches, axis=1)
        
        # # Append each subsequent patch horizontally
        # for i in range(1, len(generated_patches)):
        #     for row in range(self.patch_height):
        #         # For each row, extend it with the corresponding row from the next patch
        #         full_level[row] = np.concatenate([full_level[row], generated_patches[i][row]], axis=0)
            
        # Convert the  whole level vector back to symbolic format
        print("Converting stitched level to symbolic format...")
        id_level = processor.convert_vector_to_id(full_level)
        symbolic_level = processor.convert_identity_to_symbolic(id_level)
            
            
        print("Whole level generation complete.")
        return symbolic_level