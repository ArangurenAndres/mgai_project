import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import sys
import matplotlib.pyplot as plt
import time
from dcgan_model import DCGANGenerator, DCGANDiscriminator

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from utils.process_data import ProcessDataSymbolic

class MarioLevelDataset(Dataset):
    def __init__(self, level_patches):
        # List of level patches in one-hot encoded format
        self.level_patches = [torch.clamp(torch.tensor(patch, dtype=torch.float32), 0, 1).permute(2, 0, 1) for patch in level_patches]
        
    def __len__(self):
        return len(self.level_patches) # Number of level patches
    
    def __getitem__(self, index):
        return self.level_patches[index] # Level patch at the given index

def train_dcgan(generator, discriminator, dataloader, num_epochs=10, lr=0.0002, device='cpu'):
    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCELoss() # BCE is used to measure how well the discriminator distinguishes the real and fake data
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    real_label = 1.0 # Real mario level patches
    fake_label = 0.0 # Fake (generated) mario level patches

    # Lists to store the losses of each epoch
    g_losses = []
    d_losses = []

    for epoch in range(num_epochs):
        current_g_loss = 0.0
        current_d_loss = 0.0
        num_batches = 0
        
        for i, real_data in enumerate(dataloader):
            batch_size = real_data.size(0)
            real_data = real_data.to(device)

            # Train Discriminator
            discriminator.zero_grad()
            labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            raw_output = discriminator(real_data) # Pass real data through the discriminator
            output = raw_output[:, 0] if raw_output.dim() > 1 and raw_output.size(1) > 1 else raw_output
            
            # Calculate the loss for real data
            loss_d_real = criterion(output, labels) 
            loss_d_real.backward()

            # Generate random noise for the generator input
            z = torch.randn(batch_size, generator.latent_dim, device=device)
            if len(z.shape) == 2:  # Add spatial dimensions if needed for ConvTranspose2d
                z = z.unsqueeze(-1).unsqueeze(-1)
                
            fake_data = generator(z) # Generate fake data
            labels.fill_(fake_label)
            
            # Get discriminator output on fake data
            raw_output = discriminator(fake_data.detach())
            output = raw_output[:, 0] if raw_output.dim() > 1 and raw_output.size(1) > 1 else raw_output
            
            # Calculate the loss for fake data
            loss_d_fake = criterion(output, labels)
            loss_d_fake.backward()
            optimizer_d.step()

            # Combine the two losses to update the discriminator
            loss_d = loss_d_real + loss_d_fake

            # Train Generator
            generator.zero_grad()
            labels.fill_(real_label)
            
            # Get discriminator output on fake data again (for generator training)
            raw_output = discriminator(fake_data)
            output = raw_output[:, 0] if raw_output.dim() > 1 and raw_output.size(1) > 1 else raw_output
            
            # Calculate the loss based on how well the generator fools the discriminator
            loss_g = criterion(output, labels)
            loss_g.backward()
            optimizer_g.step()

            # Add up losses for current epoch
            current_g_loss += loss_g.item()
            current_d_loss += loss_d.item()
            num_batches += 1

        # Calculate avg loss per epoch
        avg_g_loss = current_g_loss / num_batches
        avg_d_loss = current_d_loss / num_batches
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss D: {avg_d_loss:.4f}, Avg Loss G: {avg_g_loss:.4f}")

    return generator, g_losses, d_losses

def generate_whole_level(generator, processor, num_patches_width=7, num_patches_height=14, device='cpu'):
    """
    Generate a whole level by creating multiple patches using the DCGAN generator and stitching them together.
    
    Args:
        generator: The trained DCGAN generator
        processor: ProcessDataSymbolic instance for processing patches
        num_patches_width: Number of patches to generate horizontally
        num_patches_height: Number of patches to generate vertically
        device: Device to use for generation (CPU/GPU)
    
    Returns:
        symbolic_level: The complete level in symbolic format
    """
    generator.eval()  # Set to evaluation mode
    symbolic_level = []
    
    print(f"Generating a level with dimensions: {num_patches_height} x {num_patches_width} patches")
    
    with torch.no_grad():
        for h in range(num_patches_height):
            row_patches = []
            for w in range(num_patches_width):
                # Generate a random noise vector
                z = torch.randn(1, generator.latent_dim, device=device)
                if len(z.shape) == 2:  # Add spatial dimensions if needed
                    z = z.unsqueeze(-1).unsqueeze(-1)
                
                # Generate a patch
                generated_patch = generator(z)
                
                # Convert from tensor to numpy and reshape
                patch_np = generated_patch.squeeze(0).permute(1, 2, 0).cpu().numpy()
                
                # Find the most likely tile type for each position (one-hot to symbolic)
                symbolic_patch = processor.backward_mapping(patch_np)
                
                # Extract the string representation from the result (second element of tuple)
                if isinstance(symbolic_patch, tuple):
                    symbolic_patch = symbolic_patch[1]  # Get the SECOND element (strings) if it's a tuple
                
                # Add to the current row
                row_patches.append(symbolic_patch)
            
            # For each row in the patch height
            for i in range(len(row_patches[0])):
                # Join the i-th row from each patch in this row
                symbolic_row = ''.join([patch[i] for patch in row_patches])
                symbolic_level.append(symbolic_row)
    
    return symbolic_level


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and process data
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    mapping_path = os.path.join(parent_dir, 'utils', 'mapping.yaml')
    
    # Define the processor in ordedr to handle loading symbolic files and conversion to one-hot patches
    processor = ProcessDataSymbolic(mapping_path=mapping_path)

    # Load the folder of symbolic mario levels
    symb_data_folder = processor.folder_path
    symbolic_files = [f for f in os.listdir(symb_data_folder) if f.endswith('.txt')]

    all_one_hot_patches = []
    
    # Convert the symbolic levels into one-hot encoded patches
    print("Processing symbolic levels into one-hot patches...")
    for symb_file in symbolic_files:
        processor.load_symbolic(symb_file)
        patches = processor.crop_symbolic()
        for patch in patches:
            _, vector_file = processor.forward_mapping(patch)
            
            # Remove the padding logic to keep original dimensions (14x28)
            all_one_hot_patches.append(vector_file)
            
        print(f"Finished processing {len(all_one_hot_patches)} patches.")
    
    # Create a dataset and dataloader for training
    dataset = MarioLevelDataset(all_one_hot_patches)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Get the dimensions from the one-hot patches
    # Note: after permute in the dataset, the shape is (C, H, W)
    sample = dataset[0]
    n_tile_types = sample.size(0)
    patch_height = sample.size(1)  # Height is dimension 1 after permute
    patch_width = sample.size(2)   # Width is dimension 2 after permute
    
    print(f"Patch dimensions: {patch_height}x{patch_width} with {n_tile_types} tile types")

    # # Extract the level height and with (16 x 28) and the number of tile types (10)
    # level_height, level_width = all_one_hot_patches[0].shape[:2]
    # n_tile_types = all_one_hot_patches[0].shape[2]

    # Create instances of the generator and discriminator
    generator = DCGANGenerator(latent_dim=100, n_tile_types=n_tile_types, 
                               patch_height=patch_height, patch_width=patch_width)
    discriminator = DCGANDiscriminator(n_tile_types=n_tile_types, 
                                     patch_height=patch_height, patch_width=patch_width)

    # Train the generator
    trained_generator, g_losses, d_losses = train_dcgan(generator, discriminator, dataloader)

    # Save the trained generator
    model_save_path = os.path.join(os.path.dirname(__file__), 'dcgan_mario_generator.pth')
    torch.save(trained_generator.state_dict(), model_save_path)
    print(f"Trained DCGAN generator saved to {model_save_path}")

    # Plot the training losses (Epoch-wise)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(g_losses) + 1), g_losses, label='Generator Loss', marker='o', markersize=4)
    plt.plot(range(1, len(d_losses) + 1), d_losses, label='Discriminator Loss', marker='x', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DCGAN Training Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, len(g_losses) + 1, max(1, len(g_losses) // 10))) # Show ticks for roughly every 10% of epochs
    plt.tight_layout()
    
    # Create output directory for plots and generated levels
    output_dir = os.path.join(os.path.dirname(__file__), 'generated_patches')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot with a timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plot_path = os.path.join(output_dir, f'dcgan_training_loss_{timestamp}.png')
    plt.savefig(plot_path)
    print(f"Training loss plot saved to: {plot_path}")
    plt.close()
    
    print("\nGenerating a complete level using the trained DCGAN generator...")
    level_width = 7  # Number of patches horizontally
    level_height = 3  # Number of patches vertically
    symbolic_level = generate_whole_level(trained_generator, processor, level_width, level_height, device)

    # Create output directory for the generated levels
    levels_output_dir = os.path.join(os.path.dirname(__file__), 'generated_levels')
    os.makedirs(levels_output_dir, exist_ok=True)

    # Save the generated level with timestamp
    level_timestamp = time.strftime("%Y%m%d-%H%M%S")
    level_output_file = os.path.join(levels_output_dir, f'dcgan_level_{level_timestamp}.txt')

    # Save the generated level to file
    print(f"Saving generated level to: {level_output_file}")
    with open(level_output_file, 'w') as f:
        for row in symbolic_level:
            f.write(row + '\n')

    # Display the generated level
    print("\nGenerated level after training:")
    processor.visualize_file(symbolic_level)
    