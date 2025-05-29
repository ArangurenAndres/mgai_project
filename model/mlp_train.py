import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.process_data import ProcessDataSymbolic
from model.mlp_model import MLPGenerator

class MarioLevelDataset(Dataset):
    # Dataset for Mario levels
    
    def __init__(self, level_patches):
        # Store level patches in one-hot format
        self.level_patches = [torch.clamp(torch.tensor(patch, dtype=torch.float32), 0, 1) for patch in level_patches]
    
    def __len__(self):
        return len(self.level_patches) # Returns the number of level patches for one level
    
    def __getitem__(self, idx):
        return self.level_patches[idx] # Returns the level patch at the given index


def train_mlp(generator, dataloader, num_epochs=100, lr=0.0002, device='cpu'):
    # Train the generator using binary cross entropy loss
    generator.to(device) # Move generator to specified device
    
    # Define loss and optimizer
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999)) 
    
    # Track losses for visualization
    losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        
        for _, real_levels in enumerate(dataloader):
            batch_size = real_levels.size(0) 
            real_levels = real_levels.to(device) 
            
            # Generate random noise
            z = torch.randn(batch_size, generator.latent_dim, device=device) 
            
            # Generate fake levels
            fake_levels = generator(z)
            
            # Train generator to produce levels similar to real ones
            optimizer.zero_grad() 
            loss = criterion(fake_levels, real_levels) 
            loss.backward() 
            optimizer.step() 
            
            total_loss += loss.item() 
            
        # Calculate avg loss for this epoch
        avg_loss = total_loss/len(dataloader)
        losses.append(avg_loss) # Store the loss
        
        # Print training progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")
    
    print("Training completed!")
    return generator, losses


if __name__ == "__main__":
    # Load and process data
    config_path = os.path.abspath(os.path.join(parent_dir, 'config.yaml'))
    mapping_path = os.path.abspath(os.path.join(parent_dir, 'utils', 'mapping.yaml'))
    
    # Create the data processor
    processor = ProcessDataSymbolic(mapping_path=mapping_path)
    
    # Get the symbolic data folder and files
    symb_data_folder = processor.folder_path
    symbolic_files = [f for f in os.listdir(symb_data_folder) if f.endswith('.txt')]
    
    if not symbolic_files:
        raise FileNotFoundError(f"No .txt files found in the symbolic data folder: {symb_data_folder}")
    
    # List to store all the one-hot encoded patches from the files
    all_one_hot_patches = []
    
    # Convert the symbolic levels into one-hot encoded patches
    print("Processing symbolic levels into one-hot patches...")
    for symb_file in symbolic_files:
        processor.load_symbolic(symb_file)
        patches = processor.crop_symbolic()
        for patch in patches:
            _, onehot_file = processor.forward_mapping_onehot(patch)
            all_one_hot_patches.append(onehot_file)
            
    print(f"Finished processing {len(all_one_hot_patches)} patches.")
    
    # Create dataset and dataloader from ALL collected patches
    dataset = MarioLevelDataset(all_one_hot_patches)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Get level dimensions
    sample = dataset[0]
    print(f"Sample patch shape: {sample.shape}")
    patch_height = sample.size(0)
    patch_width = sample.size(1)
    n_tile_types = sample.size(2)
    
    print(f"Patch dimensions: {patch_height}x{patch_width} with {n_tile_types} tile types")
    
    # Create and train generator
    generator = MLPGenerator(patch_height, patch_width, n_tile_types)
    trained_generator, training_losses = train_mlp(generator, dataloader)
    
    # Save the trained model
    model_save_path = os.path.join(os.path.dirname(__file__), 'mlp_mario_generator.pth')
    torch.save(trained_generator.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")
    
    # Plot the loss curve
    plt.figure(figsize=(10,5))
    plt.plot(training_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    
    # Generate a sample level 
    level_width = 7
    level_height = 1
    
    print(f"Attempting to generate a whole level of width: {level_width} patches")
    symbolic_level = trained_generator.generate_whole_level(level_width, processor, level_height)
    
    # Create output directory to store the generated levels
    output_dir = os.path.join(os.path.dirname(__file__), 'generated_levels')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the filename along with the timestamp to avoid overwriting existing levels
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(output_dir, f'mlp_level_{timestamp}.txt')
    
    # Also save the plot with the same timestamp
    plot_path = os.path.join(output_dir, f'training_loss_{timestamp}')
    plt.savefig(plot_path)
    print(f"Training loss plot saved to: {plot_path}")
    plt.close()
    
    # Save the generated level to the file
    print(f"Saving generated level to: {output_file}")
    with open(output_file, 'w') as f:
        for row in symbolic_level:
            f.write(row + '\n')
        
    # Display the generated level
    print("\nGenerated level after training:")
    processor.visualize_file(symbolic_level)

    
    