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

print("System path:", sys.path)
print("Available files in utils:", os.listdir(os.path.join(parent_dir, "utils")))


from utils.process_data import ProcessDataSymbolic
from mlp_model import MLPGenerator


class MarioLevelDataset(Dataset):
    # Dataset for Mario levels
    
    def __init__(self, level_patches):
        # List of level patches in one-hot encoded format
        # self.level_patches = [torch.tensor(patch, dtype=torch.float32) / 1.0 for patch in level_patches]
        self.level_patches = [torch.clamp(torch.tensor(patch, dtype=torch.float32), 0, 1) for patch in level_patches]
    
    def __len__(self):
        return len(self.level_patches) # Returns the number of level patches for one level
    
    def __getitem__(self, idx):
        return self.level_patches[idx] # Returns the level patch at the given index


def train_mlp(generator, dataloader, num_epochs=100, lr=0.0002, device='cpu'):
    # Train the generator using binary cross entropy loss
    generator.to(device) # Move generator to specified device
    
    # Define loss and optimizer
    criterion = nn.BCELoss() # Binary cross entropy loss
    optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999)) # Adam optimizer
    
    # Keep track of losses per epoch for visualization
    losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        
        for _, real_levels in enumerate(dataloader):
            batch_size = real_levels.size(0) # Get batch size
            real_levels = real_levels.to(device) # Move real levels to specified device
            
            # Generate random noise
            z = torch.randn(batch_size, generator.latent_dim, device=device) 
            
            # Generate fake levels
            fake_levels = generator(z)
            
            # Train generator to produce levels similar to real ones
            optimizer.zero_grad() # Zero gradients
            loss = criterion(fake_levels, real_levels) # Calculate loss
            loss.backward() # Backpropagate loss
            optimizer.step() # Update parameters
            
            total_loss += loss.item() # Add loss to total loss
            
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
    
    print(f"Loading and processing {len(symbolic_files)} symbolic files from {symb_data_folder}")
    for symb_file in symbolic_files:
        print(f"Processing {symb_file}...")
        
        try:
            # Load symbolic level data
            processor.load_symbolic(symb_file)
            
            # Crop level into patches
            patches = processor.crop_symbolic()

            # Convert patches to one-hot encoded vectors and collect them
            for patch in patches:
                id_file, vector_file = processor.forward_mapping(patch) # Convert patch to one-hot encoded vector
                all_one_hot_patches.append(vector_file)
                
            for patch in all_one_hot_patches:
                if not torch.is_tensor(patch):
                    patch = torch.tensor(patch)
                print("Min: ", patch.min(), "Max: ", patch.max())
                break
        except Exception as e:
            print(f"Error processing {symb_file}: {e}")
            continue
    
    print(f"Finished processing all files. Total patches collected: {len(all_one_hot_patches)}")
    
    # Create dataset and dataloader from ALL collected patches
    if not all_one_hot_patches:
        raise ValueError("No valid patches found. Please check the symbolic files.")
    
    dataset = MarioLevelDataset(all_one_hot_patches)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Get level dimensions
    level_height, level_width = all_one_hot_patches[0].shape[:2]
    n_tile_types = all_one_hot_patches[0].shape[2]
    
    # Create and train generator
    generator = MLPGenerator(level_height, level_width, n_tile_types)
    trained_generator, training_losses = train_mlp(generator, dataloader)
    
    # Save the trained model
    model_save_path = os.path.join(os.path.dirname(__file__), 'mlp_mario_generator.pth')
    torch.save(trained_generator.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")
    
    # Plot and the loss curve
    plt.figure(figsize=(10,5))
    plt.plot(training_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    
    # Generate a sample level 
    patch_width = trained_generator.patch_width
    num_patches = patch_width * 7
    
    print(f"Attempting to generate a whole level of width: {num_patches}")
    symbolic_level = trained_generator.generate_whole_level(num_patches, processor)
    
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