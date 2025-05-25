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
        # Convert to tensors and permute to (C, H, W) format
        self.level_patches = [torch.clamp(torch.tensor(patch, dtype=torch.float32), 0, 1).permute(2, 0, 1) for patch in level_patches]
        
    def __len__(self):
        return len(self.level_patches) # Number of level patches
    
    def __getitem__(self, index):
        return self.level_patches[index] # Level patch at the given index

def train_dcgan(generator, discriminator, dataloader, num_epochs=50, lr_g=0.0005, lr_d=0.0001, device='cpu', patience=10, min_delta=0.01, noise_std=0.02, d_train_freq=2, g_train_freq=1):
    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCELoss() # BCE is used to measure how well the discriminator distinguishes the real and fake data
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    # LR Scheduler to reduce learning rate as training progresses
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.8)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=10, gamma=0.8)

    # At the end of each epoch
    scheduler_g.step()
    scheduler_d.step()

    real_label = 0.9 # Real mario level patches
    fake_label = 0.1 # Fake (generated) mario level patches

    # Lists to store the losses of each epoch
    g_losses = []
    d_losses = []
    
    # Early stopping variables
    best_g_loss = float('inf')
    epochs_plateu = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        current_g_loss = 0.0
        current_d_loss = 0.0
        num_batches = 0
        
        for i, real_data in enumerate(dataloader):
            batch_size = real_data.size(0)
            real_data = real_data.to(device)

            # Add noise to real data if specified
            if noise_std > 0:
                noise = torch.randn_like(real_data) * noise_std
                real_data = torch.clamp(real_data + noise, 0, 1)

            # Train Discriminator
            if i % d_train_freq == 0:
                discriminator.zero_grad()

                # Real data
                labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
                output = discriminator(real_data)
                loss_d_real = criterion(output, labels)
                loss_d_real.backward()

                # Fake data
                z = torch.randn(batch_size, generator.latent_dim, device=device)
                fake_data = generator(z)
                labels.fill_(fake_label)
                output = discriminator(fake_data.detach())
                loss_d_fake = criterion(output, labels)
                loss_d_fake.backward()
                optimizer_d.step()

                # Combine the two losses to update the discriminator
                loss_d = loss_d_real + loss_d_fake
            else:
                # Skip discriminator training but still generate fake data
                z = torch.randn(batch_size, generator.latent_dim, device=device) + torch.randn_like(z) * 0.05
                fake_data = generator(z)
                loss_d = torch.tensor(0.0)

            # Train Generator
            if i % g_train_freq == 0:
                generator.zero_grad()
                labels.fill_(real_label)

                # Get discriminator output on fake data again (for generator training)
                output = discriminator(fake_data)

                # Calculate the loss based on how well the generator fools the discriminator
                loss_g = criterion(output, labels)
                div_loss = diversity_loss(fake_data)
                total_g_loss = loss_g + div_loss
                total_g_loss.backward()
                optimizer_g.step()
            else:
                loss_g = torch.tensor(0.0)

            # Add up losses for current epoch
            current_g_loss += loss_g.item() if isinstance(loss_g, torch.Tensor) else loss_g
            current_d_loss += loss_d.item() if isinstance(loss_d, torch.Tensor) else loss_d
            num_batches += 1

        # Calculate avg loss per epoch
        avg_g_loss = current_g_loss / num_batches
        avg_d_loss = current_d_loss / num_batches
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        # Early stopping check
        if avg_g_loss < best_g_loss - min_delta:
            best_g_loss = avg_g_loss
            epochs_plateu = 0
            best_epoch = epoch + 1
            print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss D: {avg_d_loss:.4f}, Avg Loss G: {avg_g_loss:.4f}")
        else:
            epochs_plateu += 1
            print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss D: {avg_d_loss:.4f}, Avg Loss G: {avg_g_loss:.4f} (No improvement: {epochs_plateu}/{patience})")
            
        # Check if we should stop early
        if epochs_plateu >= patience:
            print(f"\nEarly stopping triggered!")
            print(f"No improvement in generator loss for {patience} epochs.")
            print(f"Best generator loss: {best_g_loss:.4f} at epoch {best_epoch}")
            break
        
    # # Final summary
    # if epochs_plateu < patience:
    #     print(f"\nTraining completed all {num_epochs} epochs.")
    #     print(f"Best generator loss: {best_g_loss:.4f} at epoch {best_epoch}")
            
    print(f"\nTraining completed. Best generator loss: {best_g_loss:.4f} at epoch {best_epoch}")
    return generator, g_losses, d_losses

def generate_whole_level(generator, processor, num_patches_width=7, num_patches_height=14, device='cpu'):
    generator.eval()  # Set to evaluation mode
    symbolic_level = []
    
    print(f"Generating a level with dimensions: {num_patches_height} x {num_patches_width} patches")
    
    with torch.no_grad():
        for h in range(num_patches_height):
            row_patches = []
            for w in range(num_patches_width):
                # Generate a random noise vector
                z = torch.randn(1, generator.latent_dim, device=device)
                
                # Generate a patch
                generated_patch = generator(z)
                
                # Convert from tensor to numpy and reshape
                patch_np = generated_patch.squeeze(0).permute(1, 2, 0).cpu().numpy()
                
                # Convert to symbolic
                symbolic_patch = processor.backward_mapping_onehot(patch_np)
                
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

def diversity_loss(generated_batch):
    batch_size = generated_batch.size(0)
    generated_flat = generated_batch.view(batch_size, -1)
    
    # Calculate pairwise distances
    distances = torch.cdist(generated_flat, generated_flat)
    
    # Encourage larger distaances (more diversity)
    diversity_loss = -torch.mean(distances)
    
    return diversity_loss * 0.1

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
            _, onehot_file = processor.forward_mapping_onehot(patch)
            all_one_hot_patches.append(onehot_file)
            
    print(f"Finished processing {len(all_one_hot_patches)} patches.")
    
    # Create a dataset and dataloader for training
    dataset = MarioLevelDataset(all_one_hot_patches)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Get the dimensions from the one-hot patches
    sample = dataset[0]
    print(sample)
    n_tile_types = sample.size(0)
    patch_height = sample.size(1) 
    patch_width = sample.size(2)  
    
    print(f"Patch dimensions: {patch_height}x{patch_width} with {n_tile_types} tile types")
    
    # Create instances of the generator and discriminator
    generator = DCGANGenerator(latent_dim=128, n_tile_types=n_tile_types, 
                               patch_height=patch_height, patch_width=patch_width)
    discriminator = DCGANDiscriminator(n_tile_types=n_tile_types, 
                                     patch_height=patch_height, patch_width=patch_width)

    # Train the generator
    trained_generator, g_losses, d_losses = train_dcgan(
        generator, discriminator, dataloader, 
        num_epochs=100, lr_g=0.0006, lr_d=0.00008, device=device, 
        patience=15, min_delta=0.005, noise_std=0.02,
        d_train_freq=3, g_train_freq=1
    )

    
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
    plt.tight_layout()
    
    # Create output directory for plots and generated levels
    output_dir = os.path.join(os.path.dirname(__file__), 'generated_patches')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plot_path = os.path.join(output_dir, f'dcgan_training_loss_{timestamp}.png')
    plt.savefig(plot_path)
    print(f"Training loss plot saved to: {plot_path}")
    plt.close()
    
    print("\nGenerating a complete level using the trained DCGAN generator...")
    level_width = 7  # Number of patches horizontally
    level_height = 1
    symbolic_level = generate_whole_level(trained_generator, processor, level_width, level_height, device)

    # Create output directory for the generated levels
    levels_output_dir = os.path.join(os.path.dirname(__file__), 'generated_levels')
    os.makedirs(levels_output_dir, exist_ok=True)
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
     
    # Render the level using run_render()
    tile_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'tiles'))  # Path to tiles
    rendered_img = processor.render_level_image(
        symb_name=f"dcgan_level_{level_timestamp}.txt",
        symb_file=symbolic_level,
        tile_dir=tile_dir,
        save_folder=levels_output_dir
    )
    
    # Display the rendered image
    rendered_img.show()