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

def train_dcgan(generator, discriminator, dataloader, num_epochs=100, 
                        lr_g=0.0002, lr_d=0.0001, device='cpu', patience=20, 
                        min_delta=0.001, noise_std=0.02, d_train_freq=1, g_train_freq=1,
                        use_spectral_norm=True, gradient_penalty_weight=10.0):
    
    generator.to(device)
    discriminator.to(device)

    # Use different loss functions
    criterion = nn.BCELoss()
    
    # Different optimizers with better parameters
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999), weight_decay=1e-5)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999), weight_decay=1e-5)
    
    # Cosine annealing scheduler
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=num_epochs)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=num_epochs)

    # Label smoothing with noise
    real_label_base = 0.9
    fake_label_base = 0.1
    label_noise = 0.05

    g_losses = []
    d_losses = []
    
    best_g_loss = float('inf')
    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        current_g_loss = 0.0
        current_d_loss = 0.0
        num_batches = 0
        
        for i, real_data in enumerate(dataloader):
            batch_size = real_data.size(0)
            real_data = real_data.to(device)

            # Add noise to real data
            if noise_std > 0:
                noise = torch.randn_like(real_data) * noise_std
                real_data = torch.clamp(real_data + noise, 0, 1)

            # Dynamic label smoothing
            real_label = real_label_base + torch.randn(batch_size, device=device) * label_noise
            fake_label = fake_label_base + torch.randn(batch_size, device=device) * label_noise
            real_label = torch.clamp(real_label, 0.7, 1.0)
            fake_label = torch.clamp(fake_label, 0.0, 0.3)

            # Train Discriminator
            if i % d_train_freq == 0:
                discriminator.zero_grad()

                # Real data
                output_real = discriminator(real_data)
                loss_d_real = criterion(output_real, real_label)

                # Fake data
                z = torch.randn(batch_size, generator.latent_dim, device=device)
                fake_data = generator(z)
                output_fake = discriminator(fake_data.detach())
                loss_d_fake = criterion(output_fake, fake_label)

                # Gradient penalty (WGAN-GP style)
                gradient_penalty = compute_gradient_penalty(discriminator, real_data, fake_data, device)
                
                loss_d = loss_d_real + loss_d_fake + gradient_penalty_weight * gradient_penalty
                loss_d.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optimizer_d.step()
            else:
                z = torch.randn(batch_size, generator.latent_dim, device=device)
                fake_data = generator(z)
                loss_d = torch.tensor(0.0)

            # Train Generator
            if i % g_train_freq == 0:
                generator.zero_grad()
                
                # Use fresh fake data for generator training
                z = torch.randn(batch_size, generator.latent_dim, device=device)
                fake_data = generator(z)
                output = discriminator(fake_data)
                
                # Generator wants discriminator to think fake is real
                real_labels_for_gen = torch.full((batch_size,), real_label_base, device=device)
                loss_g = criterion(output, real_labels_for_gen)
                
                # Add diversity loss
                div_loss = diversity_loss(fake_data)
                
                # Feature matching loss
                feature_loss = feature_matching_loss(discriminator, real_data, fake_data)
                
                total_g_loss = loss_g + 0.1 * div_loss + 0.05 * feature_loss
                total_g_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_g.step()
            else:
                loss_g = torch.tensor(0.0)

            current_g_loss += loss_g.item() if isinstance(loss_g, torch.Tensor) else loss_g
            current_d_loss += loss_d.item() if isinstance(loss_d, torch.Tensor) else loss_d
            num_batches += 1

        # Update learning rates
        scheduler_g.step()
        scheduler_d.step()

        # Calculate average losses
        avg_g_loss = current_g_loss / num_batches
        avg_d_loss = current_d_loss / num_batches
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        # Early stopping with more patience
        if avg_g_loss < best_g_loss - min_delta:
            best_g_loss = avg_g_loss
            epochs_without_improvement = 0
            best_epoch = epoch + 1
            print(f"Epoch [{epoch+1}/{num_epochs}] - D: {avg_d_loss:.4f}, G: {avg_g_loss:.4f} ⭐")
        else:
            epochs_without_improvement += 1
            print(f"Epoch [{epoch+1}/{num_epochs}] - D: {avg_d_loss:.4f}, G: {avg_g_loss:.4f} ({epochs_without_improvement}/{patience})")
            
        # Generate sample every 10 epochs for monitoring
        if epoch % 10 == 0:
            generator.eval()  # Set to evaluation mode to avoid BatchNorm issues
            with torch.no_grad():
                sample_z = torch.randn(1, generator.latent_dim, device=device)
                sample_output = generator(sample_z)
                print(f"Sample output stats - Min: {sample_output.min():.3f}, Max: {sample_output.max():.3f}, Mean: {sample_output.mean():.3f}")
            generator.train()  # Set back to training mode
            
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\nTraining completed. Best G loss: {best_g_loss:.4f} at epoch {best_epoch}")
    return generator, g_losses, d_losses

def compute_gradient_penalty(discriminator, real_data, fake_data, device):
    """Compute gradient penalty for WGAN-GP"""
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    
    d_interpolated = discriminator(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

def feature_matching_loss(discriminator, real_data, fake_data):
    """Feature matching loss to stabilize training"""
    # Get intermediate features from discriminator
    real_features = []
    fake_features = []
    
    def hook_fn(module, input, output):
        return output
    
    # This is a simplified version - you'd need to modify discriminator to return intermediate features
    with torch.no_grad():
        real_output = discriminator(real_data)
        fake_output = discriminator(fake_data)
    
    return torch.nn.functional.mse_loss(fake_output.mean(), real_output.mean())

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

def progressive_training(generator, discriminator, dataloader, device):
    """Train with progressively increasing difficulty and decreasing learning rates"""
    
    print("="*60)
    print("STARTING PROGRESSIVE TRAINING")
    print("="*60)
    
    all_g_losses = []
    all_d_losses = []
    
    # Phase 1: Warm-up with higher learning rates and shorter patience
    print("\n Phase 1: Warm-up training (High LR, Fast Convergence)")
    print("-" * 50)
    _, g_losses_1, d_losses_1 = train_dcgan(
        generator, discriminator, dataloader,
        num_epochs=40,           # Shorter warm-up
        lr_g=0.0004,            # Higher generator LR
        lr_d=0.0002,            # Higher discriminator LR
        device=device,
        patience=15,            # Less patience
        min_delta=0.001,        # Larger improvement threshold
        noise_std=0.02,         # More noise for robustness
        d_train_freq=1,
        g_train_freq=1,
        gradient_penalty_weight=2.0  # Lower penalty initially
    )
    all_g_losses.extend(g_losses_1)
    all_d_losses.extend(d_losses_1)
    
    # Phase 2: Stable training with balanced learning rates
    print("\n Phase 2: Stable training (Balanced LR, Standard Training)")
    print("-" * 50)
    _, g_losses_2, d_losses_2 = train_dcgan(
        generator, discriminator, dataloader,
        num_epochs=80,           # Main training phase
        lr_g=0.0002,            # Your optimal LR from ablation
        lr_d=0.0001,            # Your optimal LR from ablation
        device=device,
        patience=25,            # More patience
        min_delta=0.0005,       # Smaller improvement threshold
        noise_std=0.015,        # Moderate noise
        d_train_freq=1,
        g_train_freq=1,
        gradient_penalty_weight=5.0  # Standard penalty
    )
    all_g_losses.extend(g_losses_2)
    all_d_losses.extend(d_losses_2)
    
    # Phase 3: Fine-tuning with lower learning rates
    print("\n Phase 3: Fine-tuning (Low LR, High Precision)")
    print("-" * 50)
    _, g_losses_3, d_losses_3 = train_dcgan(
        generator, discriminator, dataloader,
        num_epochs=50,           # Fine-tuning phase
        lr_g=0.0001,            # Lower LR for fine-tuning
        lr_d=0.00005,           # Much lower discriminator LR
        device=device,
        patience=30,            # High patience for fine details
        min_delta=0.0001,       # Very small improvement threshold
        noise_std=0.01,         # Less noise for precision
        d_train_freq=2,         # Train discriminator less frequently
        g_train_freq=1,
        gradient_penalty_weight=8.0  # Higher penalty for stability
    )
    all_g_losses.extend(g_losses_3)
    all_d_losses.extend(d_losses_3)
    
    print("\n PROGRESSIVE TRAINING COMPLETED")
    print("="*60)
    
    return generator, all_g_losses, all_d_losses

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
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Get the dimensions from the one-hot patches
    sample = dataset[0]
    print(sample)
    n_tile_types = sample.size(0)
    patch_height = sample.size(1) 
    patch_width = sample.size(2)  
    
    print(f"Patch dimensions: {patch_height}x{patch_width} with {n_tile_types} tile types")
    
    # Create instances of the generator and discriminator
    generator = DCGANGenerator(latent_dim=256, n_tile_types=n_tile_types, 
                               patch_height=patch_height, patch_width=patch_width, hidden_dim=512)
    discriminator = DCGANDiscriminator(n_tile_types=n_tile_types, 
                                     patch_height=patch_height, patch_width=patch_width)

    # Train the generator
    trained_generator, g_losses, d_losses = progressive_training(
    generator, discriminator, dataloader, device
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