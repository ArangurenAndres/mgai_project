import torch
import os
import sys
import time
from dcgan_model import DCGANGenerator

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from utils.process_data import ProcessDataSymbolic

def load_trained_generator(model_path, latent_dim, n_tile_types, patch_height, patch_width, hidden_dim, device):
    """Load a trained DCGAN generator from saved state dict"""
    generator = DCGANGenerator(
        latent_dim=latent_dim, 
        n_tile_types=n_tile_types,
        patch_height=patch_height, 
        patch_width=patch_width, 
        hidden_dim=hidden_dim
    )
    
    # Load the saved state dict
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.to(device)
    generator.eval()  # Set to evaluation mode
    
    print(f"Loaded trained generator from: {model_path}")
    return generator

def generate_whole_level(generator, processor, num_patches_width=7, num_patches_height=14, device='cpu'):
    """Generate a complete level using the trained generator"""
    generator.eval()
    symbolic_level = []
    
    print(f"Generating level: {num_patches_height} x {num_patches_width} patches")
    
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
                    symbolic_patch = symbolic_patch[1]
                
                row_patches.append(symbolic_patch)
            
            # Combine patches horizontally for each row
            for i in range(len(row_patches[0])):
                symbolic_row = ''.join([patch[i] for patch in row_patches])
                symbolic_level.append(symbolic_row)
    
    return symbolic_level

def generate_multiple_levels(generator, processor, num_levels=5, level_width=7, level_height=1, 
                           output_dir=None, device='cpu', render_images=True):
    """Generate multiple levels and save them"""
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'generated_levels_batch')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating {num_levels} levels...")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    generated_levels = []
    
    for i in range(num_levels):
        print(f"\nGenerating level {i+1}/{num_levels}...")
        
        # Generate the level
        symbolic_level = generate_whole_level(
            generator, processor, level_width, level_height, device
        )
        
        # Create filename with timestamp and index
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        level_filename = f'dcgan_level_{i+1:03d}_{timestamp}.txt'
        level_path = os.path.join(output_dir, level_filename)
        
        # Save the level as text file
        with open(level_path, 'w') as f:
            for row in symbolic_level:
                f.write(row + '\n')
        
        print(f"Saved: {level_filename}")
        
        # Display level preview
        print("Level preview:")
        for j, row in enumerate(symbolic_level[:3]):  # Show first 3 rows
            print(f"   {row}")
        if len(symbolic_level) > 3:
            print(f"   ... ({len(symbolic_level)-3} more rows)")
        
        # Render as image if requested
        if render_images:
            try:
                tile_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'tiles'))
                rendered_img = processor.render_level_image(
                    symb_name=level_filename,
                    symb_file=symbolic_level,
                    tile_dir=tile_dir,
                    save_folder=output_dir
                )
                print(f"Rendered image saved")
            except Exception as e:
                print(f"Could not render image: {e}")
        
        generated_levels.append({
            'filename': level_filename,
            'path': level_path,
            'symbolic_data': symbolic_level
        })
        
        print("-" * 40)
    
    print(f"\nSuccessfully generated {len(generated_levels)} levels!")
    print(f"All files saved to: {output_dir}")
    
    return generated_levels

def generate_levels_with_variations(generator, processor, base_seed=None, num_variations=3, 
                                  level_width=7, level_height=1, output_dir=None, device='cpu'):
    """Generate levels with controlled variations using seeds"""
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'generated_levels_variations')
    
    os.makedirs(output_dir, exist_ok=True)
    
    if base_seed is None:
        base_seed = int(time.time())
    
    print(f"\nGenerating {num_variations} level variations...")
    print(f"Base seed: {base_seed}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    generated_levels = []
    
    for i in range(num_variations):
        # Set seed for reproducible generation
        seed = base_seed + i
        torch.manual_seed(seed)
        
        print(f"\nGenerating variation {i+1}/{num_variations} (seed: {seed})...")
        
        # Generate the level
        symbolic_level = generate_whole_level(
            generator, processor, level_width, level_height, device
        )
        
        # Create filename with seed
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        level_filename = f'dcgan_level_seed{seed}_{timestamp}.txt'
        level_path = os.path.join(output_dir, level_filename)
        
        # Save the level
        with open(level_path, 'w') as f:
            for row in symbolic_level:
                f.write(row + '\n')
        
        print(f"Saved: {level_filename}")
        
        # Display level preview
        print("Level preview:")
        for j, row in enumerate(symbolic_level[:2]):
            print(f"   {row}")
        
        generated_levels.append({
            'seed': seed,
            'filename': level_filename,
            'path': level_path,
            'symbolic_data': symbolic_level
        })
        
        print("-" * 40)
    
    print(f"\nSuccessfully generated {len(generated_levels)} level variations!")
    return generated_levels

if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters (should match your training configuration)
    MODEL_CONFIG = {
        'latent_dim': 256,
        'hidden_dim': 512,
        'patch_height': 14,  # Update these based on your actual patch dimensions
        'patch_width': 28,   # Update these based on your actual patch dimensions
        'n_tile_types': 10   # Update this based on your actual number of tile types
    }
    
    # Paths
    model_path = os.path.join(os.path.dirname(__file__), 'dcgan_mario_generator.pth')
    mapping_path = os.path.join(os.path.dirname(__file__), '..', 'utils', 'mapping.yaml')
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please make sure you have trained and saved a model first.")
        exit(1)
    
    # Initialize processor
    processor = ProcessDataSymbolic(mapping_path=mapping_path)
    
    # Load the trained generator
    generator = load_trained_generator(
        model_path=model_path,
        device=device,
        **MODEL_CONFIG
    )
    
    # Choose generation mode
    print("\nChoose generation mode:")
    print("1. Generate multiple random levels")
    print("2. Generate variations with controlled seeds")
    print("3. Generate single custom level")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        # Generate multiple random levels
        num_levels = int(input("How many levels to generate? (default: 5): ") or "5")
        level_width = int(input("Level width in patches? (default: 7): ") or "7")
        level_height = int(input("Level height in patches? (default: 1): ") or "1")
        
        generated_levels = generate_multiple_levels(
            generator=generator,
            processor=processor,
            num_levels=num_levels,
            level_width=level_width,
            level_height=level_height,
            device=device,
            render_images=True
        )
        
    elif choice == "2":
        # Generate variations with seeds
        num_variations = int(input("How many variations to generate? (default: 3): ") or "3")
        base_seed = input("Base seed (press Enter for random): ").strip()
        base_seed = int(base_seed) if base_seed else None
        
        generated_levels = generate_levels_with_variations(
            generator=generator,
            processor=processor,
            base_seed=base_seed,
            num_variations=num_variations,
            device=device
        )
        
    elif choice == "3":
        # Generate single custom level
        level_width = int(input("Level width in patches? (default: 7): ") or "7")
        level_height = int(input("Level height in patches? (default: 1): ") or "1")
        
        print(f"\nGenerating custom level ({level_height}x{level_width} patches)...")
        
        symbolic_level = generate_whole_level(
            generator, processor, level_width, level_height, device
        )
        
        # Save the level
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(os.path.dirname(__file__), 'generated_levels_custom')
        os.makedirs(output_dir, exist_ok=True)
        
        level_filename = f'dcgan_custom_level_{timestamp}.txt'
        level_path = os.path.join(output_dir, level_filename)
        
        with open(level_path, 'w') as f:
            for row in symbolic_level:
                f.write(row + '\n')
        
        print(f"Saved: {level_path}")
        print("\nGenerated level:")
        processor.visualize_file(symbolic_level)
        
        # Render image
        try:
            tile_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'tiles'))
            rendered_img = processor.render_level_image(
                symb_name=level_filename,
                symb_file=symbolic_level,
                tile_dir=tile_dir,
                save_folder=output_dir
            )
            rendered_img.show()
            print("Image displayed and saved")
        except Exception as e:
            print(f"Could not render image: {e}")
    
    else:
        print("Invalid choice. Please run the script again.")
    
    print("\n Level generation completed!")