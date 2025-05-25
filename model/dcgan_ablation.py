# hyperparameter_ablation.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import sys
import matplotlib.pyplot as plt
import time
import json
import numpy as np
from itertools import product
import seaborn as sns
import pandas as pd

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dcgan_model import DCGANGenerator, DCGANDiscriminator
from dcgan_train import MarioLevelDataset, generate_whole_level
from utils.process_data import ProcessDataSymbolic

class HyperparameterAblation:
    def __init__(self, dataset, processor, device='cpu'):
        self.dataset = dataset
        self.processor = processor
        self.device = device
        self.results = []
        
        # Get data dimensions
        sample = dataset[0]
        self.n_tile_types = sample.size(0)
        self.patch_height = sample.size(1)
        self.patch_width = sample.size(2)
        
        print(f"Dataset info: {len(dataset)} patches, {self.patch_height}x{self.patch_width}, {self.n_tile_types} tile types")
        
    def train_single_config(self, config, config_id):
        """Train a single configuration and return results"""
        print(f"\n{'='*60}")
        print(f"Training Configuration {config_id}")
        print(f"Config: {config}")
        print(f"{'='*60}")
        
        # Create fresh models for each config
        generator = DCGANGenerator(
            latent_dim=config['latent_dim'], 
            n_tile_types=self.n_tile_types,
            patch_height=self.patch_height, 
            patch_width=self.patch_width
        )
        discriminator = DCGANDiscriminator(
            n_tile_types=self.n_tile_types,
            patch_height=self.patch_height, 
            patch_width=self.patch_width
        )
        
        # Create dataloader with specified batch size
        dataloader = DataLoader(self.dataset, batch_size=config['batch_size'], shuffle=True)
        
        # Train the model
        start_time = time.time()
        trained_generator, g_losses, d_losses = self.train_dcgan_ablation(
            generator, discriminator, dataloader, config
        )
        training_time = time.time() - start_time
        
        # Evaluate the results
        metrics = self.evaluate_model(trained_generator, g_losses, d_losses, config, training_time)
        
        # Store results
        result = {
            'config_id': config_id,
            'config': config.copy(),
            'metrics': metrics,
            'g_losses': g_losses,
            'd_losses': d_losses
        }
        self.results.append(result)
        
        # Save intermediate results
        self.save_intermediate_results(config_id)
        
        return result
    
    def train_dcgan_ablation(self, generator, discriminator, dataloader, config):
        """Modified training function for ablation study"""
        generator.to(self.device)
        discriminator.to(self.device)

        criterion = nn.BCELoss()
        optimizer_g = optim.Adam(generator.parameters(), lr=config['lr_g'], betas=(0.5, 0.999))
        optimizer_d = optim.Adam(discriminator.parameters(), lr=config['lr_d'], betas=(0.5, 0.999))

        # Label smoothing
        real_label = config['real_label']
        fake_label = config['fake_label']

        g_losses = []
        d_losses = []
        
        # Early stopping
        best_g_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(config['max_epochs']):
            current_g_loss = 0.0
            current_d_loss = 0.0
            num_batches = 0
            
            for i, real_data in enumerate(dataloader):
                batch_size = real_data.size(0)
                real_data = real_data.to(self.device)
                
                # Add noise to real data if specified
                if config['noise_std'] > 0:
                    noise = torch.randn_like(real_data) * config['noise_std']
                    real_data = torch.clamp(real_data + noise, 0, 1)
                
                # Train discriminator with specified frequency
                if i % config['d_train_freq'] == 0:
                    discriminator.zero_grad()
                    
                    # Real data
                    labels = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device)
                    output = discriminator(real_data)
                    loss_d_real = criterion(output, labels)
                    loss_d_real.backward()

                    # Fake data
                    z = torch.randn(batch_size, generator.latent_dim, device=self.device)
                    fake_data = generator(z)
                    labels.fill_(fake_label)
                    output = discriminator(fake_data.detach())
                    loss_d_fake = criterion(output, labels)
                    loss_d_fake.backward()
                    optimizer_d.step()

                    loss_d = loss_d_real + loss_d_fake
                else:
                    z = torch.randn(batch_size, generator.latent_dim, device=self.device)
                    fake_data = generator(z)
                    loss_d = torch.tensor(0.0)

                # Train generator with specified frequency
                if i % config['g_train_freq'] == 0:
                    generator.zero_grad()
                    labels = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device)
                    output = discriminator(fake_data)
                    loss_g = criterion(output, labels)
                    loss_g.backward()
                    optimizer_g.step()
                else:
                    loss_g = torch.tensor(0.0)

                current_g_loss += loss_g.item() if isinstance(loss_g, torch.Tensor) else loss_g
                current_d_loss += loss_d.item() if isinstance(loss_d, torch.Tensor) else loss_d
                num_batches += 1

            # Calculate average losses
            avg_g_loss = current_g_loss / num_batches
            avg_d_loss = current_d_loss / num_batches
            
            g_losses.append(avg_g_loss)
            d_losses.append(avg_d_loss)
            
            # Early stopping check
            if avg_g_loss < best_g_loss - config['min_delta']:
                best_g_loss = avg_g_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Print progress every 10 epochs
            if epoch % 10 == 0 or epoch < 5:
                print(f"Epoch [{epoch+1}/{config['max_epochs']}] - D: {avg_d_loss:.4f}, G: {avg_g_loss:.4f}")
            
            # Early stopping
            if epochs_without_improvement >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
            # Check for mode collapse
            if avg_d_loss > 50 or avg_g_loss < 0.001:
                print(f"Potential mode collapse detected at epoch {epoch+1}")
                break

        return generator, g_losses, d_losses
    
    def evaluate_model(self, generator, g_losses, d_losses, config, training_time):
        """Evaluate the trained model and return metrics"""
        metrics = {}
        
        # Training metrics
        metrics['final_g_loss'] = g_losses[-1] if g_losses else float('inf')
        metrics['final_d_loss'] = d_losses[-1] if d_losses else float('inf')
        metrics['min_g_loss'] = min(g_losses) if g_losses else float('inf')
        metrics['training_epochs'] = len(g_losses)
        metrics['training_time'] = training_time
        
        # Stability metrics
        if len(g_losses) > 10:
            metrics['g_loss_std'] = np.std(g_losses[-10:])  # Stability in last 10 epochs
            metrics['d_loss_std'] = np.std(d_losses[-10:])
        else:
            metrics['g_loss_std'] = float('inf')
            metrics['d_loss_std'] = float('inf')
        
        # Mode collapse detection
        metrics['mode_collapse'] = (
            metrics['final_d_loss'] > 50 or 
            metrics['final_g_loss'] < 0.001 or
            metrics['g_loss_std'] < 0.001
        )
        
        # Generate sample levels for quality assessment
        try:
            # Fixed level height = 1, variable width
            level_width = 7  # Example width
            sample_level = generate_whole_level(generator, self.processor, level_width, 1, self.device)
            metrics['generation_success'] = True
            
            # Simple diversity metric: count unique characters
            level_str = ''.join(sample_level)
            unique_chars = len(set(level_str))
            metrics['character_diversity'] = unique_chars
            
            # Check for repetitive patterns (simple heuristic)
            metrics['repetition_score'] = self.calculate_repetition_score(sample_level)
            
        except Exception as e:
            print(f"Generation failed: {e}")
            metrics['generation_success'] = False
            metrics['character_diversity'] = 0
            metrics['repetition_score'] = 1.0
        
        # Overall quality score (lower is better)
        quality_score = 0
        if metrics['mode_collapse']:
            quality_score += 1000
        if not metrics['generation_success']:
            quality_score += 500
        
        quality_score += metrics['final_g_loss'] * 10
        quality_score += metrics['g_loss_std'] * 5
        quality_score += (1.0 - metrics['repetition_score']) * 100  # Penalize repetition
        quality_score -= metrics['character_diversity'] * 2  # Reward diversity
        
        metrics['quality_score'] = quality_score
        
        return metrics
    
    def calculate_repetition_score(self, level):
        """Calculate how repetitive a level is (0 = very repetitive, 1 = very diverse)"""
        if not level:
            return 0
        
        # Check for repeated patterns in rows
        unique_rows = len(set(level))
        total_rows = len(level)
        row_diversity = unique_rows / total_rows if total_rows > 0 else 0
        
        # Check for repeated patterns in columns (sample a few columns)
        col_diversity = 0
        if level and len(level[0]) > 0:
            num_cols_to_check = min(10, len(level[0]))
            unique_cols = 0
            for col_idx in range(0, len(level[0]), max(1, len(level[0]) // num_cols_to_check)):
                col = ''.join([row[col_idx] if col_idx < len(row) else '' for row in level])
                if col:
                    unique_cols += 1
            col_diversity = unique_cols / num_cols_to_check if num_cols_to_check > 0 else 0
        
        return (row_diversity + col_diversity) / 2
    
    def run_ablation_study(self):
        """Run the complete ablation study"""
        print("Starting Hyperparameter Ablation Study")
        print("="*60)
        
        # Define hyperparameter search space
        hyperparameters = {
            'lr_g': [0.0001, 0.0002, 0.0005],
            'lr_d': [0.0001, 0.0002, 0.0005],
            'batch_size': [16, 32, 64],
            'latent_dim': [64, 100, 128],
            'max_epochs': [50],  # Fixed to save time
            'patience': [10, 15],
            'min_delta': [0.001, 0.005, 0.01],
            'real_label': [0.9, 1.0],
            'fake_label': [0.0, 0.1],
            'noise_std': [0.0, 0.02],
            'd_train_freq': [1, 2],  # Train discriminator every N batches
            'g_train_freq': [1],     # Train generator every N batches
        }
        
        # Generate all combinations (this will be a lot!)
        # Let's sample a subset for practical reasons
        all_combinations = list(product(*hyperparameters.values()))
        
        # Sample a manageable number of combinations
        max_experiments = 50  # Adjust based on your computational budget
        if len(all_combinations) > max_experiments:
            import random
            random.seed(42)
            selected_combinations = random.sample(all_combinations, max_experiments)
        else:
            selected_combinations = all_combinations
        
        print(f"Testing {len(selected_combinations)} hyperparameter combinations")
        
        successful_experiments = 0
        failed_experiments = 0
        
        # Run experiments
        for i, combination in enumerate(selected_combinations):
            config = dict(zip(hyperparameters.keys(), combination))
            
            # Skip invalid combinations
            if config['lr_d'] > config['lr_g'] * 2:  # Discriminator shouldn't be much faster
                print(f"Skipping invalid configuration {i+1}: lr_d too high relative to lr_g")
                continue
            
            try:
                print(f"\nStarting experiment {i+1}/{len(selected_combinations)}")
                self.train_single_config(config, i+1)
                successful_experiments += 1
                print(f"✓ Configuration {i+1} completed successfully")
                
            except Exception as e:
                failed_experiments += 1
                print(f"✗ Configuration {i+1} failed: {str(e)}")
                
                # Save error info
                error_result = {
                    'config_id': i+1,
                    'config': config,
                    'error': str(e),
                    'metrics': {
                        'generation_success': False,
                        'mode_collapse': True,
                        'quality_score': float('inf'),
                        'final_g_loss': float('inf'),
                        'final_d_loss': float('inf'),
                        'min_g_loss': float('inf'),
                        'training_epochs': 0,
                        'training_time': 0.0,
                        'g_loss_std': float('inf'),
                        'd_loss_std': float('inf'),
                        'character_diversity': 0,
                        'repetition_score': 0.0
                    },
                    'g_losses': [],
                    'd_losses': []
                }
                self.results.append(error_result)
                continue
            
        print(f"\nAblation study completed!")
        print(f"Successful experiments: {successful_experiments}")
        print(f"Failed experiments: {failed_experiments}")
        
        # Analyze results
        if successful_experiments > 0:
            self.analyze_results()
        else:
            print("No successful experiments to analyze!")
    
    def run_quick_ablation_study(self):
        """Run a smaller ablation study for quick testing"""
        print("Starting Quick Hyperparameter Ablation Study")
        print("="*60)
        
        # Smaller search space for quick testing
        hyperparameters = {
            'lr_g': [0.0001, 0.0002],
            'lr_d': [0.0001, 0.0002],
            'batch_size': [32],
            'latent_dim': [100],
            'max_epochs': [20],
            'patience': [10],
            'min_delta': [0.005],
            'real_label': [0.9, 1.0],
            'fake_label': [0.0],
            'noise_std': [0.0, 0.02],
            'd_train_freq': [1, 2],
            'g_train_freq': [1],
        }
        
        # Generate all combinations
        all_combinations = list(product(*hyperparameters.values()))
        print(f"Testing {len(all_combinations)} hyperparameter combinations")
        
        successful_experiments = 0
        
        # Run experiments
        for i, combination in enumerate(all_combinations):
            config = dict(zip(hyperparameters.keys(), combination))
            
            try:
                print(f"\nStarting experiment {i+1}/{len(all_combinations)}")
                self.train_single_config(config, i+1)
                successful_experiments += 1
                
            except Exception as e:
                print(f"Configuration {i+1} failed: {str(e)}")
                continue
        
        print(f"\nQuick ablation study completed with {successful_experiments} successful experiments!")
        
        if successful_experiments > 0:
            self.analyze_results()
            
    def save_intermediate_results(self, config_id):
        """Save results after each experiment"""
        results_dir = os.path.join(os.path.dirname(__file__), 'ablation_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results
        with open(os.path.join(results_dir, f'results_{config_id}.json'), 'w') as f:
            # Convert numpy arrays and other non-serializable types to JSON-compatible formats
            result_copy = self.results[-1].copy()
            
            # Convert losses to lists
            result_copy['g_losses'] = [float(x) for x in result_copy['g_losses']]
            result_copy['d_losses'] = [float(x) for x in result_copy['d_losses']]
            
            # Convert metrics to JSON-serializable types
            metrics_copy = {}
            for key, value in result_copy['metrics'].items():
                if isinstance(value, (np.bool_, bool)):
                    metrics_copy[key] = bool(value)
                elif isinstance(value, (np.integer, np.int64, np.int32)):
                    metrics_copy[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    metrics_copy[key] = float(value)
                elif isinstance(value, np.ndarray):
                    metrics_copy[key] = value.tolist()
                else:
                    metrics_copy[key] = value
            
            result_copy['metrics'] = metrics_copy
            
            # Convert config to JSON-serializable types
            config_copy = {}
            for key, value in result_copy['config'].items():
                if isinstance(value, (np.bool_, bool)):
                    config_copy[key] = bool(value)
                elif isinstance(value, (np.integer, np.int64, np.int32)):
                    config_copy[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    config_copy[key] = float(value)
                else:
                    config_copy[key] = value
            
            result_copy['config'] = config_copy
            
            json.dump(result_copy, f, indent=2)
    
    def analyze_results(self):
        """Analyze and visualize the ablation study results"""
        if not self.results:
            print("No results to analyze!")
            return
        
        print("\n" + "="*60)
        print("ABLATION STUDY RESULTS")
        print("="*60)
        
        # Create results directory
        results_dir = os.path.join(os.path.dirname(__file__), 'ablation_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Convert results to DataFrame for easier analysis
        df_data = []
        for result in self.results:
            row = result['config'].copy()
            row.update(result['metrics'])
            row['config_id'] = result['config_id']
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Save full results
        df.to_csv(os.path.join(results_dir, 'full_results.csv'), index=False)
        
        # Find best configurations
        valid_results = df[~df['mode_collapse'] & df['generation_success']]
        
        if len(valid_results) > 0:
            best_by_quality = valid_results.loc[valid_results['quality_score'].idxmin()]
            best_by_g_loss = valid_results.loc[valid_results['min_g_loss'].idxmin()]
            best_by_diversity = valid_results.loc[valid_results['character_diversity'].idxmax()]
            
            print("\nBEST CONFIGURATIONS:")
            print("-" * 40)
            print(f"Best Overall Quality (Score: {best_by_quality['quality_score']:.2f}):")
            self.print_config(best_by_quality)
            
            print(f"\nBest Generator Loss ({best_by_g_loss['min_g_loss']:.4f}):")
            self.print_config(best_by_g_loss)
            
            print(f"\nBest Diversity ({best_by_diversity['character_diversity']} unique chars):")
            self.print_config(best_by_diversity)
        else:
            print("No valid configurations found! All suffered from mode collapse or generation failure.")
        
        # Create visualizations
        self.create_visualizations(df, results_dir)
        
        # Print summary statistics
        self.print_summary_stats(df)
    
    def print_config(self, config_row):
        """Print a configuration in a readable format"""
        important_params = ['lr_g', 'lr_d', 'batch_size', 'latent_dim', 'patience', 'min_delta', 
                          'real_label', 'fake_label', 'noise_std', 'd_train_freq']
        for param in important_params:
            if param in config_row:
                print(f"  {param}: {config_row[param]}")
    
    def create_visualizations(self, df, results_dir):
        """Create various plots to analyze the results"""
        
        # 1. Quality score vs hyperparameters
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hyperparameter Effects on Quality Score', fontsize=16)
        
        params_to_plot = ['lr_g', 'lr_d', 'batch_size', 'latent_dim', 'patience', 'min_delta']
        
        for i, param in enumerate(params_to_plot):
            ax = axes[i//3, i%3]
            if param in df.columns:
                df_valid = df[df['generation_success'] & ~df['mode_collapse']]
                if len(df_valid) > 0:
                    df_valid.boxplot(column='quality_score', by=param, ax=ax)
                    ax.set_title(f'Quality Score vs {param}')
                    ax.set_xlabel(param)
                    ax.set_ylabel('Quality Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'hyperparameter_effects.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Training curves for best configurations
        valid_results = df[~df['mode_collapse'] & df['generation_success']]
        if len(valid_results) > 0:
            best_configs = valid_results.nsmallest(5, 'quality_score')
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            for _, config in best_configs.iterrows():
                config_id = int(config['config_id'])
                result = next(r for r in self.results if r['config_id'] == config_id)
                
                epochs = range(1, len(result['g_losses']) + 1)
                ax1.plot(epochs, result['g_losses'], label=f"Config {config_id}", alpha=0.7)
                ax2.plot(epochs, result['d_losses'], label=f"Config {config_id}", alpha=0.7)
            
            ax1.set_title('Generator Loss - Best Configurations')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Generator Loss')
            ax1.legend()
            ax1.grid(True)
            
            ax2.set_title('Discriminator Loss - Best Configurations')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Discriminator Loss')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'best_training_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Hyperparameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved to {results_dir}")
    
    def print_summary_stats(self, df):
        """Print summary statistics"""
        print("\nSUMMARY STATISTICS:")
        print("-" * 40)
        print(f"Total experiments: {len(df)}")
        print(f"Successful generations: {df['generation_success'].sum()}")
        print(f"Mode collapse cases: {df['mode_collapse'].sum()}")
        print(f"Average training time: {df['training_time'].mean():.2f}s")
        
        valid_df = df[df['generation_success'] & ~df['mode_collapse']]
        if len(valid_df) > 0:
            print(f"\nValid experiments: {len(valid_df)}")
            print(f"Best quality score: {valid_df['quality_score'].min():.2f}")
            print(f"Best generator loss: {valid_df['min_g_loss'].min():.4f}")
            print(f"Average character diversity: {valid_df['character_diversity'].mean():.2f}")

def main():
    """Main function to run the ablation study"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data (same as your main training script)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    mapping_path = os.path.join(parent_dir, 'utils', 'mapping.yaml')
    
    processor = ProcessDataSymbolic(mapping_path=mapping_path)
    symb_data_folder = processor.folder_path
    symbolic_files = [f for f in os.listdir(symb_data_folder) if f.endswith('.txt')]

    all_one_hot_patches = []
    
    print("Loading and processing data...")
    for symb_file in symbolic_files:
        processor.load_symbolic(symb_file)
        patches = processor.crop_symbolic()
        for patch in patches:
            _, onehot_file = processor.forward_mapping_onehot(patch)
            all_one_hot_patches.append(onehot_file)
    
    print(f"Loaded {len(all_one_hot_patches)} patches.")
    
    # Create dataset
    dataset = MarioLevelDataset(all_one_hot_patches)
    
    # Run ablation study
    ablation = HyperparameterAblation(dataset, processor, device)
    
    # Choose which version to run
    print("Choose ablation study type:")
    print("1. Quick study (8 experiments)")
    print("2. Full study (50 experiments)")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        ablation.run_quick_ablation_study()
    else:
        ablation.run_ablation_study()

if __name__ == "__main__":
    main()