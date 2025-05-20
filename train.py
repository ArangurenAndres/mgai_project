import os
import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.process_data import ProcessDataSymbolic
from utils.data_loader import MarioLevelDataset
from utils.load_files import load_config
from model.denoise_net import SimpleDenoiseNet
from model.diffusion import GaussianDiffusion


def train(config_path:str=None, mapping_path:str=None):
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    symb_folder = config["data_process"]["folder_path"] # Symbolic data path
    embedding_dim = config["data_process"]["embedding_dim"] # Embedding dimension

    # Load symbolic levels and crop patches
    processor = ProcessDataSymbolic(config_path=config_path, mapping_path=mapping_path)
    all_patches = [] # Store all the patches
    for fname in sorted(os.listdir(symb_folder)):
        if fname.endswith('.txt'):
            print(f"Loading {fname}")
            processor.load_symbolic(fname)
            all_patches.extend(processor.crop_symbolic())
    print(f"Total patches: {len(all_patches)}")

    # Dataset and loader
    dataset = MarioLevelDataset(patches=all_patches, processor=processor)
    loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True)

    # Model and diffusion setup
    #model = SimpleDenoiseNet(in_channels=embedding_dim, time_dim=config["model"]["time_emb_dim"]).to(device)
    #diffusion = GaussianDiffusion(
    #    model,
    #    timesteps=config["model"]["diffusion_steps"],
    #    beta_start=config["model"]["beta_start"],
    #    beta_end=config["model"]["beta_end"],
     #   device=device
    #)

    #optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

    # Training loop with tqdm
    #for epoch in range(config["train"]["epochs"]):
    #    total_loss = 0
     #   loader_tqdm = tqdm(loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}", leave=False)
#
      #  for batch_idx, batch in enumerate(loader_tqdm):
     #       batch = batch.to(device)
      #      loss = diffusion.train_step(batch, optimizer)
      #      total_loss += loss
#
       #     if (batch_idx + 1) % config["train"]["print_every"] == 0:
       #         loader_tqdm.set_postfix(loss=f"{loss:.4f}")

      # avg_loss = total_loss / len(loader)
       # print(f">>> Epoch {epoch+1} finished — Avg Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    config_path = "config.yaml"
    mapping_path = "utils/mapping.yaml"
    train(config_path=config_path,mapping_path=mapping_path)
