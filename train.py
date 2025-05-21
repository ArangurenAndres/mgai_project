import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.process_data import ProcessDataSymbolic
from utils.data_loader import MarioLevelDataset
from utils.load_files import load_config, load_mapping
from model.denoise_net import SimpleDenoiseNet
from model.diffusion import GaussianDiffusion


def collect_patches(processor:ProcessDataSymbolic, data_folder:str=None):
    all_patches = []
    for fname in sorted(os.listdir(data_folder)):
      if fname.endswith(".txt"):
          print(f"[INFO] Loading {fname}")
          processor.load_symbolic(fname)
          all_patches.extend(processor.crop_symbolic())
    print(f"[INFO] Total patches: {len(all_patches)}")
    return all_patches


class Trainer():
    def __init__(self, config_path, mapping_path):

        self.mapping_path = mapping_path 
        self.config_file = load_config(config_path) # load config file for training
        self.mapping_file = load_mapping(mapping_path)

        self.embeding_dim = self.config_file["train"]["embedding_dim"]
        self.batch_size = self.config_file["train"]["batch_size"]
        self.l_rate = self.config_file["train"]["learning_rate"]

        ## Model parameters
        self.time_emb_dim = self.config_file["model"]["time_emb_dim"]
        self.depth = self.config_file["model"]["depth"]
        ## Difussion parameters
        self.diffusion_steps = self.config_file["model"]["diffusion_steps"]
        self.beta_start = self.config_file["model"]["beta_start"]
        self.beta_end = self.config_file["model"]["beta_end"]
    
    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #set device
        #==== 1. COLLECT ALL PATCHES FOR N LEVELS IN DATA FOLDER =======
        if self.mapping_path is not None:
            processor = ProcessDataSymbolic(mapping_path=self.mapping_path) # Initialize data processor
            patches = collect_patches(processor=processor, data_folder=self.mapping_file["data_process"]["folder_path"])
            #===== 2. DATALOADER ==================
            dataset = MarioLevelDataset(patches=patches, processor=processor)
            loader = DataLoader(dataset, batch_size=self.config_file["train"]["batch_size"], shuffle=True) # should i shuffle? maybe not

            # ====3. SET MODEL =====================
            model = SimpleDenoiseNet(in_channels=self.embedding_dim,
                                 time_dim=self.time_emb_dim,
                                 depth=self.depth).to(device) # denoise network
            
            diffusion = GaussianDiffusion(model,
                                          timesteps=self.diffusion_steps,
                                          beta_start=self.beta_start,
                                          beta_end=self.beta_end,
                                          device=device
                                          ) # diffusion model
            # =====4. OPTIMIZER ======================
            optimizer = torch.optim.Adam(model.parameters(), lr=self.l_rate)

            
            

def train(data_folder:str, config_path: str = None,mapping_path: str = None):
    # -------------------- Load config and device --------------------
    config = load_config(config_path) #Load config file


    embedding_dim = config["data_process"]["embedding_dim"]

    # -------------------- Load and process data ---------------------
    processor = ProcessDataSymbolic(mapping_path=mapping_path)


    dataset = MarioLevelDataset(patches=all_patches, processor=processor)
    loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True)

    # -------------------- Model & Diffusion ------------------------
    model = SimpleDenoiseNet(
        in_channels=embedding_dim,
        time_dim=config["model"]["time_emb_dim"],
        depth=config["model"].get("depth", 5)
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        timesteps=config["model"]["diffusion_steps"],
        beta_start=config["model"]["beta_start"],
        beta_end=config["model"]["beta_end"],
        device=device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])

    # -------------------- Training Loop ----------------------------
    for epoch in range(config["train"]["epochs"]):
        total_loss = 0
        loader_tqdm = tqdm(loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}", leave=False)

        for batch_idx, batch in enumerate(loader_tqdm):
            batch = batch.to(device)

            loss = diffusion.train_step(batch, optimizer)
            total_loss += loss

            if (batch_idx + 1) % config["train"]["print_every"] == 0:
                loader_tqdm.set_postfix(loss=f"{loss:.4f}")

        avg_loss = total_loss / len(loader)
        print(f">>> Epoch {epoch+1} finished — Avg Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    config_path = "config.yaml"
    mapping_path = "utils/mapping.yaml"
    train(config_path=config_path, mapping_path=mapping_path)
