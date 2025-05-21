import os
import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.process_data import ProcessDataSymbolic
from utils.data_loader import MarioLevelDataset
from utils.load_files import load_config, load_mapping
from model.denoise_net import SimpleDenoiseNet
from model.diffusion import GaussianDiffusion



def save_loss(loss:list=[],exp_path:str=None):
    loss_path = os.path.join(exp_path,"loss_epochs.json")
    with open(loss_path, "w") as f:
        json.dump(loss, f, indent=2)

def collect_patches(processor: ProcessDataSymbolic, data_folder: str = None):
    all_patches = []
    for fname in sorted(os.listdir(data_folder)):
        if fname.endswith(".txt"):
            print(f"[INFO] Loading {fname}")
            processor.load_symbolic(fname)
            all_patches.extend(processor.crop_symbolic())
    print(f"[INFO] Total patches: {len(all_patches)}")
    return all_patches


class Trainer:
    def __init__(self, data_path:str=None, config_path:str=None, mapping_path:str=None, save_path:str=None):
        self.data_path = data_path
        self.mapping_path = mapping_path
        self.config_file = load_config(config_path)
        self.mapping_file = load_mapping(mapping_path)
        self.save_path = save_path or "checkpoints"
        os.makedirs(self.save_path, exist_ok=True)

        self.batch_size = self.config_file["train"]["batch_size"]
        self.l_rate = self.config_file["train"]["learning_rate"]
        self.epochs = self.config_file["train"]["epochs"]
        self.print_interval = self.config_file["train"]["print_interval"]

        # Model parameters
        self.time_emb_dim = self.config_file["model"]["time_emb_dim"]
        self.depth = self.config_file["model"]["depth"]
        self.diffusion_steps = self.config_file["model"]["diffusion_steps"]
        self.beta_start = self.config_file["model"]["beta_start"]
        self.beta_end = self.config_file["model"]["beta_end"]

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # === 1. Load patches ===
        processor = ProcessDataSymbolic(mapping_path=self.mapping_path)
        print("1. Creating data patches...")
        patches = collect_patches(processor=processor, data_folder=self.data_path)

        # === 2. Create dataset & loader ===
        print("2. Creating dataloader...")
        dataset = MarioLevelDataset(patches=patches, processor=processor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # === 3. Automatically detect embedding_dim from dataset ===
        example_tensor = dataset[0]  # shape: (C, H, W)
        embedding_dim = example_tensor.shape[0]
        print(f"[INFO] Detected embedding_dim (channels): {embedding_dim}")

        print("3. Setting up models...")
        model = SimpleDenoiseNet(
            in_channels=embedding_dim,
            time_dim=self.time_emb_dim,
            depth=self.depth
        ).to(device)

        diffusion = GaussianDiffusion(
            model,
            timesteps=self.diffusion_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            device=device
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.l_rate)

        # === 4. Training loop ===
        print(f"4. Starting training over {self.epochs} epochs...")
        loss_epochs = []
        for epoch in range(self.epochs):
            total_loss = 0
            loader_tqdm = tqdm(loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)

            for batch_idx, batch in enumerate(loader_tqdm):
                batch = batch.to(device)
                loss = diffusion.train_step(batch, optimizer)
                total_loss += loss

                if (batch_idx + 1) % self.print_interval == 0:
                    loader_tqdm.set_postfix(loss=f"{loss:.4f}")

            avg_loss = total_loss / len(loader)
            loss_epochs.append(avg_loss)
            print(f">>> Epoch {epoch+1} finished — Avg Loss: {avg_loss:.4f}")
        # ==== 5. SAVE MODEL =====
        # As of now we are saving the last epoch model we have to include patience criteria
        save_file = os.path.join(self.save_path, "model_test.pt")
        torch.save(model.state_dict(), save_file)
        print(f" Model saved to: {save_file}")
        return loss_epochs

if __name__ == "__main__":
    data_folder = "data/symbol"
    data_path = os.path.abspath(data_folder)
    config_path = "config.yaml"
    mapping_path = "utils/mapping.yaml"
    save_path = "results/trained_models"

    trainer = Trainer(data_path=data_path, config_path=config_path, mapping_path=mapping_path)
    loss_epochs = trainer.train()
