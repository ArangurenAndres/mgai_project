import os
import torch
import numpy as np
import re
import random
from torch.utils.data import Dataset, DataLoader
from .process_data import ProcessDataSymbolic  # Adjust import if needed
from .load_files import load_config


# Natural sorting for file names like "mario_1_1.txt"
def sort_files(text):
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r'(\d+)', text)]


class MarioLevelDataset(Dataset):
    def __init__(self, patches, processor):
        """
        Args:
            patches (list): symbolic patches from processor.crop_symbolic()
            processor (ProcessDataSymbolic): processor object for forward mapping
        """
        self.processor = processor
        self.patches = patches
        self.data = self._build_dataset()

    def _build_dataset(self): # function to build the dataset using the symbolic patches
        all_vectors = []
        for patch in self.patches:
            _, vector = self.processor.forward_mapping(patch)  # (H, W, D)
            vector = vector.astype(np.float32)
            vector = np.transpose(vector, (2, 0, 1))  # Convert to (C, H, W)
            all_vectors.append(torch.tensor(vector))
        return all_vectors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    root_dir = os.path.dirname(__file__)
    mapping_path = os.path.abspath(os.path.join(root_dir, 'mapping.yaml'))
    config_path = os.path.abspath(os.path.join(root_dir,".." ,'config.yaml'))
    symb_folder = os.path.abspath(os.path.join(root_dir, '..', 'data', 'symbol'))
    files = sorted(os.listdir(symb_folder), key=sort_files)

    config_file = load_config(config_path=config_path)   #Load config file


    #======== DATALOADER PARAMETERS ================================
    embedding_dim = config_file["train"]["embedding_dim"] # embedding dimension for each patch
    batch_size = config_file["train"]["batch_size"] # batch size

    #================================================================

    processor = ProcessDataSymbolic(embedding_dim=embedding_dim,mapping_path=mapping_path) #Initalize the processor

    test_level = random.choice(files)   # Select and process one random level file
    print(f"Processing level", {test_level})
    symb_file = processor.load_symbolic(test_level,visualize=False) # Load symbolic file
    patches = processor.crop_symbolic() # Obtain patches of symbolic files

    # Dataset and DataLoader
    dataset = MarioLevelDataset(patches=patches, processor=processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # make it an iterable dataloader
    print(f"\n Dataset contains {len(dataset)} samples.")

    # Iterate through DataLoader
    for batch_idx, batch in enumerate(dataloader):
        b,c,h,w = batch.shape
        try:
            assert b==batch_size
        except AssertionError:
            print(f"[WARNING] Batch {batch_idx} has unexpected batch size: {b} (expected {batch_size})")

        print(f"\nBatch {batch_idx} — Shape: {batch.shape}")  # (B, C, H, W)
        #print(f"First sample (channel 0):\n{batch[0][0]}")

        if batch_idx == 1:
            break

