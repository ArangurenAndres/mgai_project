import os
import torch
import numpy as np
import re
import random
from torch.utils.data import Dataset, DataLoader
from utils.process_data import ProcessDataSymbolic  # assumes this file is called process_data.py


# read the files in the correct level game order
def sort_files(text):
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r'(\d+)', text)]

class MarioLevelDataset(Dataset):
    def __init__(self, patches, processor, num_classes=10):
        """
        Args:
            patches (list): symbolic patches from processor.crop_symbolic()
            processor (ProcessDataSymbolic): processor object for forward mapping
            num_classes (int): number of tile types
        """
        self.processor = processor
        self.patches = patches
        self.num_classes = num_classes
        self.data = self._build_dataset()

    def _build_dataset(self):
        all_tensors = []
        for patch in self.patches:
            _, vector = self.processor.forward_mapping(patch)  # shape: (H, W, C)
            vector = vector.astype('float32')
            vector = np.transpose(vector, (2, 0, 1))  # (C, H, W)
            all_tensors.append(torch.tensor(vector))
        return all_tensors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    # Config paths (adjust if needed)
    root_dir = os.path.dirname(__file__)
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"..", 'config.yaml'))
    mapping_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mapping.yaml'))
    symb_folder =  symb_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'symbol'))
    files = sorted(os.listdir(symb_folder), key=sort_files)

    
    all_patches  = []
    
    processor = ProcessDataSymbolic(config_path=config_path, mapping_path=mapping_path) # Initialize processing class
    # Iterate over one level

    test_level = random.choice(files)
    print(f"Processing leve: {test_level}")
    processor.load_symbolic(test_level) # Load the level
    patches = processor.crop_symbolic() # Obtain level patches
    print(f"  Extracted {len(patches)} patches")
    dataset = MarioLevelDataset(patches = patches,processor=processor)
    dataloader = DataLoader(dataset,batch_size=4,shuffle=True) # Create dataloader

    # Iterate over data loader
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx} — Shape: {batch.shape}") # (B, C, H, W)
        print(f"First sample (channel 0):\n{batch[0][0]}")
        if batch_idx==1:
            break


