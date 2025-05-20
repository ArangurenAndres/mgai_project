import numpy as np
import os
from PIL import Image
from load_files import load_mapping

class ProcessDataSymbolic:
    def __init__(self, mapping_path: str = None):
        self.mapping = load_mapping(mapping_path)

        # Load config directly from mapping.yaml
        data_config = self.mapping["data_process"]
        self.folder_path = data_config["folder_path"]
        if not os.path.isabs(self.folder_path):
            config_dir = os.path.dirname(os.path.abspath(mapping_path))
            self.folder_path = os.path.abspath(os.path.join(config_dir, self.folder_path))

        self.window_dim = data_config["sliding_window"]
        self.stride = data_config.get("stride", 1)
        self.embedding_dim = data_config.get("embedding_dim", 32)

        # Load mappings
        self.symbol_to_id = self.mapping["symbol_identity"]
        self.id_to_symbol = {v: k for k, v in self.symbol_to_id.items()}
        self.tile_filenames = self.mapping.get("tile_filenames", {})

        self.n_classes = len(self.symbol_to_id)
        self.embedding_matrix = np.random.randn(self.n_classes, self.embedding_dim).astype(np.float32)

    # ================= FORWARD OPERATIONS ======================
    def load_symbolic(self, img_name: str = None):
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")
        img_path = os.path.join(self.folder_path, img_name)
        with open(img_path, 'r') as f:
            lines = [line.rstrip('\n') for line in f]
        self.lines = lines
        for row in lines:
            print(row)
        return lines

    def crop_symbolic(self):
        win_h, win_w = self.window_dim
        file_height = len(self.lines)
        file_width = len(self.lines[0])
        if file_height < win_h:
            raise ValueError(f"File height {file_height} is less than window height {win_h}")
        patches = []
        for x in range(0, file_width - win_w + 1, self.stride):
            patch = [line[x:x + win_w] for line in self.lines[:win_h]]
            patches.append(patch)
        print(f"Loaded level of size {file_height}x{file_width} characters")
        print(f"Extracted {len(patches)} patches of size {win_h}x{win_w}")
        self.patches = patches
        return patches

    def convert_to_identity(self, symb_file: list) -> list:
        return [
            [self.symbol_to_id.get(char, -1) for char in row]
            for row in symb_file
        ]

    def convert_id_to_embedding(self, id_file: list) -> np.ndarray:
        id_array = np.array(id_file)
        h, w = id_array.shape
        embedded = self.embedding_matrix[id_array]  # (H, W, D)
        return embedded

    # ================= BACKWARD OPERATIONS ======================
    def decode_from_embeddings(self, vector_grid: np.ndarray) -> list:
        h, w, d = vector_grid.shape
        flat_vectors = vector_grid.reshape(-1, d)
        decoded_ids = []
        for vec in flat_vectors:
            distances = np.linalg.norm(self.embedding_matrix - vec, axis=1)
            decoded_ids.append(np.argmin(distances))
        id_grid = np.array(decoded_ids).reshape(h, w)
        return id_grid.tolist()

    def convert_identity_to_symbolic(self, id_file: list) -> list:
        return [
            ''.join(self.id_to_symbol.get(cell, '?') for cell in row)
            for row in id_file
        ]

    def forward_mapping(self, symb_file: list) -> tuple:
        id_file = self.convert_to_identity(symb_file)
        vector_file = self.convert_id_to_embedding(id_file)
        return id_file, vector_file

    def backward_mapping(self, vector_file: np.ndarray, orig_symb_file=None, orig_id_file=None) -> tuple:
        id_file = self.decode_from_embeddings(vector_file)

        if orig_id_file and id_file != orig_id_file:
            raise ValueError("Identity file mismatch after decoding from vector. Aborting.")
        print("Identity file matches original.")

        symb_file = self.convert_identity_to_symbolic(id_file)
        if orig_symb_file and symb_file != orig_symb_file:
            raise ValueError("Symbolic file mismatch after reconstruction. Aborting.")
        print("Symbolic file matches original.")

        return id_file, symb_file

    # ================= VISUALIZATION ======================
    def render_level_image(self, symb_name: str, symb_file: list, tile_dir: str, save_folder: str = None) -> Image.Image:
        level_name = os.path.splitext(symb_name)[0] + ".png"

        id_file = self.convert_to_identity(symb_file)
        height = len(id_file)
        width = len(id_file[0])

        # Get tile size from any valid tile
        for test_id in self.tile_filenames:
            sample_path = os.path.join(tile_dir, self.tile_filenames[test_id])
            if os.path.exists(sample_path):
                sample_tile = Image.open(sample_path)
                tile_width, tile_height = sample_tile.size
                print(f"[INFO] Tile size: {tile_width} x {tile_height}")
                break
        else:
            raise RuntimeError("No valid tile images found in tile_filenames mapping.")

        # Load background tile (ID 2 = '-')
        background_filename = self.tile_filenames.get(2)
        if background_filename is None:
            raise RuntimeError("Missing background tile (ID 2) in tile_filenames mapping.")
        
        background_tile_path = os.path.join(tile_dir, background_filename)
        if not os.path.exists(background_tile_path):
            raise FileNotFoundError(f"Background tile file not found: {background_tile_path}")
        
        background_tile = Image.open(background_tile_path).convert("RGBA")

        # Create the image canvas
        full_img = Image.new("RGBA", (width * tile_width, height * tile_height))

        # Render each tile
        for y, row in enumerate(id_file):
            for x, tile_id in enumerate(row):
                filename = self.tile_filenames.get(tile_id)
                if filename is None:
                    tile = background_tile  # Use background for unknown/missing tile IDs
                else:
                    tile_path = os.path.join(tile_dir, filename)
                    if not os.path.exists(tile_path):
                        raise FileNotFoundError(f"Missing tile: {tile_path}")
                    tile = Image.open(tile_path).convert("RGBA")
                full_img.paste(tile, (x * tile_width, y * tile_height), tile)

        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            full_img.save(os.path.join(save_folder, level_name))

        return full_img

    @staticmethod
    def visualize_file(file=None):
        for line in file:
            print(line)
        print("\n")

# ================= MAIN ======================

if __name__ == "__main__":
    mapping_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mapping.yaml'))
    symb_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'symbol'))
    tile_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'tiles'))
    save_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))

    files = os.listdir(symb_folder)
    symb_test = files[0]  # e.g., "mario_1_1.txt"
    print(f"Rendering: {symb_test}")

    processor = ProcessDataSymbolic(mapping_path=mapping_path)
    symb_file = processor.load_symbolic(symb_test)
    patches = processor.crop_symbolic()
    print(f"Number of patches for given file: {len(patches)}")

    rendered_img = processor.render_level_image(symb_test, symb_file, tile_dir, save_folder)
    rendered_img.show()


    #for i, patch in enumerate(patches[:n_patches]):
    #    id_file, vector_file = processor.forward_mapping(patch)
     #   print("Original symbolic file:")
     #   processor.visualize_file(patch)
     #   id_file_re, symb_file_re = processor.backward_mapping(vector_file, orig_symb_file=patch, orig_id_file=id_file)
      #  print("Reconstructed symbolic file:")
     #  processor.visualize_file(symb_file_re)

