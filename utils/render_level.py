import os
from .process_data import ProcessDataSymbolic


# ONE LINE FUNCTIONS FOR EASY CALLING




def run_render(processor, symb_name:str=None, save_folder:str=None ,symb_file: list=[],visualize:bool=False):
    rendered_img = processor.render_level_image(symb_name = symb_name,
                                                symb_file = symb_file,
                                                tile_dir=tile_dir, 
                                                save_folder=save_folder,
                                                visualize=visualize )
    return rendered_img


if __name__ == "__main__":
    # use this paths if cloning the repo else specify your own absolute or relative paths : )
    mapping_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'mapping.yaml'))
    symb_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'symbol'))
    tile_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'tiles'))
    save_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    
    files = os.listdir(symb_folder)
    symb_test = files[0]   
    processor = ProcessDataSymbolic(mapping_path=mapping_path)
    symb_file = processor.load_symbolic(symb_test) 
    level_rendered = run_render(processor,symb_file=symb_file, symb_name=symb_test, save_folder=save_folder,visualize=True)

