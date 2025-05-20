from PIL import Image
import os


def create_tile(tile_size:tuple, color:tuple, folder:str=None,tile_name:str=None):
    img = Image.new("RGBA", tile_size, sky_blue)
    tile_path = os.path.join(folder,tile_name)
    img.save(tile_path)
    img.show()



if __name__ == "__main__":

    tile_size = (16, 16)  
    sky_blue = (107, 140, 255, 255)
    folder = "data/tiles/"
    tile_name = "encoding_2.png"
    create_tile(tile_size=tile_size,color=sky_blue,folder=folder,tile_name=tile_name)