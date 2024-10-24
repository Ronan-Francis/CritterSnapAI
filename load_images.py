from PIL import Image
import os

def load_images_from_directory(directory_path):
    """
    This function loads all images from the specified directory
    and returns them as a list of Pillow Image objects.
    """
    image_files = [f for f in os.listdir(directory_path) if f.endswith(('.png', '.JPG', '.jpeg'))]
    images = []
    for file in image_files:
        img = Image.open(os.path.join(directory_path, file))
        images.append(img)
    return images
