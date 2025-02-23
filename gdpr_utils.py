from PIL import Image
from typing import Any

def is_not_gdpr_image(img: Image.Image, white_pixel_threshold: int) -> bool:
    """
    Check if an image is not GDPR-protected based on the count of pure-white pixels.

    Parameters:
        img (Image.Image): The image to evaluate.
        white_pixel_threshold (int): The maximum number of pure-white pixels allowed.

    Returns:
        bool: True if the image is NOT GDPR-protected (i.e., white pixel count is less than or equal to the threshold), otherwise False.
    """
    rgb_img = img.convert("RGB")
    pixels = rgb_img.getdata()
    white_pixel_count = sum(1 for pixel in pixels if pixel == (255, 255, 255))
    return white_pixel_count <= white_pixel_threshold
