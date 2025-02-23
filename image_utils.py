from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Optional

def downsample_image(img: Image.Image, scale_factor: float = 0.5) -> Image.Image:
    """
    Downsamples (resizes) the given image by the provided scale factor.

    Parameters:
        img (Image.Image): The Pillow Image to downsample.
        scale_factor (float): The factor by which to scale the image (e.g., 0.5 for half-size).

    Returns:
        Image.Image: The resized image.
    """
    width, height = img.size
    return img.resize((int(width * scale_factor), int(height * scale_factor)))

def measure_changes(past: Optional[Image.Image], 
                    present: Optional[Image.Image], 
                    future: Optional[Image.Image], 
                    scale_factor: float = 0.5) -> float:
    """
    Measures pixel changes between three images using the Structural Similarity Index (SSIM).

    The function downscales each image, converts them to grayscale, computes SSIM between the
    past and present images as well as between the present and future images, then returns the 
    average difference (1 - average SSIM).

    If any image is missing (None), the function returns 0.

    Parameters:
        past (Optional[Image.Image]): The past image.
        present (Optional[Image.Image]): The current image.
        future (Optional[Image.Image]): The future image.
        scale_factor (float): The factor by which to downsample the images.

    Returns:
        float: The difference measure derived from SSIM (0 indicates perfect similarity).
    """
    if past is None or present is None or future is None:
        return 0.0

    past_ds = downsample_image(past, scale_factor)
    present_ds = downsample_image(present, scale_factor)
    future_ds = downsample_image(future, scale_factor)

    past_array = np.array(past_ds.convert("L"), dtype=np.float32)
    present_array = np.array(present_ds.convert("L"), dtype=np.float32)
    future_array = np.array(future_ds.convert("L"), dtype=np.float32)

    ssim_prev_curr = ssim(past_array, present_array, data_range=255)
    ssim_curr_next = ssim(present_array, future_array, data_range=255)
    avg_ssim = (ssim_prev_curr + ssim_curr_next) / 2

    return 1 - avg_ssim
