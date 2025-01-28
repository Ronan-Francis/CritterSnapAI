from PIL import Image, ImageFilter
import numpy as np
from skimage.metrics import structural_similarity as ssim

def downsample_image(img, scale_factor=0.5):
    """
    Downsamples (resizes) the given image by the scale_factor.

    Parameters:
    - img: a Pillow Image object
    - scale_factor: how much to scale (0.5 = half size)

    Returns:
    - A new, resized Pillow Image object
    """
    width, height = img.size
    return img.resize((int(width * scale_factor), int(height * scale_factor)))

def measure_changes(past, present, future, scale_factor=0.5):
    """
    Measures pixel changes between three images to classify events using SSIM.
    
    Parameters:
    - past: The past image (PIL Image)
    - present: The present image (PIL Image)
    - future: The future image (PIL Image)
    - scale_factor: The factor by which to downsample the images
    
    Returns:
    - The average SSIM between (past, present) and (present, future)
    """
    if past is None or present is None or future is None:
        return 0

    past_ds = downsample_image(past, scale_factor)
    present_ds = downsample_image(present, scale_factor)
    future_ds = downsample_image(future, scale_factor)

    past_array = np.array(past_ds.convert("L"), dtype=np.float32)
    present_array = np.array(present_ds.convert("L"), dtype=np.float32)
    future_array = np.array(future_ds.convert("L"), dtype=np.float32)

    ssim_prev_curr = ssim(past_array, present_array, data_range=255)
    ssim_curr_next = ssim(present_array, future_array, data_range=255)
    
    avg_ssim = (ssim_prev_curr + ssim_curr_next) / 2

    # Convert similarity to difference
    diff_value = 1 - avg_ssim

    return diff_value
