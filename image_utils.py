# image_utils.py
from PIL import Image, ImageFilter
import numpy as np

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
    Measures pixel changes between three images to classify events.
    
    Parameters:
    - past: The past image (PIL Image)
    - present: The present image (PIL Image)
    - future: The future image (PIL Image)
    - scale_factor: The factor by which to downsample the images
    
    Returns:
    - The average pixel difference between (past, present) and (present, future)
    """
    if past is None or present is None or future is None:
        return 0

    past_ds = downsample_image(past, scale_factor)
    present_ds = downsample_image(present, scale_factor)
    future_ds = downsample_image(future, scale_factor)

    past_array = np.array(past_ds.convert("L"), dtype=np.float32)
    present_array = np.array(present_ds.convert("L"), dtype=np.float32)
    future_array = np.array(future_ds.convert("L"), dtype=np.float32)

    diff_prev_curr = np.abs(past_array - present_array).sum()
    diff_curr_next = np.abs(present_array - future_array).sum()

    return (diff_prev_curr + diff_curr_next) / 2
