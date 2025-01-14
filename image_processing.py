from PIL import Image, ImageChops, ImageFilter
import numpy as np

def detect_edge(img):
    # Convert the image to grayscale
    gray_img = img.convert("L")
    
    # Apply Sobel filter for edge detection
    edges = gray_img.filter(ImageFilter.FIND_EDGES)
    
    return edges

def downsample_image(img, scale_factor=0.5):
    width, height = img.size
    return img.resize((int(width * scale_factor), int(height * scale_factor)))

def measure_changes(past, present, future, scale_factor=0.5):
    """
    Measures pixel changes between three images to classify events.

    Parameters:
    - past: The past image.
    - present: The present image.
    - future: The future image.
    - scale_factor: The factor by which to downsample the images.

    Returns:
    - pixel_changes: The number of pixel changes detected.
    """
    if past is None or present is None or future is None:
        return 0

    past_ds = downsample_image(past, scale_factor)
    present_ds = downsample_image(present, scale_factor)
    future_ds = downsample_image(future, scale_factor)

    past_array = np.array(past_ds.convert("L"))
    present_array = np.array(present_ds.convert("L"))
    future_array = np.array(future_ds.convert("L"))

    past_diff = np.abs(past_array - present_array)
    future_diff = np.abs(present_array - future_array)

    return (np.sum(past_diff) + np.sum(future_diff)) / 2
