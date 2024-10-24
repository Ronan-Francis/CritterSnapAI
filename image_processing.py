from PIL import Image
import numpy as np

# Convert image to grayscale for simplified comparison
def convert_to_grayscale(image):
    """
    Convert the input Pillow Image object to grayscale for easier comparison.
    """
    return image.convert("L")

# Detect motion or significant change between frames
def measure_changes(past, present, future):
    """
    This function compares the past, present, and future frames by calculating
    pixel-wise differences between them. It returns the sum of the absolute differences.
    """
    past_pixels = np.array(convert_to_grayscale(past))
    present_pixels = np.array(convert_to_grayscale(present))
    future_pixels = np.array(convert_to_grayscale(future))

    # Calculate pixel-wise differences between frames
    past_present_diff = np.abs(past_pixels - present_pixels)
    present_future_diff = np.abs(present_pixels - future_pixels)

    # Sum up changes over all pixels
    total_change = np.sum(past_present_diff) + np.sum(present_future_diff)
    
    return total_change
