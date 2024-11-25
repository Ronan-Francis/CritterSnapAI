from PIL import Image, ImageChops, ImageFilter
import numpy as np

def detect_edge(img):
    # Convert the image to grayscale
    gray_img = img.convert("L")
    
    # Apply Sobel filter for edge detection
    edges = gray_img.filter(ImageFilter.FIND_EDGES)
    
    return edges

def measure_changes(past, present, future):
    """
    Calculate the average pixel change between three consecutive images
    to detect significant changes.

    Parameters:
    - past: The previous frame (Pillow Image object).
    - present: The current frame (Pillow Image object).
    - future: The next frame (Pillow Image object).

    Returns:
    - change_score: A numerical score representing the pixel change intensity.
    """

    # Convert images to grayscale and detect edges
    past_edges = detect_edge(past)
    present_edges = detect_edge(present)
    future_edges = detect_edge(future)

    # Calculate the absolute differences between consecutive edge-detected images
    diff_past_present = ImageChops.difference(past_edges, present_edges)
    diff_present_future = ImageChops.difference(present_edges, future_edges)

    # Convert difference images to numpy arrays for pixel intensity calculation
    diff_past_present = np.array(diff_past_present)
    diff_present_future = np.array(diff_present_future)

    # Calculate the change score as the sum of absolute differences
    change_score_past_present = np.sum(diff_past_present)
    change_score_present_future = np.sum(diff_present_future)

    # Average the scores from past-present and present-future differences
    change_score = (change_score_past_present + change_score_present_future) / 2

    return change_score
