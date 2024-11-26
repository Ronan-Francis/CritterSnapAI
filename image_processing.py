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
    Measures pixel changes between three images to classify events.

    Parameters:
    - past: The past image.
    - present: The present image.
    - future: The future image.

    Returns:
    - pixel_changes: The number of pixel changes detected.
    """
    # Replace with actual logic based on your requirements
    if past is None or present is None or future is None:
        return 0

    # Convert images to grayscale
    past_gray = past.convert("L")
    present_gray = present.convert("L")
    future_gray = future.convert("L")

    # Calculate pixel differences
    past_diff = ImageChops.difference(past_gray, present_gray)
    future_diff = ImageChops.difference(present_gray, future_gray)

    # Count non-zero pixels (indicating changes)
    past_changes = sum(past_diff.getdata())
    future_changes = sum(future_diff.getdata())

    # Return the average of changes
    #print(f"Past changes: {past_changes}, Future changes: {future_changes}")
    return (past_changes + future_changes) / 2
