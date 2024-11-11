from PIL import ImageChops
import numpy as np

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

    # Convert images to grayscale to simplify comparison
    past = past.convert("L")
    present = present.convert("L")
    future = future.convert("L")

    # Calculate the absolute differences between consecutive images
    diff_past_present = ImageChops.difference(past, present)
    diff_present_future = ImageChops.difference(present, future)

    # Convert difference images to numpy arrays for pixel intensity calculation
    diff_past_present = np.array(diff_past_present)
    diff_present_future = np.array(diff_present_future)

    # Calculate the change score as the sum of absolute differences
    change_score_past_present = np.sum(diff_past_present)
    change_score_present_future = np.sum(diff_present_future)

    # Average the scores from past-present and present-future differences
    change_score = (change_score_past_present + change_score_present_future) / 2

    return change_score
