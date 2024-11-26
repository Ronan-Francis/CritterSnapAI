import os
from image_processing import measure_changes
from gdpr_detection import is_gdpr_image
from PIL import Image


def decision_tree(images, change_threshold, white_pixel_threshold, gdpr_output_directory):
    """
    Processes images and classifies them into events, non-events, or GDPR non-events.

    Parameters:
    - images: List of image file paths.
    - change_threshold: Pixel change threshold to classify an event.
    - white_pixel_threshold: The minimum number of white pixels to classify as GDPR image.
    - gdpr_output_directory: Directory to store GDPR non-events.

    Returns:
    - events: List of images classified as events (animals present).
    - non_events: List of images classified as non-events (no animals).
    """
    if not os.path.exists(gdpr_output_directory):
        os.makedirs(gdpr_output_directory)

    events = []
    non_events = []

    for i in range(1, len(images) - 1):
        # print(f"Processing image {i} of {len(images)}")
        print(f"Percentage complete: {i / len(images) * 100:.1f}%", end='\r')
        past = images[i - 1]
        present = images[i]
        future = images[i + 1]

        # Check if the image is a GDPR image
        if not is_gdpr_image(present, white_pixel_threshold):
            # Measure changes for event classification
            pixel_changes = measure_changes(past, present, future)

            if pixel_changes > change_threshold:
                events.append(present)
                #print(f"Event detected: {present}")
            else:
                non_events.append(present)
                #print(f"No event detected: {present}")
            continue


    return events, non_events
