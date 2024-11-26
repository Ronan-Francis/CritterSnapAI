import os
from image_processing import measure_changes
from gdpr_detection import is_gdpr_image
from PIL import Image
from imageObj import ImageObject

def decision_tree(image_objects, change_threshold, white_pixel_threshold, gdpr_output_directory):
    """
    Processes images and classifies them into events, non-events, or GDPR non-events.

    Parameters:
    - image_objects: List of ImageObject instances.
    - change_threshold: Pixel change threshold to classify an event.
    - white_pixel_threshold: The minimum number of white pixels to classify as GDPR image.
    - gdpr_output_directory: Directory to store GDPR non-events.

    Returns:
    - events: List of ImageObject instances classified as events (animals present).
    - non_events: List of ImageObject instances classified as non-events (no animals).
    """
    events = []
    non_events = []

    for i in range(1, len(image_objects) - 1):
        past = image_objects[i - 1].get_image()
        present = image_objects[i].get_image()
        future = image_objects[i + 1].get_image()

        pixel_changes = measure_changes(past, present, future)

        if pixel_changes > change_threshold:
            events.append(image_objects[i])
        else:
            non_events.append(image_objects[i])

    return events, non_events
