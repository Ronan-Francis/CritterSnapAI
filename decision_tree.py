import os
from PIL import Image
from imageObj import ImageObject
from image_processing import measure_changes
from gdpr_detection import is_gdpr_image
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_image(image_objects, index, change_threshold):
    past = image_objects[index - 1].get_image()
    present = image_objects[index].get_image()
    future = image_objects[index + 1].get_image()

    pixel_changes = measure_changes(past, present, future)

    if pixel_changes > change_threshold:
        return image_objects[index], None
    else:
        return None, image_objects[index]

def decision_tree(image_objects, change_threshold, white_pixel_threshold, gdpr_output_directory):
    """
    Processes a list of ImageObject instances to classify them as events or non-events.

    Parameters:
    - image_objects: List of ImageObject instances to process.
    - change_threshold: The threshold for detecting changes between images.
    - white_pixel_threshold: The minimum number of white pixels to classify as GDPR image.
    - gdpr_output_directory: Directory to store GDPR non-events.

    Returns:
    - events: List of ImageObject instances classified as events (animals present).
    - non_events: List of ImageObject instances classified as non-events (no animals).
    """
    events = []
    non_events = []

    total_images = len(image_objects) - 2
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_image, image_objects, i, change_threshold) for i in range(1, len(image_objects) - 1)]
        for idx, future in enumerate(as_completed(futures)):
            event, non_event = future.result()
            if event:
                if is_gdpr_image(event.get_image(), white_pixel_threshold):
                    events.append(event)
            if non_event:
                non_events.append(non_event)

    return events, non_events
