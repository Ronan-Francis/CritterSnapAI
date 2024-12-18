import os
from PIL import Image
from imageObj import ImageObject
from image_processing import measure_changes
from gdpr_detection import is_not_gdpr_image
from concurrent.futures import ProcessPoolExecutor, as_completed
# from wildlife_detection_classification import wildlife_detection_classification

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
    events = []
    non_events = []

    for i in range(1, len(image_objects) - 1):
        event, non_event = process_image(image_objects, i, change_threshold)
        if event:
            events.append(event)
        if non_event:
            non_events.append(non_event)

    return events, non_events
