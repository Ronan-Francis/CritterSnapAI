from image_utils import measure_changes

def process_image(image_objects, index, change_threshold):
    """
    Determines whether the image at a given index is an event or non-event,
    based on the pixel change threshold from adjacent images.

    Parameters:
    - image_objects: A list of ImageObject instances
    - index: Current index in the list
    - change_threshold: Threshold above which we classify as an event

    Returns:
    - (event_obj, non_event_obj) -> (ImageObject or None, ImageObject or None)
    """
    past = image_objects[index - 1].get_image() if index - 1 >= 0 else None
    present = image_objects[index].get_image()
    future = image_objects[index + 1].get_image() if index + 1 < len(image_objects) else None

    pixel_changes = measure_changes(past, present, future)
    print(f"Pixel changes: {pixel_changes:.2f}")

    if pixel_changes > change_threshold:
        # This image is classified as event
        return image_objects[index], None
    else:
        # This image is classified as non-event
        return None, image_objects[index]

def decision_tree(image_objects, change_threshold):
    """
    Example 'decision tree' approach that iterates over a list of images,
    classifying each one as event or non-event.

    Parameters:
    - image_objects: A list of ImageObject instances
    - change_threshold: Threshold above which we classify as event

    Returns:
    - (events, non_events) -> List[ImageObject], List[ImageObject]
    """
    events = []
    non_events = []

    for i in range(1, len(image_objects) - 1):
        event_obj, non_event_obj = process_image(image_objects, i, change_threshold)
        if event_obj:
            events.append(event_obj)
        if non_event_obj:
            non_events.append(non_event_obj)

    return events, non_events

def process_group(group, change_threshold):
    """
    Processes a group of images (event cluster) to classify each image as event or non-event.

    Parameters:
    - group: A list of ImageObject instances
    - change_threshold: Threshold above which we classify as event

    Returns:
    - (events, non_events): two lists of ImageObject
    """
    events = []
    non_events = []
    for index in range(1, len(group) - 1):
        ev, non_ev = process_image(group, index, change_threshold)
        if ev:
            events.append(ev)
        if non_ev:
            non_events.append(non_ev)
    return events, non_events
