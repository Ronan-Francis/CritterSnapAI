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
    
    if pixel_changes > change_threshold:
        # This image is classified as event
        return image_objects[index], None
    else:
        # This image is classified as non-event
        # print(f"Pixel changes: {pixel_changes:.2f}")
        return None, image_objects[index]


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
    #for index in range(1, len(group) - 1):
    for index in range(len(group)):
        ev, non_ev = process_image(group, index, change_threshold)
        if ev:
            events.append(ev)
        if non_ev:
            non_events.append(non_ev)
            #print(f"Non-event detected at index {index}")
            # print(f"Processing non_events of size {len(non_events)}")
    return events, non_events
