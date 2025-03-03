from image_processor import ImageProcessor, ImageObject

def process_image(image_objects, index, change_threshold, edge_conf_threshold):
    """
    Classify an image using SSIM change measure and edge-based confidence.
    """
    past = image_objects[index - 1].get_image() if index - 1 >= 0 else None
    present = image_objects[index].get_image()
    future = image_objects[index + 1].get_image() if index + 1 < len(image_objects) else None

    # Compute SSIM-based pixel changes.
    pixel_changes = ImageProcessor.measure_changes(past, present, future)
    # Compute edge-based confidence.
    edge_conf, _ = ImageProcessor.compute_edge_confidence(
        present, edge_threshold=50, window_size=20
    )
    composite_score = (pixel_changes / change_threshold) + (edge_conf / edge_conf_threshold)
    print(f"Composite score for image {image_objects[index].get_file_path()}: {composite_score:.3f}")

    if composite_score > 5.0:
        return image_objects[index], None
    else:
        return None, image_objects[index]

def process_group(group, change_threshold, edge_conf_threshold):
    """
    Process a group of images and classify each as event or non-event.
    """
    events = []
    non_events = []
    for index in range(len(group)):
        ev, non_ev = process_image(group, index, change_threshold, edge_conf_threshold)
        if ev:
            events.append(ev)
        if non_ev:
            non_events.append(non_ev)
    return events, non_events

def detect_motion(group, change_threshold, edge_conf_threshold):
    """
    Top-level function for motion detection in a group.
    Designed to work with parallel processing frameworks.
    """
    events, non_events = process_group(group, change_threshold, edge_conf_threshold)
    print(f"Processing group of size {len(group)}: {len(events)} events detected.")
    return events, non_events
