from edge_detection import compute_edge_confidence
from image_utils import measure_changes

def process_image(image_objects, index, change_threshold, edge_confidence_threshold):
    """
    Classify an image as event or non-event using both SSIM change measure and edge-based confidence.
    """
    past = image_objects[index - 1].get_image() if index - 1 >= 0 else None
    present = image_objects[index].get_image()
    future = image_objects[index + 1].get_image() if index + 1 < len(image_objects) else None

    # Existing measure based on SSIM changes
    pixel_changes = measure_changes(past, present, future)
    
    # Compute edge-based confidence.
    # Ensure present is a PIL Image (as expected by compute_edge_confidence)
    # edge_conf, _ = compute_edge_confidence(present, edge_threshold=50, window_size=20)
    edge_conf = 50.0  # Placeholder for now
    
    # Normalize each metric by its threshold and compute a composite score.
    # When both metrics are at their threshold, the normalized values are 1 and the sum is 2.
    # Here, we choose a cutoff of 1.0, meaning that a high value in one metric can compensate
    # for a lower value in the other.
    composite_score = (pixel_changes / change_threshold) + (edge_conf / edge_confidence_threshold)
    
    if composite_score > 1.0:
        return image_objects[index], None
    else:
        return None, image_objects[index]


def process_group(group, change_threshold, edge_confidence_threshold):
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
    for index in range(len(group)):
        ev, non_ev = process_image(group, index, change_threshold, edge_confidence_threshold)
        if ev:
            events.append(ev)
        if non_ev:
            non_events.append(non_ev)
    return events, non_events

