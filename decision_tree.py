import os
from image_processing import measure_changes

def decision_tree(images, change_threshold):
    """
    This function processes the images and classifies them into events (with animals)
    or non-events (no animals) based on pixel changes across frames.
    
    Parameters:
    - images: List of Pillow Image objects
    - change_threshold: Pixel change threshold to classify an event
    
    Returns:
    - events: List of images classified as events (animals present)
    - non_events: List of images classified as non-events (no animals)
    """
    events = []
    non_events = []
    
    print(f"Running decision tree with change threshold {change_threshold}")
    for i in range(1, len(images) - 1):
        past = images[i - 1]
        present = images[i]
        future = images[i + 1]

        change = measure_changes(past, present, future)
        #print(f"Change score for frame {i}: {change}")
        
        # Classify as event or non-event based on change threshold
        if change > change_threshold:
            events.append(present)
        else:
            non_events.append(present)
            print(f"No event detected in frame {i}")
    
    return events, non_events

def create_event_directories(events, base_output_directory):
    for i, event in enumerate(events):
        event_dir = os.path.join(base_output_directory, f"event_{i}")
        os.makedirs(event_dir, exist_ok=True)
        for img in event:
            img.save(os.path.join(event_dir, os.path.basename(img.filename)))
