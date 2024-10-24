# decision_tree.py

from image_processing import measure_changes

def decision_tree(images, change_threshold=10000):
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
    
    for i in range(1, len(images) - 1):
        past = images[i - 1]
        present = images[i]
        future = images[i + 1]

        change = measure_changes(past, present, future)
        
        # Classify as event or non-event based on change threshold
        if change > change_threshold:
            events.append(present)
            print(f"Event detected in frame {i}")
        else:
            non_events.append(present)
            print(f"No event detected in frame {i}")
    
    return events, non_events
