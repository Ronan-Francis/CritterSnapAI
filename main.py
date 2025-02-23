import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import config
from sklearn_classifier import train_animal_classifier, predict_image
from config import run
from sorting_utils import sort_images_by_date_time, group_images_by_event
from classification import detect_motion
from data_structures import ImageObject

def sort_and_log_images(directory_path: str, white_pixel_threshold: int) -> List[ImageObject]:
    """
    Sort images by date/time and log their file paths.
    
    Returns:
        A list of ImageObject instances.
    """
    images = sort_images_by_date_time(directory_path, white_pixel_threshold)
    print(f"Total images sorted: {len(images)}")
    for image_obj in images:
        print(f"  - {image_obj.get_file_path()}")
    return images

def group_and_validate_images(images: List[ImageObject]) -> List[List[ImageObject]]:
    """
    Group images into events and validate the grouping.
    
    Returns:
        A list of groups, each a list of ImageObject instances.
    """
    groups = group_images_by_event(images)
    print(f"Total groups formed: {len(groups)}")
    for group in groups:
        print(f"  - Group size: {len(group)}")
    total_grouped = sum(len(group) for group in groups)
    if total_grouped != len(images):
        print(f"Discrepancy in grouping: {total_grouped} grouped vs {len(images)} total images.")
    else:
        print("All images are correctly grouped.")
    return groups

def run_first_pass(groups: List[List[ImageObject]], change_threshold: float, edge_confidence_threshold: float) -> Tuple[List[ImageObject], List[ImageObject]]:
    """
    Perform first-pass event detection in parallel across image groups.
    
    Returns:
        A tuple of two lists: (detected events, low-confidence candidates).
    """
    detected_events: List[ImageObject] = []
    low_conf_candidates: List[ImageObject] = []
    max_workers = min(os.cpu_count() or 1, len(groups))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(detect_motion, group, change_threshold, edge_confidence_threshold): group for group in groups}
        total_groups = len(groups)
        for i, future in enumerate(as_completed(future_map)):
            events, non_events = future.result()
            detected_events.extend(events)
            low_conf_candidates.extend(non_events)
            processed = i + 1
            percentage = (processed / total_groups) * 100
            group_size = len(future_map[future])
            print(f"First-pass detection: {processed}/{total_groups} ({percentage:.2f}% complete) - Group size: {group_size}, Detected events: {len(detected_events)}, Non-events: {len(low_conf_candidates)}", end="\r")
    print()  # New line after progress output

    total_processed = len(detected_events) + len(low_conf_candidates)
    expected_total = sum(len(group) for group in groups)
    if total_processed != expected_total:
        print(f"Warning: Total processed images ({total_processed}) does not match total images sorted ({expected_total}).")
    else:
        print("All images have been processed in the first pass.")

    print(f"\nDetected high-confidence events from first pass: {len(detected_events)}")
    print(f"Below threshold (non-events) from first pass: {len(low_conf_candidates)}")
    return detected_events, low_conf_candidates

def run_second_pass(low_conf_candidates: List[ImageObject], classifier) -> Tuple[List[ImageObject], List[ImageObject]]:
    """
    Perform second-pass analysis using AI classification on low-confidence candidates.
    
    Returns:
        A tuple of two lists: (second-pass events, confirmed non-events).
    """
    second_pass_events: List[ImageObject] = []
    confirmed_non_events: List[ImageObject] = []
    total = len(low_conf_candidates)
    
    for idx, item in enumerate(low_conf_candidates, start=1):
        image_path = item.get_file_path()
        label = predict_image(image_path, classifier)
        if label == "Animal":
            second_pass_events.append(item)
        else:
            confirmed_non_events.append(item)
        print(f"Second-pass analysis: {idx}/{total} ({(idx / total) * 100:.2f}% complete)", end="\r")
    print()  # New line after progress output
    return second_pass_events, confirmed_non_events

def write_log(output_log_path: str, all_events: List[ImageObject], confirmed_non_events: List[ImageObject]) -> None:
    """
    Write a summary log of confirmed events and non-events to a file.
    """
    with open(output_log_path, 'w') as log_file:
        log_file.write(f"Events Confirmed (All): {len(all_events)}\n")
        log_file.write(f"Non-Events Confirmed: {len(confirmed_non_events)}\n")

def main() -> None:
    start_time = time.time()
    config_values = run()
    
    directory_path: str = config_values["directory_path"]
    output_directory: str = config_values["output_directory"]
    change_threshold: float = config_values["change_threshold"]
    white_pixel_threshold: int = config_values["white_pixel_threshold"]
    edge_confidence_threshold: float = config_values["edge_confidence_threshold"]
    output_log_path: str = config_values["output_log_path"]
    animal_training_path: str = config_values["animal_training_path"]
    # non_animal_training_path is in config but not used in this flow.
    
    print("Starting the image sorting process...")
    images = sort_and_log_images(directory_path, white_pixel_threshold)
    groups = group_and_validate_images(images)
    
    print("\nPerforming first-pass event detection...")
    detected_events, low_conf_candidates = run_first_pass(groups, change_threshold, edge_confidence_threshold)
    
    print("Training animal classifier...")
    classifier = train_animal_classifier(animal_training_path)
    print("Animal classifier training complete.")
    
    print("\nSecond-pass AI check for below-threshold items...")
    second_pass_events, confirmed_non_events = run_second_pass(low_conf_candidates, classifier)
    
    all_events = detected_events + second_pass_events
    print(f"\nConfirmed events after second pass: {len(all_events)}")
    print(f"Confirmed non-events: {len(confirmed_non_events)}")
    
    # Additional functionality for saving images can be integrated here.
    write_log(output_log_path, all_events, confirmed_non_events)
    
    print("\n=== FINAL RESULTS ===")
    print(f"Total Confirmed Events: {len(all_events)}")
    print(f"Total Confirmed Non-Events: {len(confirmed_non_events)}")
    
    elapsed = time.time() - start_time
    print(f"\nProcessing complete in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
