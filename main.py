import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import config
from sklearn_classifier import train_animal_classifier, predict_image
from config import run
from sorting_utils import sort_images_by_date_time, group_images_by_event
from classification import detect_motion  # now importing from classification.py

def sort_and_log_images(directory_path, white_pixel_threshold):
    """Sorts images and logs their file paths."""
    images = sort_images_by_date_time(directory_path, white_pixel_threshold)
    print(f"Total images sorted: {len(images)}")
    for image_obj in images:
        print(f"  - {image_obj.get_file_path()}")
    return images


def group_and_validate_images(images):
    """Groups images by event and validates the grouping."""
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


def run_first_pass(groups, change_threshold, edge_confidence_threshold):
    """Detects events in parallel across groups."""
    detected_events = []
    low_conf_candidates = []  # candidates below threshold
    max_workers = min(os.cpu_count(), len(groups))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(detect_motion, group, change_threshold, edge_confidence_threshold): group 
                     for group in groups}
        for i, future in enumerate(as_completed(future_map)):
            events, non_events = future.result()
            detected_events.extend(events)
            low_conf_candidates.extend(non_events)
            processed_count = i + 1
            total_count = len(groups)
            percentage = (processed_count / total_count) * 100
            print(f"First-pass detection: {processed_count}/{total_count} ({percentage:.2f}% complete)", end="\r")
            print(f"Group size: {len(future_map[future])}, Detected events: {len(detected_events)}, Non-events: {len(low_conf_candidates)}")
    print()  # new line after progress output

    total_processed = len(detected_events) + len(low_conf_candidates)
    if total_processed != sum(len(group) for group in groups):
        print(f"Warning: Total processed images ({total_processed}) does not match total images sorted ({sum(len(group) for group in groups)}).")
    else:
        print("All images have been processed in the first pass.")

    print(f"\nDetected high-confidence events from first pass: {len(detected_events)}")
    print(f"Below threshold (non_events) from first pass: {len(low_conf_candidates)}")
    return detected_events, low_conf_candidates


def run_second_pass(low_conf_candidates, classifier):
    """Checks below-threshold images using AI classification."""
    second_pass_events = []
    confirmed_non_events = []
    total_low_conf = len(low_conf_candidates)
    for idx, item in enumerate(low_conf_candidates, start=1):
        image_path = item.get_file_path()
        label = predict_image(image_path, classifier)

        # If the AI classifies the image as "Animal," treat it as an event.
        if label == "Animal":
            second_pass_events.append(item)
        else:
            confirmed_non_events.append(item)

        percent_done = (idx / total_low_conf) * 100
        print(f"Second-pass analysis: {idx}/{total_low_conf} ({percent_done:.2f}% complete)", end="\r")
    print()  # new line after progress output
    return second_pass_events, confirmed_non_events


def write_log(output_log_path, all_events, confirmed_non_events):
    """Writes a summary log of confirmed events and non-events."""
    with open(output_log_path, 'w') as log_file:
        log_file.write(f"Events Confirmed (All): {len(all_events)}\n")
        log_file.write(f"Non-Events Confirmed: {len(confirmed_non_events)}\n")


def main():
    start_time = time.time()
    # Retrieve configuration values.
    config_values = run()
    directory_path = config_values["directory_path"]
    output_directory = config_values["output_directory"]
    change_threshold = config_values["change_threshold"]
    white_pixel_threshold = config_values["white_pixel_threshold"]
    edge_confidence_threshold = config_values["edge_confidence_threshold"]
    output_log_path = config_values["output_log_path"]
    animal_training_path = config_values["animal_training_path"]
    non_animal_training_path = config_values["non_animal_training_path"]

    print("Starting the image sorting process...")
    
    # Step 1: Sort images.
    images = sort_and_log_images(directory_path, white_pixel_threshold)
    
    # Step 2: Group images and validate the grouping.
    groups = group_and_validate_images(images)
    
    # Step 3: First-pass event detection (parallel processing).
    print("\nPerforming first-pass event detection...")
    detected_events, low_conf_candidates = run_first_pass(groups, change_threshold, edge_confidence_threshold)
    
    # Step 4: Train model for second-pass classification.
    print("Training animal classifier...")
    classifier = train_animal_classifier(animal_training_path)
    print("Animal classifier training complete.")
    
    # Step 5: Second-pass AI check for below-threshold items.
    print("\nSecond-pass AI check for below-threshold items...")
    second_pass_events, confirmed_non_events = run_second_pass(low_conf_candidates, classifier)
    
    # Combine events from both passes.
    all_events = detected_events + second_pass_events
    print(f"\nConfirmed events after second pass: {len(all_events)}")
    print(f"Confirmed non-events: {len(confirmed_non_events)}")
    
    # Step 6: Save Images (implementation assumed elsewhere).
    
    # Step 7: Write log output.
    write_log(output_log_path, all_events, confirmed_non_events)
    
    # Final summaries.
    print("\n=== FINAL RESULTS ===")
    print(f"Total Confirmed Events: {len(all_events)}")
    print(f"Total Confirmed Non-Events: {len(confirmed_non_events)}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nProcessing complete in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
