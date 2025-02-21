import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import config
from sklearn_classifier import train_animal_classifier, predict_image
from config import run
from sorting_utils import sort_images_by_date_time, group_images_by_event
from classification import process_group

def detect_motion(group, change_threshold, edge_confidence_threshold):
    """
    A top-level function for motion detection.
    This avoids the pickling issue on Windows when used with ProcessPoolExecutor.
    """
    events, non_events = process_group(group, change_threshold, edge_confidence_threshold)
    
    # Logging for debugging
    print(f"Processing group of size {len(group)}")
    print(f" - Detected events: {len(events)}")
    print(f" - Non-events: {len(non_events)}")
    
    return events, non_events

def main():
    start_time = time.time()
    # Get the configuration (updates or defaults) via the run() function
    config_values = run()
    
    # Extract configuration values from the returned dictionary
    directory_path = config_values["directory_path"]
    output_directory = config_values["output_directory"]
    change_threshold = config_values["change_threshold"]
    white_pixel_threshold = config_values["white_pixel_threshold"]
    edge_confidence_threshold = config_values["edge_confidence_threshold"]
    output_log_path = config_values["output_log_path"]
    animal_training_path = config_values["animal_training_path"]
    non_animal_training_path = config_values["non_animal_training_path"]
    print("Starting the image sorting process...")

    # 1. Sort/Filter Images
    print("Running preprocessing and sorting...")
    images_with_dates = sort_images_by_date_time(directory_path, white_pixel_threshold)
    print(f"Total images sorted: {len(images_with_dates)}")
    for image_obj in images_with_dates:
        print(f"  - {image_obj.get_file_path()}")

    # 2. Group Images
    grouped_events = group_images_by_event(images_with_dates)
    print(f"Total groups formed: {len(grouped_events)}")
    for grouped_event in grouped_events:
        print(f"  - Group size: {len(grouped_event)}")
    total_grouped = sum(len(group) for group in grouped_events)
    if total_grouped != len(images_with_dates):
        print(f"Discrepancy in grouping: {total_grouped} grouped vs {len(images_with_dates)} total images.")
    else:
        print("All images are correctly grouped.")

    # 3. Detect Events in Parallel
    print("\nPerforming first-pass event detection...")
    detected_events = []
    low_conf_candidates = []  # below threshold => we'll check with AI

    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), len(grouped_events))) as executor:
        # Note: We now call the *top-level* function detect_motion
        future_map = {executor.submit(detect_motion, g, change_threshold, edge_confidence_threshold): g for g in grouped_events}

        for i, future in enumerate(as_completed(future_map)):
            events, non_events = future.result()
            detected_events.extend(events)
            low_conf_candidates.extend(non_events)

            processed_count = i + 1
            total_count = len(grouped_events)
            percentage = (processed_count / total_count) * 100
            print(f"First-pass detection: {processed_count}/{total_count} ({percentage:.2f}% complete)", end="\r")
            print(f"Group size: {len(future_map[future])}, Detected events: {len(detected_events)}, Non-events: {len(low_conf_candidates)}")

    print()  # Move to next line after loop
    # After first pass
    total_processed = len(detected_events) + len(low_conf_candidates)
    if total_processed != len(images_with_dates):
        print(f"Warning: Total processed images ({total_processed}) does not match total images sorted ({len(images_with_dates)}).")
    else:
        print("All images have been processed in the first pass.")

    print(f"\nDetected high-confidence events from first pass: {len(detected_events)}")
    print(f"Below threshold (non_events) from first pass: {len(low_conf_candidates)}")

    # 4. Train model for second-pass classification
    # Train the classifier using the provided animal training path
    print("Animal classifier training complete.")
    classifier = train_animal_classifier(animal_training_path)

    # 5. Second Pass: check below-threshold items with AI
    print("\nSecond-pass AI check for below-threshold items...")
    second_pass_events = []
    confirmed_non_events = []

    total_low_conf = len(low_conf_candidates)
    for idx, item in enumerate(low_conf_candidates, start=1):
        image_path = item.get_file_path()
        label = predict_image(image_path, classifier)

        # If the AI says "Animal," move it to events
        if label == "Animal":
            second_pass_events.append(item)
        else:
            confirmed_non_events.append(item)

        percent_done = (idx / total_low_conf) * 100
        print(f"Second-pass analysis: {idx}/{total_low_conf} ({percent_done:.2f}% complete)", end="\r")

    print()

    # Combine final events
    all_events = detected_events + second_pass_events

    print(f"\nConfirmed events after second pass: {len(all_events)}")
    print(f"Confirmed non-events: {len(confirmed_non_events)}")

    # 6. Save Images
    # (Implementation not provided, assume it's handled elsewhere or left as future work)

    # 7. Write to Log
    with open(output_log_path, 'w') as log_file:
        log_file.write(f"Events Confirmed (All): {len(all_events)}\n")
        log_file.write(f"Non-Events Confirmed: {len(confirmed_non_events)}\n")

    # Summaries
    print("\n=== FINAL RESULTS ===")
    print(f"Total Confirmed Events: {len(all_events)}")
    print(f"Total Confirmed Non-Events: {len(confirmed_non_events)}")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nProcessing complete in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
