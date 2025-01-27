import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import (
    directory_path,
    output_directory,
    change_threshold,
    white_pixel_threshold,
    output_log_path,
    animal_training_path,
    non_animal_training_path
)
from sorting_utils import sort_images_by_date_time, group_images_by_event
from classification import process_group
from sklearn_classifier import train_animal_classifier, predict_image


def detect_motion(group):
    """
    A top-level function for motion detection.
    This avoids the pickling issue on Windows when used with ProcessPoolExecutor.
    """
    events, non_events = process_group(group, change_threshold)
    return events, non_events


def main():
    start_time = time.time()
    print("Starting the image sorting process...")

    # 1. Sort/Filter Images
    print("Running preprocessing and sorting...")
    images_with_dates = sort_images_by_date_time(directory_path, white_pixel_threshold)

    # 2. Group Images
    grouped_events = group_images_by_event(images_with_dates)
    print(f"Total groups formed: {len(grouped_events)}")
    for grouped_event in grouped_events:
        print(f"  - Group size: {len(grouped_event)}")

    # 3. Detect Events in Parallel
    print("\nPerforming first-pass event detection...")
    detected_events = []
    low_conf_candidates = []  # below threshold => we'll check with AI

    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), len(grouped_events))) as executor:
        # Note: We now call the *top-level* function detect_motion
        future_map = {executor.submit(detect_motion, g): g for g in grouped_events}

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

    print(f"\nDetected high-confidence events from first pass: {len(detected_events)}")
    print(f"Below threshold (non_events) from first pass: {len(low_conf_candidates)}")

    # 4. Train model for second-pass classification
    print("\nTraining animal classifier for second-pass analysis...")
    model = train_animal_classifier(animal_training_path)

    # 5. Second Pass: check below-threshold items with AI
    print("\nSecond-pass AI check for below-threshold items...")
    second_pass_events = []
    confirmed_non_events = []

    total_low_conf = len(low_conf_candidates)
    for idx, item in enumerate(low_conf_candidates, start=1):
        image_path = item.get_file_path()
        label = predict_image(image_path, model)

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

    print()

    # 7. Write to Log
    with open(output_log_path, 'w') as log_file:
        log_file.write(f"Events Confirmed (All): {len(all_events)}\n")
        #log_file.write(f"  - Recognized Species: {len(final_event_list)}\n")
        #log_file.write(f"  - Unknown Species: {len(unknown_species)}\n")
        log_file.write(f"Non-Events Confirmed: {len(confirmed_non_events)}\n")

    # Summaries
    print("\n=== FINAL RESULTS ===")
    print(f"Total Confirmed Events: {len(all_events)}")
    #print(f"  - Recognized Species: {len(final_event_list)}")
    #print(f"  - Unknown Species: {len(unknown_species)}")
    print(f"Total Confirmed Non-Events: {len(confirmed_non_events)}")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nProcessing complete in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
