import os
import time
from config_manager import ConfigManager
from file_utils import sort_images_by_date_time, group_images_by_event, create_event_directories
from event_detector import detect_motion
from classifier import train_animal_classifier, predict_image

def run_pipeline():
    start_time = time.time()
    
    # Load and update configuration.
    config_mgr = ConfigManager()
    update_choice = input("Do you want to update the configuration? (y/n, default n): ").strip().lower()
    if update_choice == "y":
        config_mgr.interactive_update()
    config = config_mgr.config

    # Optionally apply dynamic thresholding.
    dynamic_choice = input("Apply dynamic thresholding using a sample of animal training data? (y/n, default n): ").strip().lower()
    if dynamic_choice == "y":
        from dynamic_thresholds import compute_dynamic_thresholds
        config = compute_dynamic_thresholds(config)

    # Sort and group images.
    print("Sorting images...")
    images = sort_images_by_date_time(config["directory_path"], config["white_pixel_threshold"])
    groups = group_images_by_event(images)
    print(f"Total groups formed: {len(groups)}")

    # First pass event detection.
    detected_events = []
    low_conf_candidates = []
    for group in groups:
        events, non_events = detect_motion(group, config["change_threshold"], config["edge_confidence_threshold"])
        detected_events.extend(events)
        low_conf_candidates.extend(non_events)
    print(f"First-pass: {len(detected_events)} events, {len(low_conf_candidates)} non-events.")

    # Train classifier and perform second pass.
    print("Training animal classifier...")
    classifier = train_animal_classifier(config["animal_training_path"])
    second_pass_events = []
    confirmed_non_events = []
    for idx, item in enumerate(low_conf_candidates, start=1):
        label = predict_image(item.get_file_path(), classifier)
        if label == "Animal":
            second_pass_events.append(item)
        else:
            confirmed_non_events.append(item)
        print(f"Second-pass analysis: {idx}/{len(low_conf_candidates)} complete", end="\r")
    print()  # New line after progress output.
    all_events = detected_events + second_pass_events

    # Write log of results.
    with open(config["output_log_path"], "w") as log_file:
        log_file.write(f"Events Confirmed (All): {len(all_events)}\n")
        log_file.write(f"Non-Events Confirmed: {len(confirmed_non_events)}\n")
    
    print(f"Total confirmed events: {len(all_events)}")
    print(f"Total confirmed non-events: {len(confirmed_non_events)}")

    elapsed = time.time() - start_time
    print(f"Processing complete in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_pipeline()
