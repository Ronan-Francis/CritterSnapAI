import os
import time
from config_manager import ConfigManager
from file_utils import sort_images_by_date_time, group_images_by_event, create_event_directories
from classifier import train_animal_classifier, predict_image
from event_detector import select_best_photo_in_group
from shutil import copy2

def run_pipeline():
    
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
    start_time = time.time()
    images = sort_images_by_date_time(config["directory_path"], config["white_pixel_threshold"])
    groups = group_images_by_event(images)
    print(f"Total groups formed: {len(groups)}")

    # Best photo detection: For each temporal group, select only the best image based on composite score.
    best_events = []
    threshold_score = 5.0  # Only consider images with a composite score above this threshold.
    for group in groups:
        best_photo, score = select_best_photo_in_group(group, config["change_threshold"], config["edge_confidence_threshold"])
        if best_photo and score > threshold_score:
            best_events.append(best_photo)
    print(f"Best photo detection: {len(best_events)} events selected from {len(groups)} groups.")

    # Train classifier and perform second pass on best event candidates.
    print("Training animal classifier...")
    classifier = train_animal_classifier(config["animal_training_path"])
    final_events = []
    confirmed_non_events = []
    for idx, item in enumerate(best_events, start=1):
        label = predict_image(item.get_file_path(), classifier)
        if label == "Animal":
            final_events.append(item)
        else:
            confirmed_non_events.append(item)
        print(f"Second-pass analysis: {idx}/{len(best_events)} complete", end="\r")
    print()  # New line after progress output.

    # Write log of results.
    with open(config["output_log_path"], "w") as log_file:
        log_file.write(f"Events Confirmed (All): {len(final_events)}\n")
        log_file.write(f"Non-Events Confirmed: {len(confirmed_non_events)}\n")
    
    print(f"Total confirmed events: {len(final_events)}")
    print(f"Total confirmed non-events: {len(confirmed_non_events)}")
    # Create the destination directory for final events.
    final_events_output_dir = os.path.join(config["output_directory"], "final_events")
    os.makedirs(final_events_output_dir, exist_ok=True)

    # Copy each final event's file into the destination directory.
    for event in final_events:
        src = event.get_file_path()
        copy2(src, final_events_output_dir)
        print(f"Copied {src} to {final_events_output_dir}")

    elapsed = time.time() - start_time
    print(f"Processing complete in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_pipeline()
