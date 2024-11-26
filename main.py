from config import directory_path, output_directory, change_threshold, white_pixel_threshold, output_log_path
from image_sorter import sort_images_by_date_time, group_images_by_event, create_event_directories
from decision_tree import decision_tree
from imageObj import ImageObject
import os
from PIL import Image
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_group(group, change_threshold, white_pixel_threshold, output_directory):
    image_objects = [ImageObject(img[0], img[1], img[0]) for img in group]
    if len(image_objects) < 3:
        return [], []

    # Optimize decision tree processing (assumes decision_tree itself is efficient)
    events, non_events = decision_tree(image_objects, change_threshold, white_pixel_threshold, output_directory)
    return events, non_events

def main():
    # Sort images by date and time
    print("Starting the image sorting and classification process...")
    images_with_dates = sort_images_by_date_time(directory_path)
    print("Image sorting complete.")

    # Group images by event
    print("Grouping images into events based on time gaps...")
    grouped_events = group_images_by_event(images_with_dates)

    all_events = []
    all_non_events = []

    # Parallel processing of groups
    with ThreadPoolExecutor(max_workers=min(os.cpu_count() * 2, len(grouped_events))) as executor:
        futures = {executor.submit(process_group, group, change_threshold, white_pixel_threshold, output_directory): group for group in grouped_events}
        for i, future in enumerate(as_completed(futures)):
            events, non_events = future.result()
            all_events.extend(events)
            all_non_events.extend(non_events)

            if i % max(1, len(grouped_events) // 10) == 0:  # Update every 10%
                percentage_complete = (i / len(grouped_events)) * 100
                print(f"Processed: {i}/{len(grouped_events)} ({percentage_complete:.2f}% complete)", end="\r")

    print(f"\nImage sorting and classification process completed.")

    # Output the classification results
    print(f"Total Events Detected: {len(all_events)}")
    print(f"Total Non-Events Detected: {len(all_non_events)}")

    # Log events to output_log.txt
    with open(output_log_path, 'w') as log_file:
        log_file.write(f"Total Events Detected: {len(all_events)}\n")
        log_file.write(f"Total Non-Events Detected: {len(all_non_events)}\n\nEvents:\n")
        log_file.writelines(f"{event.get_file_path()}\n" for event in all_events)

if __name__ == "__main__":
    main()
