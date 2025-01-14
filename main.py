from config import directory_path, output_directory, change_threshold, white_pixel_threshold, output_log_path
from image_sorter import sort_images_by_date_time, group_images_by_event
from decision_tree import process_image
from imageObj import ImageObject
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import shutil

def process_group(group, change_threshold, white_pixel_threshold, output_directory):
    # Convert tuples to ImageObject instances if necessary
    if isinstance(group[0], tuple):
        group = [ImageObject(img, date, path) for img, date, path in group]

    events = []
    non_events = []
    for index in range(1, len(group) - 1):
        event, non_event = process_image(group, index, change_threshold)
        if event:
            events.append(event)
        if non_event:
            non_events.append(non_event)

    return events, non_events


def main():
    start_time = time.time()  # Record the start time
    # Sort images by date and time
    print("Starting the image sorting process...")
    images_with_dates = sort_images_by_date_time(directory_path)

    # Group images by event
    grouped_events = group_images_by_event(images_with_dates)

    all_events = []
    all_non_events = []

    # Parallel processing of groups using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), len(grouped_events))) as executor:
        futures = {
            executor.submit(process_group, group, change_threshold, white_pixel_threshold, output_directory): group 
            for group in grouped_events
        }
        for i, future in enumerate(as_completed(futures)):
            events, non_events = future.result()
            all_events.extend(events)
            all_non_events.extend(non_events)

            percentage_complete = ((i + 1) / len(grouped_events)) * 100
            print(f"Processed: {i + 1}/{len(grouped_events)} ({percentage_complete:.2f}% complete)", end="\r")

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"\nEvent sorting and classification process completed.")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")

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
