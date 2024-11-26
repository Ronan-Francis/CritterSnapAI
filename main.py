from config import directory_path, output_directory, change_threshold, white_pixel_threshold, output_log_path
from image_sorter import sort_images_by_date_time, group_images_by_event, create_event_directories
from decision_tree import decision_tree
from imageObj import ImageObject
import os
from PIL import Image
from datetime import datetime

def main():
    # Sort images by date and time
    print("Starting the image sorting and classification process...")
    images_with_dates = sort_images_by_date_time(directory_path)
    
    # Create ImageObject instances
    image_objects = [ImageObject(Image.open(img[0]), img[1], img[0]) for img in images_with_dates]
    
    if len(image_objects) < 3:
        print("Not enough images to perform comparison (need at least 3).")
        return

    # Run the decision tree logic to classify images
    events, non_events = decision_tree(image_objects, change_threshold, white_pixel_threshold, output_directory)
    
    # Output the classification results
    print(f"\nTotal Events Detected: {len(events)}")
    print(f"Total Non-Events Detected: {len(non_events)}")
    
    # Log events to output_log.txt
    with open(output_log_path, 'w') as log_file:
        log_file.write(f"Total Events Detected: {len(events)}\n")
        log_file.write(f"Total Non-Events Detected: {len(non_events)}\n")
        log_file.write("\nEvents:\n")
        for event in events:
            log_file.write(f"{event.get_file_path()}\n")

if __name__ == "__main__":
    main()
