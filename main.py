from config import directory_path, output_directory, change_threshold
from image_sorter import sort_images_by_date_time, group_images_by_event
from decision_tree import create_event_directories, decision_tree

def main():
    # Sort images by date and time
    print("Starting the image sorting and classification process...")
    images_with_dates = sort_images_by_date_time(directory_path)
    
    # Group images by event
    # events = group_images_by_event(images_with_dates)

    # Load images after sorting
    images = [img[0] if isinstance(img, tuple) else img for img in images_with_dates]
    
    if len(images) < 3:
        print("Not enough images to perform comparison (need at least 3).")
        return

    # Run the decision tree logic to classify images
    events, non_events = decision_tree(images, change_threshold)
    
    # Output the classification results
    print(f"\nTotal Events Detected: {len(events)}")
    print(f"Total Non-Events Detected: {len(non_events)}")
    
    # Create directories for each event and move the corresponding images
    create_event_directories(events, output_directory)

if __name__ == "__main__":
    main()
