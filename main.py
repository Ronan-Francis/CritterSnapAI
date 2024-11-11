from decision_tree import decision_tree
from event_processing import copy_events_with_red_box
from image_sorter import sort_images_by_date_time, write_sorted_images_to_log

def main():
    directory_path = r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\example\as_conservationistFrankfurt\IE_Forest_County_Wicklow_21_loc_01-20241031T145429Z-001\IE_Forest_County_Wicklow_21_loc_01"  # Adjust this with your correct path
    output_directory = r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\output" # Adjust this with your correct path
    output_log_path = r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\output\output_log.txt"  # Adjust this with your correct path
    
    # Sort images by date and time
    sorted_images = sort_images_by_date_time(directory_path)
    write_sorted_images_to_log(sorted_images, output_log_path)
    print(f"Sorted image list has been written to {output_log_path}")
    
    # Load images after sorting
    images = [img[0] if isinstance(img, tuple) else img for img in sorted_images]
    
    if len(images) < 3:
        print("Not enough images to perform comparison (need at least 3).")
        return

    # Define a suitable change threshold for your dataset
    change_threshold = 10000  # Adjust this based on your images
    
    # Run the decision tree logic to classify images
    events, non_events = decision_tree(images, change_threshold=change_threshold)
    
    # Output the classification results
    print(f"\nTotal Events Detected: {len(events)}")
    print(f"Total Non-Events Detected: {len(non_events)}")
    
    # Copy events to the output folder with a red box around the suspected area
    copy_events_with_red_box(events, output_directory)

if __name__ == "__main__":
    main()
