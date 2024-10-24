from load_images import load_images_from_directory
from decision_tree import decision_tree
from event_processing import copy_events_with_red_box

def main():
    directory_path = r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\example"  # Adjust this with your correct path
    output_directory = r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\output"  # Adjust this with your correct path
    images = load_images_from_directory(directory_path)
    
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
