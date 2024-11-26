import os
import shutil
from PIL import Image
from datetime import datetime, timedelta

def get_images_from_folder(folder_path):
    # Fetch all files in the directory
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

def extract_date_time_from_exif(filepath):
    try:
        # Open the image file
        with Image.open(filepath) as img:
            exif_data = img._getexif()
            if exif_data:
                # Extract the DateTimeOriginal tag if available
                date_time_str = exif_data.get(36867)  # Tag 36867 is DateTimeOriginal in EXIF
                if not date_time_str:
                    date_time_str = exif_data.get(306)  # Tag 306 is DateTime in EXIF
                if date_time_str:
                    # Convert EXIF date format "YYYY:MM:DD HH:MM:SS" to datetime object
                    return datetime.strptime(date_time_str, "%Y:%m:%d %H:%M:%S")
            print(f"No EXIF date found for {filepath}")
            return None
    except Exception as e:
        print(f"Error reading EXIF data from {filepath}: {e}")
        return None

def sort_images_by_date_time(folder_path):
    print(f"Sorting images by date and time in {folder_path}")
    images = get_images_from_folder(folder_path)
    images_with_dates = []
    
    for image in images:
        print(f"Processing {image}")
        image_path = os.path.join(folder_path, image)
        date_time = extract_date_time_from_exif(image_path)
        if date_time:
            with Image.open(image_path) as img:
                images_with_dates.append((img.copy(), date_time))
    
    # Sort images by date and time
    images_with_dates.sort(key=lambda x: x[1])
    
    return images_with_dates

def group_images_by_event(images_with_dates, time_gap_threshold=timedelta(hours=1)):
    print("Grouping images into events based on time gaps")
    grouped_events = []
    current_event = []

    for i, (image, date_time) in enumerate(images_with_dates):
        if not current_event:
            # Start a new event
            current_event.append((image, date_time))
        else:
            # Compare time difference with the last image in the current event
            _, last_date_time = current_event[-1]
            if date_time - last_date_time > time_gap_threshold:
                # Big gap detected; finalize the current event
                grouped_events.append(current_event)
                current_event = [(image, date_time)]
            else:
                # Continue the current event
                current_event.append((image, date_time))

    # Add the last event if not empty
    if current_event:
        grouped_events.append(current_event)

    return grouped_events

def create_event_directories(grouped_events, output_directory):
    """
    Creates directories for each event and moves the corresponding images.

    Parameters:
    - grouped_events: List of events, where each event is a list of image file paths.
    - output_directory: Directory to store event images.
    """
def create_event_directories(grouped_events, output_directory):
    """
    Creates directories for each event and moves the corresponding images.

    Parameters:
    - grouped_events: List of events, where each event is a list of image file paths.
    - output_directory: Directory to store event images.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, event in enumerate(grouped_events):
        event_directory = os.path.join(output_directory, f"event_{i}")
        if not os.path.exists(event_directory):
            os.makedirs(event_directory)

        for image_path in event:
            image_name = os.path.basename(image_path)
            destination_path = os.path.join(event_directory, image_name)

            # Moving the file 
            with open(image_path, 'rb') as source_file:
                with open(destination_path, 'wb') as destination_file:
                    destination_file.write(source_file.read())

            # Remove the original file after copying
            os.remove(image_path)