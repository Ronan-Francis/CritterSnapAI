import os
from datetime import datetime, timedelta
from PIL import Image
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from shutil import copy2

from gdpr_utils import is_not_gdpr_image
from data_structures import ImageObject

@lru_cache(maxsize=100)
def load_image_cached(file_path):
    """
    Loads an image from disk using a cache decorator to avoid reloading.
    """
    return Image.open(file_path)

def get_images_from_folder(folder_path):
    """
    Retrieves a list of file names from the specified folder.
    """
    return [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

def load_images_in_parallel(image_paths):
    """
    Loads images in parallel using a thread pool.
    """
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(load_image_cached, image_paths))
    return images

def extract_date_time_from_exif(filepath):
    """
    Extracts the datetime from an image file’s EXIF data if available.
    Falls back to file modification time if no EXIF data is found.
    """
    try:
        with Image.open(filepath) as img:
            exif_data = img._getexif()
            if exif_data:
                date_time_str = exif_data.get(36867)  # DateTimeOriginal
                if not date_time_str:
                    date_time_str = exif_data.get(306)  # DateTime
                if not date_time_str:
                    date_time_str = exif_data.get(36868)  # DateTimeDigitized

                if date_time_str:
                    # Convert from "YYYY:MM:DD HH:MM:SS" format
                    return datetime.strptime(date_time_str, "%Y:%m:%d %H:%M:%S")

        file_mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        return file_mod_time

    except Exception as e:
        print(f"Error reading EXIF data from {filepath}: {e}")
        return None

def sort_images_by_date_time(folder_path, white_pixel_threshold):
    """
    1. Loads images from a directory
    2. Filters out GDPR-protected images
    3. Extracts date/time
    4. Sorts by date/time
    5. Returns a list of tuples (ImageObject)
    """
    print(f"Sorting images by date/time in {folder_path}")
    image_files = get_images_from_folder(folder_path)
    image_paths = [os.path.join(folder_path, image_name) for image_name in image_files]
    
    # Load images in parallel
    images = load_images_in_parallel(image_paths)
    
    images_with_dates = []
    for idx, (img, image_path) in enumerate(zip(images, image_paths), start=1):
        print(f"Processing ({idx}/{len(image_files)}) - "
              f"{(idx / len(image_files)) * 100:.2f}% complete", end="\r")

        if is_not_gdpr_image(img, white_pixel_threshold):
            date_time = extract_date_time_from_exif(image_path)
            if date_time:
                # Build an ImageObject
                images_with_dates.append(ImageObject(img.copy(), date_time, image_path))

    images_with_dates.sort(key=lambda x: x.get_date())
    print("Image sorting complete.                ")
    return images_with_dates

def group_images_by_event(images_with_dates, time_gap_threshold=timedelta(hours=1)):
    """
    Groups sorted images into event clusters if consecutive images are within time_gap_threshold.
    Returns a list of lists (each sublist is a “group” or “event”).
    """
    print("Grouping images into events based on time gaps")
    grouped_events = []
    current_event = []

    for image_obj in images_with_dates:
        if not current_event:
            current_event.append(image_obj)
        else:
            last_date_time = current_event[-1].get_date()
            if image_obj.get_date() - last_date_time > time_gap_threshold:
                # Start a new event group
                grouped_events.append(current_event)
                current_event = [image_obj]
            else:
                current_event.append(image_obj)

    # Append the last event
    if current_event:
        grouped_events.append(current_event)

    print(f"Total number of groups: {len(grouped_events)}")
    return grouped_events

def create_event_directories(grouped_events, output_directory):
    """
    Copies images from the grouped events into a specified output directory.

    Parameters:
    - grouped_events: List of lists of ImageObject, where each sublist is an event group.
    - output_directory: Directory to store the event images.

    Behavior:
    - For each event group, create a subfolder (e.g., event_0, event_1, ...).
    - Copy all images in that group into the subfolder.
    - Leaves the original files in place (no deletion or moving).
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, event_group in enumerate(grouped_events):
        event_directory = os.path.join(output_directory, f"event_{i}")
        os.makedirs(event_directory, exist_ok=True)

        for image_obj in event_group:
            # Retrieve the original file path
            src_path = image_obj.get_file_path()
            # Derive the destination file path
            dest_filename = os.path.basename(src_path)
            dest_path = os.path.join(event_directory, dest_filename)
            
            # Copy the file instead of moving it or deleting the source
            copy2(src_path, dest_path)