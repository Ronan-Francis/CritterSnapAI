import os
from datetime import datetime, timedelta
from PIL import Image
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from shutil import copy2
from typing import List, Optional

from gdpr_utils import is_not_gdpr_image
from data_structures import ImageObject

@lru_cache(maxsize=100)
def load_image_cached(file_path: str) -> Image.Image:
    """Load an image from disk with caching."""
    return Image.open(file_path)

def get_images_from_folder(folder_path: str) -> List[str]:
    """Return a list of file names from the specified folder."""
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

def load_images_in_parallel(image_paths: List[str]) -> List[Image.Image]:
    """Load images in parallel using a thread pool."""
    with ThreadPoolExecutor() as executor:
        return list(executor.map(load_image_cached, image_paths))

def extract_date_time_from_exif(filepath: str) -> Optional[datetime]:
    """
    Extract the datetime from an image's EXIF data.
    Falls back to the file's modification time if EXIF is unavailable.
    """
    try:
        with Image.open(filepath) as img:
            exif_data = img._getexif()
            if exif_data:
                # Check multiple potential EXIF tags for date/time.
                for tag in (36867, 306, 36868):  # DateTimeOriginal, DateTime, DateTimeDigitized
                    date_time_str = exif_data.get(tag)
                    if date_time_str:
                        return datetime.strptime(date_time_str, "%Y:%m:%d %H:%M:%S")
        return datetime.fromtimestamp(os.path.getmtime(filepath))
    except Exception as e:
        print(f"Error reading EXIF data from {filepath}: {e}")
        return None

def sort_images_by_date_time(folder_path: str, white_pixel_threshold: int) -> List[ImageObject]:
    """
    Loads images from a directory, filters out GDPR-protected ones,
    extracts their date/time, and returns a sorted list of ImageObjects.
    """
    print(f"Sorting images by date/time in {folder_path}")
    image_files = get_images_from_folder(folder_path)
    image_paths = [os.path.join(folder_path, image_name) for image_name in image_files]
    images = load_images_in_parallel(image_paths)

    images_with_dates: List[ImageObject] = []
    total_images = len(image_files)
    for idx, (img, path) in enumerate(zip(images, image_paths), start=1):
        print(f"Processing images ({idx}/{total_images}) - {(idx / total_images * 100):.2f}% complete", end="\r")
        if is_not_gdpr_image(img, white_pixel_threshold):
            date_time = extract_date_time_from_exif(path)
            if date_time:
                images_with_dates.append(ImageObject(img.copy(), date_time, path))
        else:
            print(f"GDPR-protected image found: {path}")
    print("Image sorting complete.                ")
    images_with_dates.sort(key=lambda x: x.get_date())
    return images_with_dates

def group_images_by_event(images_with_dates: List[ImageObject], time_gap_threshold: timedelta = timedelta(hours=1)) -> List[List[ImageObject]]:
    """
    Groups sorted images into events based on a time gap threshold.
    Returns a list of image groups.
    """
    print("Grouping images into events based on time gaps")
    grouped_events: List[List[ImageObject]] = []
    current_event: List[ImageObject] = []

    for image_obj in images_with_dates:
        if not current_event:
            current_event.append(image_obj)
        else:
            if image_obj.get_date() - current_event[-1].get_date() > time_gap_threshold:
                grouped_events.append(current_event)
                current_event = [image_obj]
            else:
                current_event.append(image_obj)
    if current_event:
        grouped_events.append(current_event)

    print(f"Total number of groups: {len(grouped_events)}")
    return grouped_events

def create_event_directories(grouped_events: List[List[ImageObject]], output_directory: str) -> None:
    """
    Copies images from grouped events into subdirectories under the output directory.
    Each event group is saved in a separate folder.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, event_group in enumerate(grouped_events):
        event_directory = os.path.join(output_directory, f"event_{i}")
        os.makedirs(event_directory, exist_ok=True)
        for image_obj in event_group:
            src_path = image_obj.get_file_path()
            dest_path = os.path.join(event_directory, os.path.basename(src_path))
            copy2(src_path, dest_path)
