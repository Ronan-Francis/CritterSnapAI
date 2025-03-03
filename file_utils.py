import os
from datetime import datetime, timedelta
from PIL import Image
from shutil import copy2
from concurrent.futures import ThreadPoolExecutor
from image_processor import ImageObject
from gdpr_utils import is_not_gdpr_image  # This function is defined in gdpr_utils.py

def get_images_from_folder(folder_path: str):
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

def load_images_in_parallel(image_paths):
    with ThreadPoolExecutor() as executor:
        return list(executor.map(lambda fp: Image.open(fp), image_paths))

def extract_date_time_from_exif(filepath: str):
    try:
        with Image.open(filepath) as img:
            exif_data = img._getexif()
            if exif_data:
                for tag in (36867, 306, 36868):  # DateTimeOriginal, DateTime, DateTimeDigitized
                    date_time_str = exif_data.get(tag)
                    if date_time_str:
                        return datetime.strptime(date_time_str, "%Y:%m:%d %H:%M:%S")
        return datetime.fromtimestamp(os.path.getmtime(filepath))
    except Exception as e:
        print(f"Error reading EXIF data from {filepath}: {e}")
        return None

def sort_images_by_date_time(folder_path: str, white_pixel_threshold: int):
    image_files = get_images_from_folder(folder_path)
    image_paths = [os.path.join(folder_path, f) for f in image_files]
    images = load_images_in_parallel(image_paths)
    images_with_dates = []
    total_images = len(image_files)
    for idx, (img, path) in enumerate(zip(images, image_paths), start=1):
        print(f"Processing images ({idx}/{total_images}) - {(idx / total_images * 100):.2f}% complete", end="\r")
        if is_not_gdpr_image(img, white_pixel_threshold):
            dt = extract_date_time_from_exif(path)
            if dt:
                images_with_dates.append(ImageObject(img.copy(), dt, path))
        else:
            print(f"GDPR-protected image found: {path}")
    print("Image sorting complete.                ")
    images_with_dates.sort(key=lambda x: x.get_date())
    return images_with_dates

def group_images_by_event(images_with_dates, time_gap_threshold=timedelta(hours=1)):
    grouped_events = []
    current_group = []
    for image_obj in images_with_dates:
        if not current_group:
            current_group.append(image_obj)
        else:
            if image_obj.get_date() - current_group[-1].get_date() > time_gap_threshold:
                grouped_events.append(current_group)
                current_group = [image_obj]
            else:
                current_group.append(image_obj)
    if current_group:
        grouped_events.append(current_group)
    print(f"Total number of groups: {len(grouped_events)}")
    return grouped_events

def create_event_directories(grouped_events, output_directory: str):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for i, group in enumerate(grouped_events):
        event_dir = os.path.join(output_directory, f"event_{i}")
        os.makedirs(event_dir, exist_ok=True)
        for image_obj in group:
            src = image_obj.get_file_path()
            dest = os.path.join(event_dir, os.path.basename(src))
            copy2(src, dest)
