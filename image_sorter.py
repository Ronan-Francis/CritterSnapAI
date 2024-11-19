import os
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
        image_path = os.path.join(folder_path, image)
        date_time = extract_date_time_from_exif(image_path)
        if date_time:
            with Image.open(image_path) as img:
                images_with_dates.append((img.copy(), date_time))
    
    # Sort images by date and time
    images_with_dates.sort(key=lambda x: x[1])
    
    return images_with_dates

