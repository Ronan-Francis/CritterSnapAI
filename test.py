import urllib.request
import os
from config import output_directory, base_url, start_file, end_file

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Download files in sequence
for i in range(start_file, end_file):
    file_name = f"{i:04d}.jpg"
    file_url = f"{base_url}{file_name}"
    output_path = os.path.join(output_directory, file_name)

    try:
        urllib.request.urlretrieve(file_url, output_path)
        print(f"Downloaded: {file_name}")
    except Exception as e:
        print(f"Failed to download {file_name}: {e}")

