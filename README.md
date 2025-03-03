# CritterSnap: Image Motion & Animal Event Detection

CritterSnap is a Python-based project designed to process and analyze image sequences for detecting motion events and classifying animal presence. The system automates the workflow of sorting images by capture time, grouping them into events based on temporal proximity, and applying multi-step event detection using both image processing techniques and machine learning.

---

## Overview

CritterSnap performs the following steps:

1. **Image Preprocessing & Sorting**  
   Images are loaded from a specified directory, filtered to remove GDPR-protected images (based on white pixel thresholds), and sorted by their date/time metadata extracted from EXIF or file modification timestamps.  
   *See [sorting_utils.py](sorting_utils.py) and [gdpr_utils.py](gdpr_utils.py).*

2. **Grouping by Event**  
   Sorted images are grouped into events if the time difference between consecutive images is less than a defined threshold (e.g., one hour).  
   *Refer to [sorting_utils.py](sorting_utils.py).*

3. **Motion Detection & Event Classification**  
   Each image group is analyzed in parallel using a two-tier approach:
   - **First Pass:** Uses structural similarity (SSIM) and edge detection methods to compute a composite score for identifying high-confidence events.  
     *See [edge_detection.py](edge_detection.py) and [image_utils.py](image_utils.py).*
   - **Second Pass:** Applies a machine learning classifier (trained with a One-Class SVM) to re-evaluate images that did not meet the initial threshold, confirming animal presence when applicable.  
     *See [classification.py](classification.py) and [sklearn_classifier.py](sklearn_classifier.py).*

4. **Dynamic Thresholding**  
   An optional dynamic thresholding step computes new thresholds based on a random sample of the animal training data, adjusting the edge confidence and SSIM change thresholds automatically.  
   *Refer to [dynamic_thresholds.py](dynamic_thresholds.py).*

5. **Parallel Processing**  
   The project leverages Python’s `concurrent.futures` to parallelize both event detection and image loading, ensuring efficient processing of large datasets.  
   *Details can be found in [main.py](main.py) and [sorting_utils.py](sorting_utils.py).*

6. **Interactive Configuration & Logging**  
   An interactive configuration menu allows users to update various parameters (such as directory paths, thresholds, and URLs) at runtime. Results are logged to a specified file, and images can be organized into directories for further inspection.  
   *Configuration is managed via [config.py](config.py).*

7. **Structured Data Management**  
   The `ImageObject` class (found in [data_structures.py](data_structures.py)) encapsulates image data along with its timestamp and file path, simplifying downstream processing and logging.

---

## Features

- **Automated Image Sorting:**  
  Automatically organizes images by timestamp, even when EXIF data is missing or incomplete.  
  *See [sorting_utils.py](sorting_utils.py).*

- **GDPR Compliance Check:**  
  Filters out images that may be GDPR-protected by counting pure white pixels.  
  *See [gdpr_utils.py](gdpr_utils.py).*

- **Edge Detection & SSIM Analysis:**  
  Utilizes Sobel edge detection and the Structural Similarity Index Measure (SSIM) to quantify pixel changes and edge features in images.  
  *See [edge_detection.py](edge_detection.py) and [image_utils.py](image_utils.py).*

- **Dynamic Thresholding:**  
  Optionally compute new thresholds based on a sample of animal training images to fine-tune detection parameters.  
  *See [dynamic_thresholds.py](dynamic_thresholds.py).*

- **Machine Learning Classification:**  
  Trains an animal classifier using a One-Class SVM on pre-labeled animal images to improve detection accuracy.  
  *See [sklearn_classifier.py](sklearn_classifier.py).*

- **Parallel Processing:**  
  Employs multi-threading and multi-processing to handle large image datasets efficiently.  
  *See [main.py](main.py) and [sorting_utils.py](sorting_utils.py).*

- **Interactive Configuration:**  
  An interactive menu in [config.py](config.py) lets users customize parameters such as file paths and thresholds, ensuring flexibility across different datasets and environments.

---

## Requirements

- **Python 3.6+**
- **Required Libraries:**
  - Pillow
  - NumPy
  - scikit-image
  - scikit-learn
  - joblib
  - (Other standard libraries such as `os`, `datetime`, and `concurrent.futures`)

Install the required packages using pip:

```bash
pip install Pillow numpy scikit-image scikit-learn joblib
