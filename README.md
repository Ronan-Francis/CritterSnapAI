# CritterSnap: Image Motion & Animal Event Detection

CritterSnap is a Python-based project designed to process and analyze image sequences for detecting motion events and classifying animal presence. The system automates the workflow of sorting images by capture time, grouping them into events based on temporal proximity, and applying multi-step event detection using both image processing techniques and machine learning.

---

## Overview

The project performs the following steps:

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

4. **Parallel Processing**  
   The project leverages Python’s `concurrent.futures` to parallelize both event detection and image loading, ensuring efficient processing of large datasets.  
   *Details can be found in [main.py](main.py) and [sorting_utils.py](sorting_utils.py).*

5. **Interactive Configuration & Logging**
An interactive configuration menu allows users to update various parameters (such as directory paths, thresholds, and URLs) at runtime. Results are logged to a specified file, and images can be organized into directories for further inspection.
*Configuration is managed via [config.py](config.py).*

---

## Features

- **Automated Image Sorting:**  
  Automatically organizes images by timestamp, taking into account missing or incomplete EXIF data.  
  *[sorting_utils.py](sorting_utils.py)*

- **GDPR Compliance Check:**  
  Filters out images that may be GDPR-protected by counting pure white pixels.  
  *[gdpr_utils.py](gdpr_utils.py)*

- **Edge Detection & SSIM Analysis:**  
  Utilizes Sobel edge detection and the Structural Similarity Index Measure (SSIM) to quantify pixel changes and edge features in images.  
  *[edge_detection.py](edge_detection.py) and [image_utils.py](image_utils.py)*

- **Machine Learning Classification:**  
  Trains an animal classifier using a One-Class SVM on pre-labeled animal images to improve detection accuracy.  
  *[sklearn_classifier.py](sklearn_classifier.py)*

- **Parallel Processing:**  
  Employs multi-threading and multi-processing to handle large sets of images efficiently.  
  *[main.py](main.py) and [sorting_utils.py](sorting_utils.py)*

- **Interactive Configuration:**
   An interactive menu in [config.py](config.py) lets users customize parameters, including paths and thresholds, ensuring flexibility across different datasets and environments.

Structured Data Management:
The ImageObject class encapsulated in [data_structures.py](data_structures.py) neatly stores image data along with its timestamp and file path for easier processing and logging.

---

## Requirements

- **Python 3.6+**
- **Libraries:**  
  - Pillow  
  - NumPy  
  - scikit-image  
  - scikit-learn  
  - joblib  
  - (Other standard libraries such as `os`, `datetime`, `concurrent.futures`)

Install the required packages using pip:

```bash
pip install Pillow numpy scikit-image scikit-learn joblib
```
## Setup & Configuration


## Project Structure
**`config.py:`**
Contains configurable parameters such as file paths, thresholds, and URLs.

**`data_structures.py:`**
Defines the ImageObject class used for encapsulating image data along with capture date/time and file path.

**`edge_detection.py:`**
Implements functions for converting images to grayscale and computing Sobel gradients for edge detection.

**`gdpr_utils.py:`**
Provides utilities to check if an image is GDPR-protected based on white pixel count.
gdpr_utils.py

**`image_utils.py:`**
Includes functions for image downsampling and measuring image differences using SSIM.

**`main.py:`**
The entry point that orchestrates image sorting, grouping, motion detection, and classification.

**`classification.py:`**
Contains the logic for classifying images as events or non-events by combining SSIM and edge detection metrics.

**`sklearn_classifier.py:`**
Manages the training and prediction of the animal classifier using a One-Class SVM.

**`sorting_utils.py:`**
Handles image loading, EXIF extraction, sorting, and grouping based on time gaps.
