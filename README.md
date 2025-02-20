# CritterSnap: Image Motion & Animal Event Detection

CritterSnap is a Python-based project designed to process and analyze image sequences for detecting motion events and classifying animal presence. The system automates the workflow of sorting images by capture time, grouping them into events based on temporal proximity, and applying multi-step event detection using both image processing techniques and machine learning.

---

## Overview

The project performs the following steps:

1. **Image Preprocessing & Sorting**  
   Images are loaded from a specified directory, filtered to remove GDPR-protected images (based on white pixel thresholds), and sorted by their date/time metadata extracted from EXIF or file modification timestamps.  
   *See [sorting_utils.py](&#8203;:contentReference[oaicite:0]{index=0}) and [gdpr_utils.py](&#8203;:contentReference[oaicite:1]{index=1}).*

2. **Grouping by Event**  
   Sorted images are grouped into events if the time difference between consecutive images is less than a defined threshold (e.g., one hour).  
   *Refer to [sorting_utils.py](&#8203;:contentReference[oaicite:2]{index=2}).*

3. **Motion Detection & Event Classification**  
   Each image group is analyzed in parallel using a two-tier approach:
   - **First Pass:** Uses structural similarity (SSIM) and edge detection methods to compute a composite score for identifying high-confidence events.  
     *See [edge_detection.py](&#8203;:contentReference[oaicite:3]{index=3}) and [image_utils.py](&#8203;:contentReference[oaicite:4]{index=4}).*
   - **Second Pass:** Applies a machine learning classifier (trained with a One-Class SVM) to re-evaluate images that did not meet the initial threshold, confirming animal presence when applicable.  
     *See [classification.py](&#8203;:contentReference[oaicite:5]{index=5}) and [sklearn_classifier.py](&#8203;:contentReference[oaicite:6]{index=6}).*

4. **Parallel Processing**  
   The project leverages Python’s `concurrent.futures` to parallelize both event detection and image loading, ensuring efficient processing of large datasets.  
   *Details can be found in [main.py](&#8203;:contentReference[oaicite:7]{index=7}) and [sorting_utils.py](&#8203;:contentReference[oaicite:8]{index=8}).*

5. **Logging & Output**  
   After processing, results including confirmed events and non-events are logged to a user-specified output file, with options for further image saving and directory organization.

---

## Features

- **Automated Image Sorting:**  
  Automatically organizes images by timestamp, taking into account missing or incomplete EXIF data.  
  *[sorting_utils.py](&#8203;:contentReference[oaicite:9]{index=9})*

- **GDPR Compliance Check:**  
  Filters out images that may be GDPR-protected by counting pure white pixels.  
  *[gdpr_utils.py](&#8203;:contentReference[oaicite:10]{index=10})*

- **Edge Detection & SSIM Analysis:**  
  Utilizes Sobel edge detection and the Structural Similarity Index Measure (SSIM) to quantify pixel changes and edge features in images.  
  *[edge_detection.py](&#8203;:contentReference[oaicite:11]{index=11}) and [image_utils.py](&#8203;:contentReference[oaicite:12]{index=12})*

- **Machine Learning Classification:**  
  Trains an animal classifier using a One-Class SVM on pre-labeled animal images to improve detection accuracy.  
  *[sklearn_classifier.py](&#8203;:contentReference[oaicite:13]{index=13})*

- **Parallel Processing:**  
  Employs multi-threading and multi-processing to handle large sets of images efficiently.  
  *[main.py](&#8203;:contentReference[oaicite:14]{index=14}) and [sorting_utils.py](&#8203;:contentReference[oaicite:15]{index=15})*

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
