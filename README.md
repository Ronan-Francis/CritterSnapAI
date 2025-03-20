# CritterSnap: Image Motion & Animal Event Detection

CritterSnap is a Python-based pipeline for processing image sequences to detect motion events and classify animal presence. The system automates the workflow of sorting images by capture time, grouping them into temporal events, selecting the best representative photo from each event, and applying multi-step event detection with both image processing techniques and machine learning.

---

## Overview

CritterSnap performs the following steps:

1. **Image Preprocessing & Sorting**  
   Images are loaded from a specified directory, filtered to remove GDPR-protected images (using a white pixel threshold), and sorted by their date/time metadata extracted from EXIF data or file modification timestamps.  
   *See [file_utils.py](file_utils.py) and [gdpr_utils.py](gdpr_utils.py).*

2. **Grouping by Event**  
   Sorted images are grouped into events if the time difference between consecutive images is less than a defined threshold (for example, one hour).  
   *Refer to [file_utils.py](file_utils.py).*

3. **Best Photo Selection**  
   Within each event group, the system selects a single best representative image based on a composite score. This score is computed by combining SSIM-based pixel change measurements and edge detection confidence. Only images with a composite score above a defined threshold (e.g., >5.0) are considered for further analysis.  
   *See [event_detector.py](event_detector.py).*

4. **Motion Detection & Event Classification**  
   The best representative images are then passed to a second analysis stage. Here, a machine learning classifier—built using a One-Class SVM trained on pre-labeled animal images—confirms the presence of animals in each event.  
   *See [classifier.py](classifier.py) and [event_detector.py](event_detector.py).*

5. **Dynamic Thresholding (Optional)**  
   Optionally, CritterSnap can compute dynamic thresholds using a sample of animal training data. This step adjusts the pixel change and edge confidence thresholds automatically to fine-tune detection performance.  
   *Refer to [dynamic_thresholds.py](dynamic_thresholds.py) if available.*

6. **Parallel Processing**  
   To efficiently process large datasets, the pipeline leverages Python’s `concurrent.futures` and joblib’s `Parallel` for both image loading and event detection tasks.  
   *See [file_utils.py](file_utils.py), [classifier.py](classifier.py), and [main.py](main.py).*

7. **Interactive Configuration & Logging**  
   An interactive configuration menu—managed by the ConfigManager—allows users to update parameters such as file paths, thresholds (including edge and white pixel thresholds), training data directories, and file range settings at runtime. All processing details and results are logged for later review.  
   *See [config_manager.py](config_manager.py) and [config.py](config.py).*

8. **Structured Data Management**  
   The `ImageObject` class encapsulates each image’s data along with its timestamp and file path, simplifying downstream processing and logging.

---

## Features

- **Automated Image Sorting:**  
  Automatically organizes images by timestamp. If EXIF data is missing, it falls back to file modification times.  
  *See [file_utils.py](file_utils.py).*

- **GDPR Compliance Check:**  
  Filters out images with an excessive number of pure white pixels to ensure GDPR protection.  
  *See [gdpr_utils.py](gdpr_utils.py).*

- **Edge Detection & SSIM Analysis:**  
  Uses Sobel edge detection and the Structural Similarity Index Measure (SSIM) to quantify pixel changes and extract edge features from images.  
  *See [event_detector.py](event_detector.py) and related modules.*

- **Best Photo Selection:**  
  Selects a single best representative image from each event group based on a composite score that balances pixel changes with edge confidence.

- **Dynamic Thresholding (Optional):**  
  Automatically computes adaptive thresholds using animal training data to optimize event detection parameters.

- **Machine Learning Classification:**  
  Applies a One-Class SVM classifier trained on pre-labeled animal images to confirm detected events.  
  *See [classifier.py](classifier.py).*

- **Parallel Processing:**  
  Implements multi-threading and multi-processing to efficiently handle large image datasets during sorting, detection, and classification.  
  *See [main.py](main.py) and [file_utils.py](file_utils.py).*

- **Interactive Configuration:**  
  An easy-to-use interactive menu (via ConfigManager) lets users adjust a variety of parameters—including directory paths, change thresholds, edge thresholds, window size, and training data paths—on the fly.  
  *See [config_manager.py](config_manager.py) and [config.py](config.py).*

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
