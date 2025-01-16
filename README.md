# CritterSnapAI

CritterSnapAI is a project designed to process and classify images captured by wildlife cameras. It identifies frames with animal activity and marks them for further analysis.

## Project Structure

- **app_config.py**: Stores configuration parameters (paths, thresholds).
- **data_structures.py**: Contains lightweight classes (e.g., `ImageObject`) for holding image data.
- **gdpr_utils.py**: Handles GDPR filtering (white-pixel thresholds, etc.).
- **image_utils.py**: Provides image-processing functions (downsampling, measuring changes).
- **classification.py**: Implements decision-tree logic and additional classification steps.
- **sorting_utils.py**: Sorts and groups images by date/time, organizes them into events.
- **main.py**: Orchestrates the end-to-end workflow (sorting, classification, output logging).

## Usage

1. Place your images in the `data/example` directory.
2. Adjust the `directory_path` and `output_directory` variables in [`main.py`](CritterSnapAI/main.py) to match your setup.
3. Run the main script:
    ```sh
    python CritterSnapAI/main.py
    ```
    ## Code Overview

    The code is organized into distinct modules, each with a specific responsibility to streamline the processing and classification of wildlife images:

    - **[app_config.py](CritterSnapAI/app_config.py)**: Manages configuration settings, including paths and processing thresholds.
    - **[data_structures.py](CritterSnapAI/data_structures.py)**: Defines simple classes to encapsulate image data and related attributes.
    - **[gdpr_utils.py](CritterSnapAI/gdpr_utils.py)**: Ensures compliance with GDPR by applying necessary filters to the images.
    - **[image_utils.py](CritterSnapAI/image_utils.py)**: Contains functions for image manipulation, such as downsampling and detecting changes.
    - **[classification.py](CritterSnapAI/classification.py)**: Implements the logic for classifying images using decision trees and other methods.
    - **[sorting_utils.py](CritterSnapAI/sorting_utils.py)**: Organizes images chronologically and groups them into events based on timestamps.
    - **[main.py](CritterSnapAI/main.py)**: Coordinates the entire workflow, from sorting and classification to logging the results.

    This modular approach enhances the maintainability and scalability of the project.

