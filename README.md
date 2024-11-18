# CritterSnapAI

CritterSnapAI is a project designed to process and classify images captured by wildlife cameras. It identifies frames with animal activity and marks them for further analysis.

## Project Structure



## Usage

1. Place your images in the `data/example` directory.
2. Adjust the `directory_path` and `output_directory` variables in [`main.py`](CritterSnapAI/main.py) to match your setup.
3. Run the main script:
    ```sh
    python CritterSnapAI/main.py
    ```

## Code Overview

- [`load_images.py`](CritterSnapAI/load_images.py): Contains the function [`load_images_from_directory`](CritterSnapAI/load_images.py) to load images from a specified directory.
- [`decision_tree.py`](CritterSnapAI/decision_tree.py): Implements the [`decision_tree`](CritterSnapAI/decision_tree.py) function to classify images based on pixel changes.
- [`image_processing.py`](CritterSnapAI/image_processing.py): Provides image processing utilities like [`convert_to_grayscale`](CritterSnapAI/image_processing.py) and [`measure_changes`](CritterSnapAI/image_processing.py).
- [`event_processing.py`](CritterSnapAI/event_processing.py): Contains the function [`copy_events_with_red_box`](CritterSnapAI/event_processing.py) to mark and save event images.
- [`main.py`](CritterSnapAI/main.py): The main script to run the entire image processing and classification pipeline.
