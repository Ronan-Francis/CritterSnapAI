import json
import os
from random import sample
from statistics import mean
from PIL import Image
from edge_detection import compute_edge_confidence
from image_utils import measure_changes

# Default configuration parameters
DEFAULT_CONFIG = {
    "directory_path": r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\example\ds_researchATU",
    "output_directory": r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\output",
    "output_log_path": r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\output\output_log.txt",
    "change_threshold": 0.1,           # This value may be updated dynamically.
    "white_pixel_threshold": 50000,
    "edge_threshold": 50,
    "edge_confidence_threshold": 0.15,  # This value may be updated dynamically.
    "window_size": 20,
    "base_url": "https://lilawildlife.blob.core.windows.net/lila-wildlife/wcs-unzipped/animals/0011/",
    "start_file": 1,
    "end_file": 700,
    "animal_training_path": r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\example\as_conservationistFrankfurt\IE_Forest_County_Wicklow_21_loc_01-20241031T145429Z-001",
    "non_animal_training_path": r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\example\eccv_18_all_images_sm"
}


def load_config_from_file(file_path):
    """
    Loads the configuration from a JSON file.
    
    Parameters:
      - file_path: The path to the JSON configuration file.
      
    Returns:
      - A configuration dictionary loaded from the file.
    """
    with open(file_path, 'r') as f:
        config = json.load(f)
    print(f"Configuration loaded from {file_path}.")
    return config


def save_config_to_file(config, file_path):
    """
    Saves the provided configuration dictionary to a JSON file.

    Parameters:
      - config: The configuration dictionary.
      - file_path: The file path where the configuration will be saved.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {file_path}.")
    except Exception as e:
        print(f"Failed to save configuration: {e}")


def config_menu(initial_config=None):
    """
    Display an interactive menu to update configuration settings.
    Returns a configuration dictionary with the updated parameters.
    
    Parameters:
      - initial_config: A configuration dictionary to pre-load the menu with.
                        If None, DEFAULT_CONFIG is used.
    """
    config = initial_config.copy() if initial_config else DEFAULT_CONFIG.copy()
    
    menu_text = """
Configuration Menu:
1. Directory Path             : {directory_path}
2. Output Directory           : {output_directory}
3. Output Log Path            : {output_log_path}
4. Change Threshold           : {change_threshold}
5. White Pixel Threshold      : {white_pixel_threshold}
6. Edge Threshold             : {edge_threshold}
7. Edge Confidence Threshold  : {edge_confidence_threshold}
8. Window Size                : {window_size}
9. Base URL                   : {base_url}
10. Start File Number         : {start_file}
11. End File Number           : {end_file}
12. Animal Training Path      : {animal_training_path}
13. Non-Animal Training Path  : {non_animal_training_path}
0. Save and Exit
    """
    
    while True:
        # Print the menu with current configuration values
        print(menu_text.format(**config))
        choice = input("Enter the number of the parameter to change (or 0 to exit): ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            config["directory_path"] = input("Enter new directory path: ").strip()
        elif choice == "2":
            config["output_directory"] = input("Enter new output directory: ").strip()
        elif choice == "3":
            config["output_log_path"] = input("Enter new output log path: ").strip()
        elif choice == "4":
            try:
                config["change_threshold"] = float(input("Enter new change threshold (e.g., 0.1): ").strip())
            except ValueError:
                print("Invalid input. Please enter a number.")
        elif choice == "5":
            try:
                config["white_pixel_threshold"] = int(input("Enter new white pixel threshold: ").strip())
            except ValueError:
                print("Invalid input. Please enter an integer.")
        elif choice == "6":
            try:
                config["edge_threshold"] = int(input("Enter new edge threshold: ").strip())
            except ValueError:
                print("Invalid input. Please enter an integer.")
        elif choice == "7":
            try:
                config["edge_confidence_threshold"] = float(input("Enter new edge confidence threshold: ").strip())
            except ValueError:
                print("Invalid input. Please enter a number.")
        elif choice == "8":
            try:
                config["window_size"] = int(input("Enter new window size: ").strip())
            except ValueError:
                print("Invalid input. Please enter an integer.")
        elif choice == "9":
            config["base_url"] = input("Enter new base URL: ").strip()
        elif choice == "10":
            try:
                config["start_file"] = int(input("Enter new start file number: ").strip())
            except ValueError:
                print("Invalid input. Please enter an integer.")
        elif choice == "11":
            try:
                config["end_file"] = int(input("Enter new end file number: ").strip())
            except ValueError:
                print("Invalid input. Please enter an integer.")
        elif choice == "12":
            config["animal_training_path"] = input("Enter new animal training path: ").strip()
        elif choice == "13":
            config["non_animal_training_path"] = input("Enter new non-animal training path: ").strip()
        else:
            print("Invalid choice. Please try again.")
    
    print("Configuration updated.")
    return config


def compute_dynamic_thresholds(config: dict, sample_fraction: float = 0.1) -> dict:
    """
    Computes dynamic thresholds using a small random sample of the animal training data.
    
    It calculates average edge confidence and change measures (using triplets of images)
    from a subset of images, then updates the 'edge_confidence_threshold' and 'change_threshold'
    in the configuration.

    Parameters:
        config (dict): The current configuration dictionary.
        sample_fraction (float): The fraction of images to sample from the training set.

    Returns:
        dict: The updated configuration dictionary with new threshold values.
    """
    training_path = config.get("animal_training_path")
    if not training_path or not os.path.isdir(training_path):
        print("Invalid animal training path for dynamic thresholding.")
        return config

    # Gather all image file paths from the training folder.
    files = [os.path.join(training_path, f)
             for f in os.listdir(training_path)
             if os.path.isfile(os.path.join(training_path, f))]
    if not files:
        print("No training images found.")
        return config

    # Determine sample size (at least 3 images required for triplet-based measures).
    sample_size = max(3, int(len(files) * sample_fraction))
    sample_size = min(sample_size, len(files))
    sample_files = sample(files, sample_size)
    sample_files.sort()  # Sorting by file name; adjust if needed for chronological order.

    # Compute edge confidences for each sample image.
    edge_confidences = []
    for file in sample_files:
        try:
            with Image.open(file) as img:
                edge_conf, _ = compute_edge_confidence(
                    img,
                    edge_threshold=config.get("edge_threshold", 50),
                    window_size=config.get("window_size", 20)
                )
                edge_confidences.append(edge_conf)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if edge_confidences:
        new_edge_conf_threshold = mean(edge_confidences)
    else:
        new_edge_conf_threshold = config.get("edge_confidence_threshold", 0.15)

    # Compute change measures using triplets of images.
    change_measures = []
    if len(sample_files) >= 3:
        images = []
        for file in sample_files:
            try:
                images.append(Image.open(file))
            except Exception as e:
                print(f"Error loading {file}: {e}")
        # Compute measure_changes for each consecutive triplet.
        for i in range(1, len(images) - 1):
            change = measure_changes(images[i - 1], images[i], images[i + 1])
            change_measures.append(change)
        if change_measures:
            new_change_threshold = mean(change_measures)
        else:
            new_change_threshold = config.get("change_threshold", 0.1)
    else:
        new_change_threshold = config.get("change_threshold", 0.1)

    print(f"Dynamic Thresholding: new change_threshold = {new_change_threshold:.3f}, "
          f"new edge_confidence_threshold = {new_edge_conf_threshold:.3f}")
    config["change_threshold"] = new_change_threshold
    config["edge_confidence_threshold"] = new_edge_conf_threshold
    return config


def run():
    """
    Run the configuration setup. This function allows you to either use the default
    configuration, load an existing configuration, or update it interactively.
    You also have the option to apply dynamic thresholding using a small sample of
    the animal training data.

    Returns:
      - The configuration dictionary to be used.
    """
    # Attempt to load existing configuration from file if available.
    try:
        config_data = load_config_from_file("config.json")
    except Exception:
        print("No existing configuration file found. Using default configuration.")
        config_data = DEFAULT_CONFIG.copy()

    update_choice = input("Do you want to update the configuration? (y/n, default n): ").strip().lower()
    if update_choice == "y":
        config_data = config_menu(config_data)
        save_config_to_file(config_data, "config.json")
    else:
        print("Using the current configuration as is.")

    dynamic_choice = input("Apply dynamic thresholding using a sample of animal training data? (y/n, default n): ").strip().lower()
    if dynamic_choice == "y":
        config_data = compute_dynamic_thresholds(config_data)

    print("\nFinal configuration:")
    for key, value in config_data.items():
        print(f"{key}: {value}")

    return config_data