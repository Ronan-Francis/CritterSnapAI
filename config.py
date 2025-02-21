import json

# Default configuration parameters
DEFAULT_CONFIG = {
    "directory_path": r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\example\ds_researchATU",
    "output_directory": r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\output",
    "output_log_path": r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\output\output_log.txt",
    "change_threshold": 0.1,           # 0.1 to 0.6: low, 0.61 to 0.80: medium, 0.81 to 1.00: high
    "white_pixel_threshold": 50000,
    "edge_threshold": 50,
    "edge_confidence_threshold": 0.15,
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

def run():
    """
    Run the configuration setup. This function allows you to either use the default
    configuration or update it interactively.
    
    If you choose not to update, the default (or file-loaded) configuration is used.
    Otherwise, you can edit the configuration and have it saved to a file.
    
    Returns:
      - The configuration dictionary to be used.
    """
    # Ask the user whether to use the current configuration as-is or update it.
    update_choice = input("Do you want to update the configuration? (y/n, default n): ").strip().lower()
    
    # Attempt to load existing config from file if available
    try:
        config = load_config_from_file("config.json")
    except Exception:
        print("No existing configuration file found. Using default configuration.")
        config = DEFAULT_CONFIG.copy()
    
    if update_choice == "y":
        config = config_menu(config)
        save_config_to_file(config, "config.json")
    else:
        print("Using the current configuration as is.")
    
    print("\nFinal configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    return config