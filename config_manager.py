import json
import os

DEFAULT_CONFIG = {
    "directory_path": r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\example\as_conservationistFrankfurt\IE_Forest_County_Wicklow_21_loc_01-20241031T145429Z-001",
    "output_directory": r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\output",
    "output_log_path": r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\output\output_log.txt",
    "change_threshold": 0.1,
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

class ConfigManager:
    def __init__(self, file_path="config.json"):
        self.file_path = file_path
        self.config = self.load_config()

    def load_config(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    config = json.load(f)
                print(f"Configuration loaded from {self.file_path}.")
                return config
            except Exception as e:
                print(f"Error loading configuration: {e}")
        print("No existing configuration file found. Using default configuration.")
        return DEFAULT_CONFIG.copy()

    def save_config(self):
        try:
            with open(self.file_path, "w") as f:
                json.dump(self.config, f, indent=4)
            print(f"Configuration saved to {self.file_path}.")
        except Exception as e:
            print(f"Failed to save configuration: {e}")

    def interactive_update(self):
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
            print(menu_text.format(**self.config))
            choice = input("Enter the number of the parameter to change (or 0 to exit): ").strip()
            if choice == "0":
                break
            elif choice == "1":
                self.config["directory_path"] = input("Enter new directory path: ").strip()
            elif choice == "2":
                self.config["output_directory"] = input("Enter new output directory: ").strip()
            elif choice == "3":
                self.config["output_log_path"] = input("Enter new output log path: ").strip()
            elif choice == "4":
                try:
                    self.config["change_threshold"] = float(input("Enter new change threshold (e.g., 0.1): ").strip())
                except ValueError:
                    print("Invalid input. Please enter a number.")
            elif choice == "5":
                try:
                    self.config["white_pixel_threshold"] = int(input("Enter new white pixel threshold: ").strip())
                except ValueError:
                    print("Invalid input. Please enter an integer.")
            elif choice == "6":
                try:
                    self.config["edge_threshold"] = int(input("Enter new edge threshold: ").strip())
                except ValueError:
                    print("Invalid input. Please enter an integer.")
            elif choice == "7":
                try:
                    self.config["edge_confidence_threshold"] = float(input("Enter new edge confidence threshold: ").strip())
                except ValueError:
                    print("Invalid input. Please enter a number.")
            elif choice == "8":
                try:
                    self.config["window_size"] = int(input("Enter new window size: ").strip())
                except ValueError:
                    print("Invalid input. Please enter an integer.")
            elif choice == "9":
                self.config["base_url"] = input("Enter new base URL: ").strip()
            elif choice == "10":
                try:
                    self.config["start_file"] = int(input("Enter new start file number: ").strip())
                except ValueError:
                    print("Invalid input. Please enter an integer.")
            elif choice == "11":
                try:
                    self.config["end_file"] = int(input("Enter new end file number: ").strip())
                except ValueError:
                    print("Invalid input. Please enter an integer.")
            elif choice == "12":
                self.config["animal_training_path"] = input("Enter new animal training path: ").strip()
            elif choice == "13":
                self.config["non_animal_training_path"] = input("Enter new non-animal training path: ").strip()
            else:
                print("Invalid choice. Please try again.")
        print("Configuration updated.")
        self.save_config()
