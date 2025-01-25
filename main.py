# main.py
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import (
    directory_path,
    output_directory,
    change_threshold,
    white_pixel_threshold,
    output_log_path,
    animal_training_path,      # New: path for training animal images
    non_animal_training_path   # New: path for training non-animal images
)
from sorting_utils import sort_images_by_date_time, group_images_by_event
from classification import process_group

# Import the classifier functions
from sklearn_classifier import train_animal_classifier, predict_image

def main():
    start_time = time.time()
    print("Starting the image sorting process...")

    # 1. Sort images by date/time (GDPR filter included)
    images_with_dates = sort_images_by_date_time(directory_path, white_pixel_threshold)

    # 2. Group images by event
    grouped_events = group_images_by_event(images_with_dates)

    all_events = []
    all_non_events = []

    # 3. Parallel classification of each group
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), len(grouped_events))) as executor:
        futures = {
            executor.submit(process_group, group, change_threshold): group
            for group in grouped_events
        }
        for i, future in enumerate(as_completed(futures)):
            events, non_events = future.result()
            all_events.extend(events)
            all_non_events.extend(non_events)

            percentage = ((i + 1) / len(grouped_events)) * 100
            print(f"Processed: {i + 1}/{len(grouped_events)} ({percentage:.2f}% complete)", end="\r")


    # 5. Log results
    with open(output_log_path, 'w') as log_file:
        log_file.write(f"Total Events Detected: {len(all_events)}\n")
        log_file.write(f"Total Non-Events Detected: {len(all_non_events)}\n\nEvents:\n")
        log_file.writelines(f"{event.get_file_path()}\n" for event in all_events)

    # 6. Train the animal classifier using labeled data
    print("\nTraining animal classifier...")
    model = train_animal_classifier(animal_training_path)
    event_animal_total = 0
    event_non_animal_total = 0
    nEvent_animal_total = 0
    nEvent_non_animal_total = 0

    # 7. Use the trained model to predict on each event
    print("\nPredicting event content (Animal vs. Non-Animal):")
    for event in all_events:
        image_path = event.get_file_path()
        prediction = predict_image(image_path, model)
        relative_image_path = os.path.relpath(image_path, directory_path)
        print(f"{relative_image_path}: {prediction}")
        
        if "Animal" in prediction:
            event_animal_total += 1
        if "Non-Animal" in prediction:
            event_non_animal_total += 1

    print("\nPredicting non_event content (Animal vs. Non-Animal):")
    for non_event in all_non_events:
        image_path = non_event.get_file_path()
        prediction = predict_image(image_path, model)
        relative_image_path = os.path.relpath(image_path, directory_path)
        print(f"{relative_image_path}: {prediction}")
        
        if "Animal" in prediction:
            nEvent_animal_total += 1
        if "Non-Animal" in prediction:
            nEvent_non_animal_total += 1


    # Calculate event percentages
    total_events = len(all_events)
    total_non_events = len(all_non_events)
    event_animal_percentage = (event_animal_total / total_events) * 100 
    event_nAnimal_percentage = (event_non_animal_total / total_events) * 100 
    nEvent_animal_percentage = (nEvent_animal_total / total_non_events) * 100 
    nEvent_nAnimal_percentage = (nEvent_non_animal_total / total_non_events) * 100 

    end_time = time.time()
    elapsed_time = end_time - start_time

    # 4. Output metrics
    print(f"\nEvent sorting and classification complete.")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Total Events Detected: {len(all_events)}")
    print(f"Total Non-Events Detected: {len(all_non_events)}")
    print(f"Percentage of Events that are Animals: {event_animal_percentage:.0f}%")
    print(f"Percentage of Events that are Non-Animals: {event_nAnimal_percentage:.0f}%")
    print(f"Percentage of Non-Events that are Animals: {nEvent_animal_percentage:.0f}%")
    print(f"Percentage of Non-Events that are Non-Animals: {nEvent_nAnimal_percentage:.0f}%")


if __name__ == "__main__":
    main()
