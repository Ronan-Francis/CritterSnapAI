import os
from random import sample
from statistics import mean
from PIL import Image
from image_processor import ImageProcessor

def compute_dynamic_thresholds(config: dict, sample_fraction: float = 0.1) -> dict:
    """
    Computes dynamic thresholds using a small random sample of the animal training data.
    
    It calculates average edge confidence and change measures (using triplets of images)
    from a subset of images, then updates the 'edge_confidence_threshold' and 'change_threshold'
    in the configuration.
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
    sample_files.sort()  # Sorting by file name.

    # Compute edge confidences for each sample image.
    edge_confidences = []
    for file in sample_files:
        try:
            with Image.open(file) as img:
                edge_conf, _ = ImageProcessor.compute_edge_confidence(
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
        for i in range(1, len(images) - 1):
            change = ImageProcessor.measure_changes(images[i-1], images[i], images[i+1])
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
