import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

###############################################################################
# 1. Utility Functions
###############################################################################

def get_all_file_paths(root_dir):
    """
    Return a list of all file paths in `root_dir` (recursive).
    """
    file_paths = []
    for root, dirs, files in os.walk(root_dir):
        for fname in files:
            file_paths.append(os.path.join(root, fname))
    return file_paths

def _process_image(file_path, image_size=(64, 64)):
    """
    Helper function to load and preprocess a single image.
    Returns a flattened numpy array, or None if loading fails.
    """
    try:
        img = Image.open(file_path).convert("L").resize(image_size)
        return np.array(img).flatten()
    except Exception:
        return None

def load_images_parallel(root_dir, label=1, image_size=(64, 64), 
                         max_workers=8, print_every=50):
    """
    Load all images from `root_dir` (and subdirectories) in parallel.
    Converts them to grayscale, resizes, flattens, and assigns the same label.
    """
    file_paths = get_all_file_paths(root_dir)
    total_files = len(file_paths)
    if total_files == 0:
        return np.array([]), np.array([])

    data = []
    # Parallel loading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_image, fp, image_size): fp for fp in file_paths}
        
        for i, future in enumerate(as_completed(futures)):
            arr = future.result()
            if arr is not None:
                data.append(arr)
            
            # Print progress less frequently
            if (i + 1) % print_every == 0:
                percent_done = (i + 1) / total_files * 100
                print(f"Loading images... {percent_done:.1f}% complete", end="\r")
    
    print("")  # Move to the next line after loop
    labels = np.full(len(data), label, dtype=int)
    return np.array(data), labels

###############################################################################
# 2. Training the Animal Classifier
###############################################################################

def evaluate_model(params, X_train, X_val):
    nu_val, gamma_val = params
    model = OneClassSVM(kernel="rbf", nu=nu_val, gamma=gamma_val)
    model.fit(X_train)
    val_preds = model.predict(X_val)
    # Compute outlier rate: proportion of predictions labeled as outliers (-1)
    outlier_rate = np.sum(val_preds == -1) / len(X_val)
    return outlier_rate, model, params

def train_animal_classifier(animal_path):
    # Load the data using your parallel loader
    X, _ = load_images_parallel(animal_path, label=1, image_size=(64, 64))
    if len(X) == 0:
        raise ValueError(f"No images found in '{animal_path}'.")
    
    # Create train/validation split
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    
    # Define hyperparameter grid
    param_grid = {
        "nu":    [0.001, 0.01, 0.1],
        "gamma": [0.005, 1e-3, 1e-4]
    }
    # Create list of parameter combinations
    param_combinations = [(nu, gamma) for nu in param_grid["nu"] for gamma in param_grid["gamma"]]
    
    print("Training models with different hyperparameters in parallel...")
    # Evaluate each parameter combination in parallel
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_model)(params, X_train, X_val) for params in param_combinations
    )
    
    # Select the best model (lowest outlier rate)
    best_outlier_rate, best_model, best_params = min(results, key=lambda x: x[0])
    print(f"Best hyperparameters: nu={best_params[0]}, gamma={best_params[1]} with outlier rate: {best_outlier_rate:.3f}")
    return best_model

###############################################################################
# 3. Prediction
###############################################################################

def predict_image(file_path, model, image_size=(64, 64)):
    """
    Predict if a single image is 'Animal' or 'Non-Animal' using the trained model.
    """
    img = Image.open(file_path).convert("L").resize(image_size)
    arr = np.array(img).flatten().reshape(1, -1)
    prediction = model.predict(arr)[0]
    return "Animal" if prediction == 1 else "Non-Animal"
