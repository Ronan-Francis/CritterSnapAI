import os
from PIL import Image
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor

def get_all_file_paths(root_dir: str):
    return [os.path.join(root, fname)
            for root, _, files in os.walk(root_dir)
            for fname in files]

def _process_image(file_path: str, image_size=(64, 64)):
    try:
        with Image.open(file_path) as img:
            img = img.convert("L").resize(image_size)
            return np.array(img).flatten()
    except Exception:
        return None

def load_images_parallel(root_dir: str, image_size=(64, 64), max_workers=8, print_every=50):
    file_paths = get_all_file_paths(root_dir)
    total_files = len(file_paths)
    if total_files == 0:
        return np.array([])
    data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_image, fp, image_size): fp for fp in file_paths}
        for i, future in enumerate(futures):
            result = future.result()
            if result is not None:
                data.append(result)
            if (i + 1) % print_every == 0:
                percent_done = (i + 1) / total_files * 100
                print(f"Loading images... {percent_done:.1f}% complete", end="\r")
    print("")
    return np.array(data)

def evaluate_model(params, X_train, X_val):
    nu_val, gamma_val = params
    model = OneClassSVM(kernel="rbf", nu=nu_val, gamma=gamma_val)
    model.fit(X_train)
    val_preds = model.predict(X_val)
    outlier_rate = np.sum(val_preds == -1) / len(X_val)
    return outlier_rate, model, params

def train_animal_classifier(animal_path: str) -> OneClassSVM:
    X = load_images_parallel(animal_path, image_size=(64, 64))
    if X.size == 0:
        raise ValueError(f"No images found in '{animal_path}'.")
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    param_grid = {
        "nu": [0.001, 0.01, 0.1],
        "gamma": [0.005, 1e-3, 1e-4]
    }
    param_combinations = [(nu, gamma) for nu in param_grid["nu"] for gamma in param_grid["gamma"]]
    
    print("Training models with different hyperparameters in parallel...")
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_model)(params, X_train, X_val) for params in param_combinations
    )
    best_outlier_rate, best_model, best_params = min(results, key=lambda x: x[0])
    print(f"Best hyperparameters: nu={best_params[0]}, gamma={best_params[1]} with outlier rate: {best_outlier_rate:.3f}")
    return best_model

def predict_image(file_path: str, model: OneClassSVM, image_size=(64, 64)) -> str:
    with Image.open(file_path) as img:
        img = img.convert("L").resize(image_size)
        arr = np.array(img).flatten().reshape(1, -1)
    prediction = model.predict(arr)[0]
    return "Animal" if prediction == 1 else "Non-Animal"
