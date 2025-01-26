import os
import numpy as np
from PIL import Image
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

def count_files_in_directory(root_dir):
    total_count = 0
    for entry in os.scandir(root_dir):
        if entry.is_file():
            total_count += 1
        elif entry.is_dir():
            total_count += count_files_in_directory(entry.path)
    return total_count

def load_images_recursive(root_dir, label, image_size=(64, 64), total_files=None, processed=None):
    data, labels = [], []
    if total_files is None:
        total_files = count_files_in_directory(root_dir)
        processed = [0]  # use list to mutate inside function
    for entry in os.scandir(root_dir):
        if entry.is_file():
            try:
                img = Image.open(entry.path).convert("L").resize(image_size)
                data.append(np.array(img).flatten())
                labels.append(label)
            except:
                pass
            processed[0] += 1
            percent_done = (processed[0] / total_files) * 100
            print(f"Loading images for training... {percent_done:.1f}% complete", end="\r")
        elif entry.is_dir():
            sub_data, sub_labels = load_images_recursive(entry.path, label, image_size, total_files, processed)
            data.extend(sub_data)
            labels.extend(sub_labels)
    return np.array(data), np.array(labels)

def train_animal_classifier(animal_path):
    X, _ = load_images_recursive(animal_path, label=1, image_size=(64, 64))
    if len(X) == 0:
        raise ValueError(f"No images found in '{animal_path}'.")
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    param_grid = {"nu": [0.001, 0.01, 0.1], "gamma": ["scale", 1e-3, 1e-4]}
    total_combinations = len(param_grid["nu"]) * len(param_grid["gamma"])
    combination_count = 0

    best_model, best_outlier_rate = None, float("inf")
    for nu_val in param_grid["nu"]:
        for gamma_val in param_grid["gamma"]:
            combination_count += 1
            progress = (combination_count / total_combinations) * 100
            print(f"Training... {progress:.1f}% of hyperparameter search completed", end="\r")

            model = OneClassSVM(kernel="rbf", nu=nu_val, gamma=gamma_val)
            model.fit(X_train)
            val_preds = model.predict(X_val)
            outlier_count = np.sum(val_preds == -1)
            outlier_rate = outlier_count / len(X_val)

            if outlier_rate < best_outlier_rate:
                best_outlier_rate = outlier_rate
                best_model = model
    return best_model

def predict_image(file_path, model, image_size=(64, 64)):
    img = Image.open(file_path).convert("L").resize(image_size)
    arr = np.array(img).flatten().reshape(1, -1)
    prediction = model.predict(arr)[0]
    return "Animal" if prediction == 1 else "Non-Animal"
