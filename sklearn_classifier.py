import os
import numpy as np
from PIL import Image

from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

def load_images(image_dir, label, image_size=(64, 64)):
    """
    Loads images from a directory, converts them to grayscale, resizes, and flattens.
    Returns (X, y) where X is a NumPy array of shape (N, width*height), and y is
    a 1D array of the same label (for compatibility, though only 'animal' images here).

    :param image_dir: Directory with images (all are considered the same 'label').
    :param label: Integer label (e.g., 1 for animal). Not used for one-class training,
                  but included for consistent function signature.
    :param image_size: Tuple (width, height) for resizing.
    """
    data = []
    labels = []
    for fname in os.listdir(image_dir):
        fpath = os.path.join(image_dir, fname)
        try:
            img = Image.open(fpath).convert("L").resize(image_size)
            data.append(np.array(img).flatten())
            labels.append(label)
        except Exception as e:
            # Skip files that cannot be opened/processed
            pass
    return np.array(data), np.array(labels)

def train_animal_classifier(animal_path):
    """
    Trains a One-Class SVM on the provided animal images. This is a one-class scenario
    because no negative data is available. We perform:
      1) Loading and splitting the animal data into train/validation sets
      2) A simple hyperparameter grid search for OneClassSVM
      3) Return the best-performing model (i.e., minimal outlier rate on validation)

    :param animal_path: Directory containing images of animals (our "normal" data).
    :return: A trained OneClassSVM model.
    """
    # -- Load animal data --
    X, _ = load_images(animal_path, label=1, image_size=(64, 64))
    if len(X) == 0:
        raise ValueError(f"No images found in '{animal_path}'. Cannot train the model.")

    # -- Split: train (80%) and validation (20%) --
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    # -- Define a small grid of hyperparameters to try --
    param_grid = {
        "nu":    [0.001, 0.01, 0.1],    # controls fraction of outliers
        "gamma": ["scale", 1e-3, 1e-4], # kernel coefficient for rbf
    }

    best_model = None
    best_outlier_rate = float("inf")
    best_params = None

    # -- Grid search manually over OneClassSVM parameters --
    for nu_val in param_grid["nu"]:
        for gamma_val in param_grid["gamma"]:
            model = OneClassSVM(kernel="rbf", nu=nu_val, gamma=gamma_val)
            model.fit(X_train)

            # Predict on validation set: +1 (inlier), -1 (outlier)
            val_preds = model.predict(X_val)
            # Count how many of our own animal images were flagged as outliers
            outlier_count = np.sum(val_preds == -1)
            outlier_rate = outlier_count / len(X_val)

            if outlier_rate < best_outlier_rate:
                best_outlier_rate = outlier_rate
                best_model = model
                best_params = (nu_val, gamma_val)

    print(f"Best One-Class SVM params: nu={best_params[0]}, gamma={best_params[1]}")
    print(f"Validation outlier rate on animal data: {best_outlier_rate:.2%}")

    # best_model is the OneClassSVM with minimal outlier rate on validation
    return best_model

def predict_image(file_path, model, image_size=(64, 64)):
    """
    Predict if an image is "Animal" or "Non-Animal" based on a trained OneClassSVM.

    :param file_path: Path to the image file
    :param model: Trained OneClassSVM
    :param image_size: Tuple (width, height) to resize the input image
    :return: "Animal" if the model predicts +1, else "Non-Animal"
    """
    img = Image.open(file_path).convert("L").resize(image_size)
    arr = np.array(img).flatten().reshape(1, -1)

    prediction = model.predict(arr)[0]  # +1 or -1
    return "Animal" if prediction == 1 else "Non-Animal"

