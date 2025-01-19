import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def load_images(image_dir, label, image_size=(64, 64)):
    data = []
    labels = []
    for fname in os.listdir(image_dir):
        fpath = os.path.join(image_dir, fname)
        try:
            img = Image.open(fpath).convert('L').resize(image_size)
            data.append(np.array(img).flatten())
            labels.append(label)
        except:
            pass
    return np.array(data), np.array(labels)

def train_animal_classifier(animal_path, non_animal_path):
    # Load labeled data
    animal_data, animal_labels = load_images(animal_path, 1)
    non_animal_data, non_animal_labels = load_images(non_animal_path, 0)
    # Combine
    X = np.vstack((animal_data, non_animal_data))
    y = np.hstack((animal_labels, non_animal_labels))
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    # Evaluate
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return clf

def predict_image(file_path, model, image_size=(64, 64)):
    img = Image.open(file_path).convert('L')
    img = img.resize(image_size)
    arr = np.array(img).flatten().reshape(1,-1)
    return "Animal" if model.predict(arr)[0] == 1 else "Non-Animal"


