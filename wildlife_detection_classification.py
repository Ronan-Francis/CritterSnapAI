import numpy as np
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification

def wildlife_detection_classification(img):
    # Detection
    detection_model = pw_detection.MegaDetectorV6()  # Model weights are automatically downloaded.
    detection_result = detection_model.single_image_detection(img)
    
    # Classification
    classification_model = pw_classification.AI4GAmazonRainforest()  # Model weights are automatically downloaded.
    classification_results = classification_model.single_image_classification(img)
    
    return detection_result, classification_results

