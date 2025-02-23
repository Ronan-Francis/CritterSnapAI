import numpy as np
from PIL import Image
from typing import Tuple

def rgb2gray(image_array: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image (as a NumPy array) to grayscale.
    
    Parameters:
        image_array: A NumPy array representing an RGB image.
        
    Returns:
        A 2D NumPy array representing the grayscale image.
    """
    return np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])

def sobel_gradients(gray_image: np.ndarray) -> np.ndarray:
    """
    Compute the gradient magnitude using the Sobel operator.
    
    Parameters:
        gray_image: A 2D NumPy array representing the grayscale image.
        
    Returns:
        A 2D NumPy array of the gradient magnitude.
    """
    # Define Sobel kernels.
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)
    
    H, W = gray_image.shape
    padded = np.pad(gray_image, ((1, 1), (1, 1)), mode='edge')
    Gx = np.empty((H, W), dtype=np.float32)
    Gy = np.empty((H, W), dtype=np.float32)
    
    for i in range(H):
        for j in range(W):
            patch = padded[i:i+3, j:j+3]
            Gx[i, j] = np.sum(Kx * patch)
            Gy[i, j] = np.sum(Ky * patch)
    
    grad_magnitude = np.sqrt(Gx**2 + Gy**2)
    return grad_magnitude

def compute_edge_confidence(pil_image: Image.Image, edge_threshold: int = 50, window_size: int = 20) -> Tuple[float, np.ndarray]:
    """
    Compute an edge-based confidence score indicating the likelihood of an animal presence.
    
    Parameters:
        pil_image: A PIL Image object.
        edge_threshold: Threshold on gradient magnitude to detect edges.
        window_size: Size of the sliding window to capture local edge blobs.
    
    Returns:
        A tuple containing:
          - confidence: A float value between 0 and 1.
          - edge_map: A binary NumPy array representing the detected edges.
    """
    # Convert image to grayscale array.
    gray_array = np.array(pil_image.convert('L'), dtype=np.float32)
    grad_mag = sobel_gradients(gray_array)
    
    # Create binary edge map.
    edge_map = (grad_mag > edge_threshold).astype(np.uint8)
    edge_fraction = np.sum(edge_map) / edge_map.size

    # Determine maximum edge blob within sliding windows.
    H, W = edge_map.shape
    max_blob = 0
    for i in range(0, H - window_size + 1, window_size):
        for j in range(0, W - window_size + 1, window_size):
            window = edge_map[i:i+window_size, j:j+window_size]
            max_blob = max(max_blob, np.sum(window))
    blob_fraction = max_blob / (window_size * window_size)
    
    # Combine metrics into a single confidence score.
    confidence = (edge_fraction + blob_fraction) / 2.0
    return confidence, edge_map
