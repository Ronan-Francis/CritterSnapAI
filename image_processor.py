from dataclasses import dataclass
from datetime import datetime
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

@dataclass
class ImageObject:
    image: Image.Image
    date: datetime
    file_path: str

    def get_image(self) -> Image.Image:
        return self.image

    def get_date(self) -> datetime:
        return self.date

    def get_file_path(self) -> str:
        return self.file_path

class ImageProcessor:
    @staticmethod
    def rgb2gray(image_array: np.ndarray) -> np.ndarray:
        return np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])

    @staticmethod
    def sobel_gradients(gray_image: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def compute_edge_confidence(pil_image: Image.Image, edge_threshold: int = 50, window_size: int = 20):
        gray_array = np.array(pil_image.convert('L'), dtype=np.float32)
        grad_mag = ImageProcessor.sobel_gradients(gray_array)
        edge_map = (grad_mag > edge_threshold).astype(np.uint8)
        edge_fraction = np.sum(edge_map) / edge_map.size

        H, W = edge_map.shape
        max_blob = 0
        for i in range(0, H - window_size + 1, window_size):
            for j in range(0, W - window_size + 1, window_size):
                window = edge_map[i:i+window_size, j:j+window_size]
                max_blob = max(max_blob, np.sum(window))
        blob_fraction = max_blob / (window_size * window_size)
        confidence = (edge_fraction + blob_fraction) / 2.0
        return confidence, edge_map

    @staticmethod
    def downsample_image(img: Image.Image, scale_factor: float = 0.5) -> Image.Image:
        width, height = img.size
        return img.resize((int(width * scale_factor), int(height * scale_factor)))

    @staticmethod
    def measure_changes(past: Image.Image, present: Image.Image, future: Image.Image, scale_factor: float = 0.5) -> float:
        if past is None or present is None or future is None:
            return 0.0

        past_ds = ImageProcessor.downsample_image(past, scale_factor)
        present_ds = ImageProcessor.downsample_image(present, scale_factor)
        future_ds = ImageProcessor.downsample_image(future, scale_factor)

        past_array = np.array(past_ds.convert("L"), dtype=np.float32)
        present_array = np.array(present_ds.convert("L"), dtype=np.float32)
        future_array = np.array(future_ds.convert("L"), dtype=np.float32)

        ssim_prev_curr = ssim(past_array, present_array, data_range=255)
        ssim_curr_next = ssim(present_array, future_array, data_range=255)
        avg_ssim = (ssim_prev_curr + ssim_curr_next) / 2

        return 1 - avg_ssim
