import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_image(title, img, cmap=None):
    """Helper function to plot images."""
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def load_image():
    """Generate a synthetic image (since we can't load files here)."""
    # Creating a synthetic image with shapes for demonstration
    img = np.zeros((400, 400, 3), dtype="uint8")
    
    # Draw some synthetic shapes to simulate an animal or objects
    cv2.circle(img, (200, 200), 100, (255, 255, 255), -1)  # Simulate a round object
    cv2.rectangle(img, (50, 50), (100, 100), (255, 255, 255), -1)  # Simulate a square
    
    return img

def preprocess_image(img):
    """Convert the image to grayscale and apply Gaussian blur."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blurred

def detect_edges(img):
    """Detect edges using Canny edge detection."""
    edges = cv2.Canny(img, 100, 200)
    return edges

def find_contours(edges):
    """Find contours in the edge-detected image."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(edges)
    cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 1)
    return contours, contour_img

# Step 1: Load the image
img = load_image()
plot_image("Original Image", img)

# Step 2: Preprocess the image (grayscale and blur)
gray, blurred = preprocess_image(img)
plot_image("Grayscale Image", gray, cmap='gray')
plot_image("Blurred Image", blurred, cmap='gray')

# Step 3: Perform edge detection
edges = detect_edges(blurred)
plot_image("Edge Detection (Canny)", edges, cmap='gray')

# Step 4: Find and visualize contours
contours, contour_img = find_contours(edges)
plot_image("Contours", contour_img, cmap='gray')

# Print contour count as part of the decision tree process
print(f"Number of contours detected: {len(contours)}")
