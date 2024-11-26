from PIL import Image

def is_gdpr_image(img, white_pixel_threshold=50000):
    """
    Checks if the image qualifies as a GDPR image based on the count of white (#ffffff) pixels.

    Parameters:
    - img: Image object to check.
    - white_pixel_threshold: Number of white pixels to classify as a GDPR image.

    Returns:
    - True if the image is a GDPR image, False otherwise.
    """
    # Convert image to RGB format and load pixel data
    img = img.convert("RGB")
    pixels = img.getdata()

    # Count white pixels
    white_pixel_count = sum(1 for pixel in pixels if pixel == (255, 255, 255))
    
    # print(f"White pixel count: {white_pixel_count}")
    return white_pixel_count >= white_pixel_threshold
