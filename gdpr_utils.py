from PIL import Image

def is_not_gdpr_image(img, white_pixel_threshold):
    """
    Checks if the image qualifies as a GDPR image based on the count of pure-white (#ffffff) pixels.

    Parameters:
    - img: Image object to check
    - white_pixel_threshold: Number of white pixels above which we consider the image GDPR-masked

    Returns:
    - True if the image is NOT GDPR-protected, False otherwise.
    """
    # Convert image to RGB format and load pixel data
    img = img.convert("RGB")
    pixels = img.getdata()

    # Count pure-white pixels
    white_pixel_count = sum(1 for pixel in pixels if pixel == (255, 255, 255))
    
    return white_pixel_count <= white_pixel_threshold
