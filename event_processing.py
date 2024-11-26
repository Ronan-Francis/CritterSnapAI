from PIL import ImageDraw
import os
from imageObj import ImageObject

def copy_events_with_red_box(events, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for i, event in enumerate(events):
        image = event.get_image()
        # Draw a red box around the suspected area
        draw = ImageDraw.Draw(image)
        width, height = image.size
        box_size = 50  # Size of the red box
        box = [(width // 2 - box_size, height // 2 - box_size), (width // 2 + box_size, height // 2 + box_size)]
        draw.rectangle(box, outline="red", width=5)
        
        # Save the image to the output directory
        image.save(os.path.join(output_directory, f"event_{i}.png"))