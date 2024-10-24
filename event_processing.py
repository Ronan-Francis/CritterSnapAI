from PIL import ImageDraw
import os

def copy_events_with_red_box(events, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    for i, event in enumerate(events):
        # Draw a red box around the suspected area
        draw = ImageDraw.Draw(event)
        width, height = event.size
        box_size = 50  # Size of the red box
        box = [(width // 2 - box_size, height // 2 - box_size), (width // 2 + box_size, height // 2 + box_size)]
        draw.rectangle(box, outline="red", width=5)
        
        # Save the image to the output directory
        event.save(os.path.join(output_directory, f"event_{i}.png"))