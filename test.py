from datetime import timedelta


def group_images_by_event(images_with_dates, time_gap_minutes=30):
    events = []
    current_event = []
    last_image_time = None
    
    for image, date_time in images_with_dates:
        if last_image_time and (date_time - last_image_time > timedelta(minutes=time_gap_minutes)):
            # Large gap detected, start a new event
            events.append(current_event)
            current_event = []
        
        current_event.append(image)
        last_image_time = date_time
    
    # Add the last event if it exists
    if current_event:
        events.append(current_event)
    
    return events