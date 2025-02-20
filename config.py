# directory_path = r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\example\as_conservationistFrankfurt\IE_Forest_County_Wicklow_21_loc_01-20241031T145429Z-001\IE_Forest_County_Wicklow_21_loc_01"  # Adjust this with your correct path
# directory_path = r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\example\eccv_18_all_images_sm"  # Adjust this with your correct path
directory_path = r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\example\ds_researchATU"
output_directory = r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\output"  # Adjust this with your correct path
output_log_path = r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\output\output_log.txt"  # Adjust this with your correct path
change_threshold = 0.1 # Adjust this based on your images
#Low values: 0.1 to 0.6
#Medium values: 0.61 to 0.80
#High values: 0.81 to 1.00

# use the above values to adjust the change_threshold based on the images you are working with
# The change_threshold is the pixel change threshold to classify an event
white_pixel_threshold = 50000  # Adjust this based on your images

# Edge detection configuration parameters
edge_threshold = 50           # For Sobel gradient thresholding
edge_confidence_threshold = 0.15  # Tune this value based on your experiments
window_size = 20              # For blob detection in the edge map

base_url = "https://lilawildlife.blob.core.windows.net/lila-wildlife/wcs-unzipped/animals/0011/"
start_file = 1    # Starting file number
end_file = 700    # Ending file number

animal_training_path = r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\example\as_conservationistFrankfurt\IE_Forest_County_Wicklow_21_loc_01-20241031T145429Z-001"  # Adjust this with your correct path
non_animal_training_path = r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\example\eccv_18_all_images_sm"  # Adjust this with your correct path
