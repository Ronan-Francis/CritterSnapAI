directory_path = r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\example\as_conservationistFrankfurt\IE_Forest_County_Wicklow_21_loc_01-20241031T145429Z-001\IE_Forest_County_Wicklow_21_loc_01"  # Adjust this with your correct path
# directory_path = r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\example\eccv_18_all_images_sm"  # Adjust this with your correct path
output_directory = r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\output"  # Adjust this with your correct path
output_log_path = r"C:\Users\rf4thyrvm\Documents\CritterSnap\data\output\output_log.txt"  # Adjust this with your correct path
change_threshold = 51803623.25  # Adjust this based on your images
#Low values: 2,659,548.5 to 10,000,000
#Medium values: 10,000,000 to 50,000,000
#High values: 50,000,000 to 190,555,785.0

# use the above values to adjust the change_threshold based on the images you are working with
# The change_threshold is the pixel change threshold to classify an event
white_pixel_threshold = 50000  # Adjust this based on your images

base_url = "https://lilawildlife.blob.core.windows.net/lila-wildlife/wcs-unzipped/animals/0011/"
start_file = 1    # Starting file number
end_file = 700    # Ending file number


