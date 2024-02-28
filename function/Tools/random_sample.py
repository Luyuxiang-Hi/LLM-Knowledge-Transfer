import random
import os
import shutil

target_directory = r'./02pv_test'
source_directory = r'./02pv_train'

# Total number of files in the image directory
total_images = len(os.listdir(os.path.join(source_directory, 'image')))

# Calculate 20% of the total images
num_to_select = int(total_images * 0.2)

# Randomly select 20% of the images
selected_images = random.sample(os.listdir(os.path.join(source_directory, 'image')), num_to_select)

# Directories for the new subset of images and labels
subset_image_directory = os.path.join(target_directory, 'image')
subset_label_directory = os.path.join(target_directory, 'label')

# Create the directories if they don't exist
os.makedirs(subset_image_directory, exist_ok=True)
os.makedirs(subset_label_directory, exist_ok=True)

# Copy the selected images and their corresponding labels to the new directories
for image_file in selected_images:
    # Copy image
    shutil.move(os.path.join(os.path.join(source_directory, 'image'), image_file), os.path.join(subset_image_directory, image_file))

    # Copy corresponding label
    label_file = image_file  # Since label file has been renamed to match image file
    shutil.move(os.path.join(os.path.join(source_directory, 'label'), label_file), os.path.join(subset_label_directory, label_file))

# selected_images  # Display the list of selected images for reference
print(len(selected_images))

