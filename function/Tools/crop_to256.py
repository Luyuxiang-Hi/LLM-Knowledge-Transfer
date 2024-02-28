import os
from PIL import Image
import torchvision.transforms.functional as TF

def crop_and_save_images(origin_folder, target_folder):
    # Create output directories for cropped images and labels
    cropped_images_folder = os.path.join(target_folder, 'image')
    cropped_labels_folder = os.path.join(target_folder, 'label')
    os.makedirs(cropped_images_folder, exist_ok=True)
    os.makedirs(cropped_labels_folder, exist_ok=True)

    # Get all image filenames
    origin_images_folder = os.path.join(origin_folder, 'image')
    origin_labels_folder = os.path.join(origin_folder, 'label')
    image_filenames = [f for f in os.listdir(origin_images_folder) if f.endswith('.tif')]

    for filename in image_filenames:
        # Load the image and corresponding label
        image_path = os.path.join(origin_images_folder, filename)
        label_path = os.path.join(origin_labels_folder, filename)  # assuming label has same filename
        image = Image.open(image_path)
        label = Image.open(label_path)
        image = TF.resize(image, [512, 512])
        label = TF.resize(label, [512, 512])

        # top, left, height, width
        coordinates = [
            (0, 0, 256, 256),  # Top-Left
            (256, 0, 256, 256),  # Top-Right
            (0, 256, 256, 256),  # Bottom-Left
            (256, 256, 256, 256)  # Bottom-Right
        ]

        # Crop and save each part
        for i, coord in enumerate(coordinates):
            cropped_image = TF.crop(image, *coord)
            cropped_label = TF.crop(label, *coord)

            cropped_image_path = os.path.join(cropped_images_folder, f"{filename[:-4]}_part{i}.tif")
            cropped_label_path = os.path.join(cropped_labels_folder, f"{filename[:-4]}_part{i}.tif")

            cropped_image.save(cropped_image_path)
            cropped_label.save(cropped_label_path)

# Example usage
oringin_path = r'./'
target_folder = r'../2022.05.17_GH_2m_256'
crop_and_save_images(oringin_path, target_folder)
