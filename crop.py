import csv
import os
from PIL import Image

def crop_image(input_dir, csv_file):
    # Open the CSV file
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        # Iterate through the rows in the CSV
        for row in reader:
            filename = row['filename']
            x = float(row['normalized_x'])
            y = float(row['normalized_y'])

            # Construct the image path
            image_path = os.path.join(input_dir, filename)

            if os.path.exists(image_path):
                # Open the image
                with Image.open(image_path) as img:
                    width, height = img.size

                    # Denormalize the coordinates
                    x = int(x * width)
                    y = int(y * height)

                    # Define the crop box
                    left = max(0, x - 320)
                    top = max(0, y - 320)
                    right = min(width, x + 320)
                    bottom = min(height, y + 320)

                    # Crop the image
                    cropped = img.crop((left, top, right, bottom))

                    # Save the cropped image (replacing the original)
                    cropped.save(image_path)
            else:
                print(f"Image {filename} not found in {input_dir}.")

crop_image("datasets/crops-crosswalk-tags/test", "datasets/crops-crosswalk-tags/test/test.csv")
crop_image("datasets/crops-obstacle-tags/test", "datasets/crops-obstacle-tags/test/test.csv")
crop_image("datasets/crops-surfaceproblem-tags/test", "datasets/crops-surfaceproblem-tags/test/test.csv")
crop_image("datasets/crops-curbramp-tags/test", "datasets/crops-curbramp-tags/test/test.csv")
