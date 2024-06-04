import os
import glob
import numpy as np
from PIL import Image
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import pySPM as spm

# Function to read configuration from a text file
def read_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or '=' not in line:
                continue  # Skip empty lines and lines without '='
            name, value = line.split('=', 1)  # Split only on the first '='
            config[name.strip()] = value.strip()
    return config

# Read configuration from config.txt
config = read_config('config.txt')
CONNECTION_STRING = config['AZURE_STORAGE_CONNECTION_STRING']
CONTAINER_NAME = config['AZURE_CONTAINER_NAME']

# Directory where your Bruker files and PNG images are located
bruker_directory = r"data/unannotated_images/bruker_files"
png_directory = r"data/unannotated_images/pngs"

# Create the 'pngs' directory if it does not exist
os.makedirs(png_directory, exist_ok=True)

# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# Use glob to find all files with the Bruker extension pattern
file_pattern = os.path.join(bruker_directory, "*.[0-9][0-9][0-9]")
files = glob.glob(file_pattern)

for file in files:
    try:
        scan = spm.Bruker(file)
        
        # Process the "Height Sensor" channel for each file
        height = scan.get_channel("Height Sensor").correct_plane().zero_min().filter_scars_removal()

        # Get the height data as a numpy array
        height_data = height.pixels

        # Normalize the height data to span the full 0-255 8-bit range
        normalized_data = 255 * (height_data - np.min(height_data)) / (np.max(height_data) - np.min(height_data))
        normalized_data = normalized_data.astype(np.uint8)

        # Reverse the height array if it's backwards
        normalized_data = np.flipud(normalized_data)  # For vertical flip

        # Save the height data as an 8-bit PNG image
        base_name = os.path.splitext(os.path.basename(file))[0]  # Get the base name without extension
        numeric_extension = os.path.splitext(file)[1][1:]  # Get numeric part, remove leading '.'
        save_filename = f"{base_name}{numeric_extension}.png"  # Construct filename with numeric part and '.png'
        save_path = os.path.join(png_directory, save_filename)
        
        Image.fromarray(normalized_data).save(save_path)

        print(f"Successfully saved: {save_path}")

    except Exception as e:
        print(f"Failed to process {file}. Error: {e}")

# Upload PNG images to Azure Blob Storage
png_files = glob.glob(os.path.join(png_directory, "*.png"))
for png_file in png_files:
    try:
        file_name = os.path.basename(png_file)
        blob_client = container_client.get_blob_client(file_name)
        with open(png_file, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        print(f"Successfully uploaded to Azure: {file_name}")

    except Exception as e:
        print(f"Failed to upload {png_file}. Error: {e}")
