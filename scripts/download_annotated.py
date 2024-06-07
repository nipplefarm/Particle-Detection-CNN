import os
import zipfile
from azure.storage.blob import BlobServiceClient

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

# Define the path to the configuration file
config_file_path = os.path.join('..', 'data', 'configs', 'config_upload.txt')

# Read configuration from config_upload.txt
config = read_config(config_file_path)
CONNECTION_STRING = config['AZURE_STORAGE_CONNECTION_STRING']
CONTAINER_NAME = config['AZURE_CONTAINER_NAME']

# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# Directory to check for zip files
zips_dir = os.path.join('..', 'data', 'zips')

# Directories to save extracted files
output_image_dir = os.path.join('..', 'data', 'annotated_images', 'images')
output_annotation_dir = os.path.join('..', 'data', 'annotated_images', 'Annotations')
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_annotation_dir, exist_ok=True)

# Check for zip files in the zips directory
zip_files = [f for f in os.listdir(zips_dir) if f.endswith('.zip')]

for zip_file in zip_files:
    zip_file_path = os.path.join(zips_dir, zip_file)
    
    # Extract contents of the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join('..', 'data', 'annotated_images'))
        print(f"Extracted files from {zip_file_path} to data/annotated_images")
    
    # Move extracted files to the appropriate directories if needed
    for root, dirs, files in os.walk(os.path.join('..', 'data', 'annotated_images')):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.png'):
                target_dir = output_image_dir
            elif file.endswith('.xml'):
                target_dir = output_annotation_dir
            else:
                continue
            
            # Ensure target directory exists
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, file)
            
            # Move file to the target directory
            os.replace(file_path, target_path)
            print(f"Moved {file_path} to {target_path}")

# Upload files to Azure Blob Storage
def upload_files(directory, container_path):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png') or file.endswith('.xml'):
                file_path = os.path.join(root, file)
                blob_name = os.path.relpath(file_path, directory)
                blob_client = container_client.get_blob_client(os.path.join(container_path, blob_name))
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                print(f"Uploaded: {blob_name}")

# Upload images and annotations
upload_files(output_image_dir, 'images')
upload_files(output_annotation_dir, 'Annotations')

print("All files extracted, moved, and uploaded successfully.")
