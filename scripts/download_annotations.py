import os
from azure.storage.blob import BlobServiceClient, ContainerClient

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

# Define the base directory (you can adjust this to your specific base directory)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Define the path to the configuration file
config_file_path = os.path.join(base_dir, 'data', 'configs', 'config_download.txt')

# Read configuration from config_download.txt
config = read_config(config_file_path)
CONNECTION_STRING = config['AZURE_STORAGE_CONNECTION_STRING']
CONTAINER_NAME = config['AZURE_CONTAINER_NAME']

# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# Directories to save downloaded files
output_image_dir = os.path.join(base_dir, 'data', 'annotated_images', 'images')
output_annotation_dir = os.path.join(base_dir, 'data', 'annotated_images', 'Annotations')
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_annotation_dir, exist_ok=True)

# Function to download files from Azure Blob Storage
def download_files(container_client, output_dir, file_extension):
    blobs_list = container_client.list_blobs()
    for blob in blobs_list:
        if blob.name.endswith(file_extension):
            download_file_path = os.path.join(output_dir, os.path.basename(blob.name))
            blob_client = container_client.get_blob_client(blob.name)
            with open(download_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            print(f"Downloaded: {blob.name} to {download_file_path}")

# Download images and annotations
download_files(container_client, output_image_dir, '.png')
download_files(container_client, output_annotation_dir, '.xml')

print("All files downloaded successfully.")
