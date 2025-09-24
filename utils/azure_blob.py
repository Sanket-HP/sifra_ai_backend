import os
from azure.storage.blob import BlobServiceClient

# Load connection string from environment variable
connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

if not connect_str:
    raise EnvironmentError("❌ ERROR: 'AZURE_STORAGE_CONNECTION_STRING' not found in environment variables!")

# Blob container name
container_name = "datasets"

# Initialize Blob service
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

# Ensure container exists
try:
    container_client = blob_service_client.get_container_client(container_name)
    if not container_client.exists():
        container_client.create_container()
except Exception as e:
    raise RuntimeError(f"❌ Failed to access or create container: {e}")

def upload_dataset(file: bytes, filename: str) -> str:
    try:
        blob_client = container_client.get_blob_client(blob=filename)
        blob_client.upload_blob(file, overwrite=True)
        return blob_client.url
    except Exception as e:
        raise RuntimeError(f"❌ Upload failed: {e}")
