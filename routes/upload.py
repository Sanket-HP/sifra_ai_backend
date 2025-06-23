from fastapi import APIRouter, File, UploadFile, HTTPException
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import os

router = APIRouter()

# Load connection string
connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
if not connect_str:
    raise EnvironmentError("❌ AZURE_STORAGE_CONNECTION_STRING not found in environment variables!")

# Blob storage setup
container_name = "datasets"
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

def upload_to_blob(file_bytes: bytes, filename: str) -> str:
    try:
        container_client = blob_service_client.get_container_client(container_name)

        # Create container if it does not exist
        if not container_client.exists():
            container_client.create_container()
            print(f"🆕 Container '{container_name}' created.")
        else:
            print(f"ℹ️ Container '{container_name}' already exists.")

        # Upload blob
        blob_client = container_client.get_blob_client(filename)
        blob_client.upload_blob(BytesIO(file_bytes), overwrite=True)

        blob_url = blob_client.url
        print(f"✅ Uploaded '{filename}' → {blob_url}")
        return blob_url

    except Exception as e:
        print("❌ Azure Blob Upload Error:", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {str(e)}")

@router.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_url = upload_to_blob(contents, file.filename)

        return {
            "message": "Upload successful",
            "file_url": file_url,
            "file_id": file.filename
        }

    except Exception as e:
        print("❌ Upload Endpoint Error:", str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
