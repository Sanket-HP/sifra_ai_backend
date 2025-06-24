# routes/upload.py
from fastapi import APIRouter, File, UploadFile, HTTPException
from azure.storage.blob import BlobServiceClient
from io import BytesIO
import os

router = APIRouter()

# Load connection string from environment variable
connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
if not connect_str:
    raise EnvironmentError("❌ AZURE_STORAGE_CONNECTION_STRING not found in environment!")

# Azure Blob Storage setup
container_name = "datasets"
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

# Upload logic
def upload_dataset(file: bytes, filename: str) -> str:
    try:
        container_client = blob_service_client.get_container_client(container_name)
        try:
            container_client.create_container()
        except Exception as e:
            print(f"⚠️ Container creation skipped (may already exist): {e}")

        blob_client = container_client.get_blob_client(blob=filename)
        blob_client.upload_blob(BytesIO(file), overwrite=True)
        return blob_client.url
    except Exception as e:
        print("❌ Azure Blob Upload Error:", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to upload: {str(e)}")

@router.post("/upload_dataset")
async def upload(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        url = upload_dataset(contents, file.filename)
        return {
            "message": "Upload successful",
            "file_url": url,
            "file_id": file.filename
        }
    except Exception as e:
        print("❌ FastAPI Upload Route Error:", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {str(e)}")
