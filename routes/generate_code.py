from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import requests

from utils.azureopenai_api import generate_code_with_output

router = APIRouter()

class GenerateCodeInput(BaseModel):
    prompt: str
    language: str
    dataset_url: str

@router.post("/generate_code_with_output")
def generate_code_api(input_data: GenerateCodeInput):
    # Validate that dataset_url is accessible
    try:
        response = requests.get(input_data.dataset_url)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Dataset URL not accessible")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid dataset URL: {e}")

    # Call the code generation utility
    try:
        code = generate_code_with_output(
            input_data.prompt,
            input_data.language,
            input_data.dataset_url
        )
        return {"code": code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")
