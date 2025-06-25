from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from utils.azureopenai_api import generate_code_with_output

router = APIRouter()

class GenerateCodeInput(BaseModel):
    prompt: str
    language: str
    dataset_url: str

@router.post("/generate_code_with_output")
def generate_code_api(input_data: GenerateCodeInput):
    try:
        code = generate_code_with_output(input_data.prompt, input_data.language, input_data.dataset_url)
        return {"code": code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
