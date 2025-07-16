from fastapi import APIRouter, HTTPException
from models.schemas import GenerateCodeInput
from utils.azureopenai_api import generate_code_with_output
from services.code_executor import execute_code_blocks

router = APIRouter()

@router.post("/generate_code_with_output")
def generate_code_api(input_data: GenerateCodeInput):
    try:
        code = generate_code_with_output(
            input_data.prompt,
            input_data.language,
            input_data.dataset_url
        )
        result = execute_code_blocks(code)
        return {"blocks": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
