from fastapi import APIRouter
from models.schemas import GenerateCodeInput
from utils.azureopenai_api import generate_code_with_output
from services.code_executor import execute_code_blocks

router = APIRouter()

@router.post("/generate_code_with_output")
def generate_code_api(input_data: GenerateCodeInput):
    """
    Endpoint to:
    1. Generate code from user prompt via LLM.
    2. Execute the code cell-wise.
    3. Return execution results (stdout, errors, visualizations).
    """
    try:
        # Step 1: Generate code using LLM
        code = generate_code_with_output(
            input_data.prompt,
            input_data.language,
            input_data.dataset_url
        )

        # Step 2: Execute generated code and capture result
        result = execute_code_blocks(
            code,
            language=input_data.language,
            dataset_url=input_data.dataset_url
        )

        # Final Response (no dashboard/insights)
        return {
            "blocks": result
        }

    except Exception as e:
        return {
            "blocks": [
                {
                    "input": "",
                    "output": "",
                    "error": str(e),
                    "visualizations": []
                }
            ]
        }
