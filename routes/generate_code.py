from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.azureopenai_api import generate_code_with_output
import traceback

router = APIRouter()

class GenerateCodeInput(BaseModel):
    prompt: str
    language: str
    dataset_url: str

@router.post("/generate_code_with_output")
def generate_code_api(input_data: GenerateCodeInput):
    try:
        # Step 1: Generate complete code
        full_code = generate_code_with_output(
            input_data.prompt,
            input_data.language,
            input_data.dataset_url
        )

        # Step 2: Split into blocks
        code_blocks = full_code.split('\n\n')  # Adjust if needed
        executed_blocks = []

        # Safe namespace to persist variables across blocks
        exec_globals = {}

        # Step 3: Execute each block and capture output
        for block in code_blocks:
            output = ""
            try:
                exec(block, exec_globals)
            except Exception as e:
                output = f"Error: {traceback.format_exc()}"
            else:
                output = "Executed successfully."  # Or capture printed output via redirect stdout if needed

            executed_blocks.append({
                "code": block,
                "output": output
            })

        # Step 4: Return structured list
        return {"blocks": executed_blocks}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
