from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.azureopenai_api import generate_code_with_output
import traceback
import io
import contextlib

router = APIRouter()

class GenerateCodeInput(BaseModel):
    prompt: str
    language: str
    dataset_url: str

@router.post("/generate_code_with_output")
def generate_code_api(input_data: GenerateCodeInput):
    try:
        # Step 1: Generate code from OpenAI or logic
        full_code = generate_code_with_output(
            input_data.prompt,
            input_data.language,
            input_data.dataset_url
        )

        # Step 2: Split code by logic block (adjust if needed)
        code_blocks = full_code.split('\n\n')  # Ensure backend generates blocks like this

        executed_blocks = []
        exec_globals = {}

        # Step 3: Run each code block, capture print output
        for block in code_blocks:
            output_buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(output_buffer):
                    exec(block, exec_globals)
                output = output_buffer.getvalue().strip() or "Executed successfully."
            except Exception:
                output = f"Error:\n{traceback.format_exc()}"
            finally:
                output_buffer.close()

            executed_blocks.append({
                "code": block,
                "output": output
            })

        return {"blocks": executed_blocks}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
