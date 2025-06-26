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

        # Step 2: Split code into logical blocks
        code_blocks = full_code.split('\n\n')

        executed_blocks = []
        exec_globals = {}

        for block in code_blocks:
            cleaned_block = block.strip()

            # Skip markdown or explanatory lines
            if (
                cleaned_block.startswith("```")
                or "Sure!" in cleaned_block
                or "Make sure" in cleaned_block
                or cleaned_block.lower().startswith("here is")
                or cleaned_block.lower().startswith("example")
            ):
                continue

            output = ""
            try:
                # Capture stdout using contextlib
                stdout_buffer = io.StringIO()
                with contextlib.redirect_stdout(stdout_buffer):
                    exec(cleaned_block, exec_globals)

                output = stdout_buffer.getvalue().strip()
                if not output:
                    output = "Executed successfully."

            except Exception as e:
                output = f"Error:\n{traceback.format_exc()}"

            executed_blocks.append({
                "code": cleaned_block,
                "output": output
            })

        return {"blocks": executed_blocks}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
