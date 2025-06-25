from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.azureopenai_api import generate_code
import io
import contextlib

router = APIRouter()

class GenerateCodeInput(BaseModel):
    prompt: str
    language: str
    dataset_url: str

@router.post("/generate_code_with_output")
async def generate_code_with_output(data: GenerateCodeInput):
    print("📥 Incoming Request:", data.dict())

    gpt_prompt = (
        f"You are a data scientist. Using {data.language}, write code to load dataset "
        f"from this URL: {data.dataset_url} and perform the following task: {data.prompt}"
    )

    try:
        code = generate_code(gpt_prompt, data.language)
    except Exception as e:
        print("❌ Code generation failed:", str(e))
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

    # Run the code only if Python
    if data.language.lower() == "python":
        try:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(code, {})
            program_output = output.getvalue()
        except Exception as e:
            program_output = f"Execution Error: {str(e)}"
    else:
        program_output = "Execution only supported for Python."

    return {"code": code, "output": program_output}
