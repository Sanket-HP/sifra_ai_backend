# routes/generate_code.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from utils.openai_api import generate_code
import io
import contextlib

router = APIRouter()

class GenerateCodeInput(BaseModel):
    prompt: str
    language: str
    dataset_url: str

@router.post("/generate_code_with_output")
def generate_code_with_output(data: GenerateCodeInput):
    print("\ud83d\udcc5 Incoming Request:", data.dict())

    gpt_prompt = (
        f"You are a data scientist. Using {data.language}, write code to load dataset "
        f"from this URL: {data.dataset_url} and perform the following task: {data.prompt}"
    )

    try:
        code = generate_code(gpt_prompt, data.language)
    except Exception as e:
        print("\u274c Code generation failed:", str(e))
        raise HTTPException(status_code=500, detail="Code generation failed. Check server logs.")

    # Optional execution for Python
    if data.language.lower() == "python":
        try:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(code, {})
            program_output = output.getvalue()
        except Exception as e:
            program_output = f"Execution Error: {str(e)}"
    else:
        program_output = "Execution supported only for Python in this version."

    return {"code": code, "output": program_output}

