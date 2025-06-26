from fastapi import APIRouter
from pydantic import BaseModel
import io
import contextlib
import traceback

router = APIRouter()

class CodeInput(BaseModel):
    code: str

# Optional: global variable context to maintain session (careful for concurrent users!)
exec_globals = {}

@router.post("/run_code")
async def run_code(data: CodeInput):
    try:
        # Capture standard output
        stdout_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer):
            exec(data.code, exec_globals)  # Using exec_globals for variable reuse

        return {
            "output": stdout_buffer.getvalue(),
            "error": None
        }

    except Exception as e:
        return {
            "output": "",
            "error": traceback.format_exc()
        }
