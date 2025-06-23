from fastapi import APIRouter
from pydantic import BaseModel
import io
import contextlib

router = APIRouter()

class CodeInput(BaseModel):
    code: str

@router.post("/run_code")
def run_code(data: CodeInput):
    try:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(data.code, {})
        return {"output": output.getvalue()}
    except Exception as e:
        return {"error": str(e)}
