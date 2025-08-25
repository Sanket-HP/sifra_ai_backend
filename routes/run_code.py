# routes/run_code.py
from fastapi import APIRouter
from pydantic import BaseModel
from services.code_executor import execute_code_blocks
import traceback

router = APIRouter()

# Supported languages
SUPPORTED_LANGUAGES = ["python", "r", "sql"]

class CodeInput(BaseModel):
    code: str
    language: str = "python"  # default
    dataset_url: str | None = None
    dataset_type: str | None = None  # "csv", "json", "xlsx"


@router.post("/run_code")
async def run_code(data: CodeInput):
    """
    Execute code cell-wise with real-time outputs, visualizations,
    and optional dataset integration (CSV, JSON, XLSX).
    Supports Python, R, SQL.
    """
    try:
        # Validate language
        if data.language.lower() not in SUPPORTED_LANGUAGES:
            return {
                "blocks": [
                    {
                        "input": data.code,
                        "output": "",
                        "error": f"Unsupported language: {data.language}. "
                                 f"Supported: {SUPPORTED_LANGUAGES}",
                        "visualizations": [],
                    }
                ]
            }

        results = execute_code_blocks(
            data.code,
            language=data.language.lower(),
            dataset_url=data.dataset_url,
            dataset_type=data.dataset_type
        )
        return {"blocks": results}

    except Exception:
        return {
            "blocks": [
                {
                    "input": data.code,
                    "output": "",
                    "error": traceback.format_exc(),
                    "visualizations": [],
                }
            ]
        }
