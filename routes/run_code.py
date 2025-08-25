# routes/run_code.py

from fastapi import APIRouter
from pydantic import BaseModel
from services.code_executor import execute_code_blocks
import traceback
from typing import Optional, Literal

router = APIRouter()

# Supported languages
SUPPORTED_LANGUAGES = ["python", "r", "sql"]

# Supported dataset types
SUPPORTED_DATASETS = ["csv", "json", "xlsx"]


class CodeInput(BaseModel):
    """
    Request body model for code execution.
    """
    code: str
    language: Literal["python", "r", "sql"] = "python"  # default is Python
    dataset_url: Optional[str] = None
    dataset_type: Optional[Literal["csv", "json", "xlsx"]] = None


@router.post("/run_code")
async def run_code(data: CodeInput):
    """
    Execute code cell-wise with real-time outputs, visualizations,
    and optional dataset integration (CSV, JSON, XLSX).
    Supports Python, R, SQL.
    """

    try:
        # Validate programming language
        if data.language.lower() not in SUPPORTED_LANGUAGES:
            return {
                "blocks": [
                    {
                        "input": data.code,
                        "output": "",
                        "error": (
                            f"Unsupported language: {data.language}. "
                            f"Supported languages: {SUPPORTED_LANGUAGES}"
                        ),
                        "visualizations": [],
                    }
                ]
            }

        # Validate dataset type if provided
        if data.dataset_type and data.dataset_type not in SUPPORTED_DATASETS:
            return {
                "blocks": [
                    {
                        "input": data.code,
                        "output": "",
                        "error": (
                            f"Unsupported dataset type: {data.dataset_type}. "
                            f"Supported dataset types: {SUPPORTED_DATASETS}"
                        ),
                        "visualizations": [],
                    }
                ]
            }

        # Execute code and return results
        results = execute_code_blocks(
            code=data.code,
            language=data.language.lower(),
            dataset_url=data.dataset_url,
            dataset_type=data.dataset_type,
        )

        return {"blocks": results}

    except Exception:
        # Return detailed traceback for debugging
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
