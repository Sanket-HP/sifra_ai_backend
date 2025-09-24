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
    generate_dashboard: Optional[bool] = False   # NEW: auto dashboard flag
    generate_insights: Optional[bool] = False    # NEW: auto insights flag


@router.post("/run_code")
async def run_code(data: CodeInput):
    """
    Execute code cell-wise with real-time outputs, visualizations,
    and optional dataset integration (CSV, JSON, XLSX).
    Supports Python, R, SQL.
    Optionally generates dashboards and insights.
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
                ],
                "dashboard": None,
                "insights": None,
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
                ],
                "dashboard": None,
                "insights": None,
            }

        # Step 1: Execute user-provided code
        results = execute_code_blocks(
            code=data.code,
            language=data.language.lower(),
            dataset_url=data.dataset_url,
            dataset_type=data.dataset_type,
        )

        # Step 2: Optionally auto-generate dashboard
        dashboard = None
        if data.generate_dashboard and data.dataset_url:
            dashboard = generate_dashboard(
                dataset_url=data.dataset_url,
                language=data.language.lower(),
            )

        # Step 3: Optionally auto-generate insights
        insights = None
        if data.generate_insights and data.dataset_url:
            insights = generate_insights(
                dataset_url=data.dataset_url,
            )

        return {
            "blocks": results,
            "dashboard": dashboard,
            "insights": insights,
        }

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
            ],
            "dashboard": None,
            "insights": None,
        }
