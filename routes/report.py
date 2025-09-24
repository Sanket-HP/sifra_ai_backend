# ✅ FILE: routes/report.py

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, JSONResponse
import os

from models.schemas import GenerateCodeInput
from utils.azureopenai_api import generate_code_with_output
from services.code_executor import execute_code_blocks

router = APIRouter(prefix="/report", tags=["Reports"])


@router.post("/generate")
async def generate_report(input_data: GenerateCodeInput, format: str = Query("pdf", description="Report format: pdf or excel")):
    """
    Generate report:
    1. Generate report-focused code via LLM
    2. Execute generated code
    3. If output file is created, return it; else return execution results
    """

    # Step 1: Generate code with "report" mode
    code = generate_code_with_output(
        prompt=input_data.prompt,
        language=input_data.language,
        dataset_url=input_data.dataset_url,
        mode="report"
    )

    # Step 2: Execute generated code
    results = execute_code_blocks(
        code,
        language=input_data.language,
        dataset_url=input_data.dataset_url
    )

    # Step 3: Force-check for expected report filenames
    expected_files = [
        "student_report.pdf",
        "student_report.xlsx"
    ]
    for f in expected_files:
        if os.path.exists(f):
            return FileResponse(f, filename=f)

    # Step 4: If no file was found, return logs instead
    return JSONResponse({
        "message": "⚠️ Report file not found. Returning execution results instead.",
        "report": results,
        "generated_code": code
    })
