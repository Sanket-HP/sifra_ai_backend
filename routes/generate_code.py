from fastapi import APIRouter
from models.schemas import GenerateCodeInput
from utils.azureopenai_api import generate_code_with_output
from services.code_executor import execute_code_blocks
from typing import List, Dict, Any
import re

router = APIRouter()


@router.post("/generate_code_with_output")
def generate_code_api(input_data: GenerateCodeInput) -> Dict[str, Any]:
    """
    Endpoint to:
    1. Generate code from user prompt via LLM.
    2. If dashboard toggle ON → return dashboard code only (no cell execution).
    3. If toggle OFF → split into cells and execute step-by-step.
    4. Return execution results (stdout, errors, visualizations).
    """
    try:
        # Step 1: Decide mode based on toggle
        mode = "dashboard" if input_data.is_dashboard else "default"

        # Step 2: Generate full code using LLM
        code: str = generate_code_with_output(
            input_data.prompt,
            input_data.language,
            input_data.dataset_url,
            mode=mode
        )

        # 🚀 Dashboard mode → just return raw code
        if input_data.is_dashboard:
            return {
                "mode": "dashboard",
                "blocks": [
                    {
                        "input": code,
                        "output": "✅ Dashboard code generated. Run separately as a Streamlit/Dash app.",
                        "error": "",
                        "visualizations": []
                    }
                ]
            }

        # 🚀 Notebook mode → split into logical cells
        raw_cells: List[str] = []

        # Split by # Cell markers or double newlines
        for part in re.split(r"(#\s*Cell\s*\d+|#\s*%%|\n\n)", code):
            part = part.strip()
            if part and not re.match(r"(#\s*Cell\s*\d+|#\s*%%)", part):
                raw_cells.append(part)

        if not raw_cells:
            return {
                "mode": "notebook",
                "blocks": [
                    {
                        "input": code,
                        "output": "",
                        "error": "⚠️ No code cells were generated.",
                        "visualizations": []
                    }
                ]
            }

        # Step 3: Execute each cell separately
        blocks: List[Dict[str, Any]] = []
        for cell in raw_cells:
            try:
                result = execute_code_blocks(
                    cell,
                    language=input_data.language,
                    dataset_url=input_data.dataset_url
                )

                # execute_code_blocks may return a list (if multiple sub-cells executed)
                if isinstance(result, list):
                    blocks.extend(result)
                else:
                    blocks.append(result)

            except Exception as e:
                blocks.append({
                    "input": cell,
                    "output": "",
                    "error": str(e),
                    "visualizations": []
                })

        # Step 4: Return executed cells
        return {"mode": "notebook", "blocks": blocks}

    except Exception as e:
        # Fatal error fallback
        return {
            "mode": "error",
            "blocks": [
                {
                    "input": "",
                    "output": "",
                    "error": f"🔥 Fatal backend error: {str(e)}",
                    "visualizations": []
                }
            ]
        }
