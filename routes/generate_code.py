from fastapi import APIRouter
from models.schemas import GenerateCodeInput
from utils.azureopenai_api import generate_code_with_output
from services.code_executor import execute_code_blocks
from services.dashboard_generator import generate_dashboard  # fixed
from services.insight_generator import generate_insights     # fixed

router = APIRouter()

@router.post("/generate_code_with_output")
def generate_code_api(input_data: GenerateCodeInput):
    """
    Main endpoint to generate code, execute it, optionally build dashboards,
    and generate insights.
    """
    try:
        # Step 1: Generate code using LLM
        code = generate_code_with_output(
            input_data.prompt,
            input_data.language,
            input_data.dataset_url
        )

        # Step 2: Execute code and capture result
        result = execute_code_blocks(
            code,
            language=input_data.language,
            dataset_url=input_data.dataset_url
        )

        # Step 3: Auto-generate dashboard (if requested in prompt)
        dashboard = None
        if "dashboard" in input_data.prompt.lower():
            dashboard = generate_dashboard(
                dataset_url=input_data.dataset_url,
                language=input_data.language
            )

        # Step 4: Auto-generate insights
        insights = generate_insights(
            dataset_url=input_data.dataset_url
        )

        return {
            "blocks": result,
            "dashboard": dashboard,  # HTML/JSON for visualization
            "insights": insights     # Text summary
        }

    except Exception as e:
        return {
            "blocks": [
                {
                    "input": "",
                    "output": "",
                    "error": str(e),
                    "visualizations": []
                }
            ],
            "dashboard": None,
            "insights": None
        }
