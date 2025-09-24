# ✅ FILE: utils/azureopenai_api.py

import os
import re
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)


def clean_code_output(raw_output: str) -> str:
    """
    Clean LLM output to return only code (remove ```python fences etc.).
    """
    # Remove markdown code fences if present
    cleaned = re.sub(r"^```[a-zA-Z]*\n", "", raw_output.strip())
    cleaned = re.sub(r"```$", "", cleaned)
    return cleaned.strip()


def generate_code_with_output(
    prompt: str,
    language: str,
    dataset_url: str,
    mode: str = "default"
) -> str:
    """
    Generate executable code via Azure OpenAI.

    Modes:
    - mode="dashboard": produce interactive dashboards (Streamlit)
    - mode="report": produce summarization & file export (PDF or Excel)
    - mode="default": notebook-style step-by-step cells (NO dashboards)
    """
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    # Base system message
    system_message = f"You generate only clean and executable {language} code."

    # Mode-specific guidance
    if mode == "dashboard":
        system_message += (
            " The user needs an interactive dashboard. "
            "✅ Generate code using Streamlit that loads the dataset "
            "with pandas (pd.read_csv, pd.read_excel, or pd.read_json) and produces charts/plots. "
            "Ensure the app runs on Azure by binding to host '0.0.0.0' and port from "
            "the PORT environment variable like:\n"
            "import os\n"
            "port = int(os.getenv('PORT', 8000))\n"
            "For Streamlit, use: `streamlit run file.py --server.port $PORT --server.address 0.0.0.0`."
        )
    elif mode == "report":
        system_message += (
            " The user needs an automated report. "
            "✅ Generate code that:\n"
            "- Loads the dataset with pandas\n"
            "- Cleans column names (lowercase, underscores)\n"
            "- Computes summary statistics (total students, unique classes, divisions if exist, marks if exist)\n"
            "- Creates tables (class distribution, division distribution)\n"
            "- Builds visualizations (matplotlib/seaborn bar chart, pie chart, histogram)\n"
            "- Exports the entire report to a **local file** named 'student_report.pdf' "
            "using matplotlib.backends.backend_pdf.PdfPages, OR 'student_report.xlsx' using pandas.ExcelWriter.\n"
            "⚠️ DO NOT use Streamlit, Dash, Flask, or any web framework.\n"
            "⚠️ DO NOT return server code or 'Dashboard launched' messages.\n"
            "✅ The script must run end-to-end without user edits."
        )
    else:
        # Default notebook mode
        system_message += (
            " ✅ Generate general-purpose notebook-style code only. "
            "⚠️ DO NOT use Streamlit, Dash, Gradio, Flask, or any server-based frameworks in this mode. "
            "⚠️ DO NOT output text like 'Dashboard launched...'. "
            "Always split the solution into multiple cells with markers like `# Cell 1`, `# Cell 2`, etc. "
            "Each cell should only perform one logical step:\n"
            "- Cell 1: import libraries\n"
            "- Cell 2: load dataset\n"
            "- Cell 3+: analysis, visualization, or ML logic\n"
            "If user asks to 'show rows', use `print(df.head(N))`. "
            "Only generate pure notebook-style outputs."
        )

    # Construct user-facing prompt
    full_prompt = (
        f"Dataset is located at: {dataset_url}\n"
        f"Language: {language}\n"
        f"User Request: {prompt}\n"
        "Make sure to:\n"
        "- Always load the dataset before analysis.\n"
        "- Return code in separated cells with comments (# Cell 1, # Cell 2, ...).\n"
        "- Avoid unnecessary explanations; return only code.\n"
    )

    # Call Azure OpenAI
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": full_prompt},
        ],
        temperature=0.2,
        max_tokens=2000,
    )

    raw_output = response.choices[0].message.content.strip()
    return clean_code_output(raw_output)
