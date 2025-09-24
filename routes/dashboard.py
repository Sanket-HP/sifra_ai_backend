# ✅ FILE: routes/dashboard.py
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
import os, tempfile, subprocess, uuid, signal, re
from models.schemas import GenerateCodeInput
from utils.azureopenai_api import generate_code_with_output

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

# Store running dashboards
running_dashboards = {}


@router.post("/generate")
async def generate_dashboard(input_data: GenerateCodeInput):
    """
    1. Generate dashboard code via LLM.
    2. Clean any markdown fences.
    3. Save to temp file.
    4. Launch as background process.
    5. Return dashboard URL + PID.
    """
    # Step 1: Generate dashboard code
    code = generate_code_with_output(
        input_data.prompt,
        input_data.language,
        input_data.dataset_url,
        mode="dashboard"
    )

    # Step 2: Extra cleanup to remove stray markdown fences
    code = re.sub(r"```[a-zA-Z]*", "", code)  # remove ```python or ```py
    code = code.replace("```", "").strip()

    # Step 3: Save to file
    app_id = str(uuid.uuid4())[:8]
    dashboard_file = os.path.join(tempfile.gettempdir(), f"dashboard_{app_id}.py")
    with open(dashboard_file, "w", encoding="utf-8") as f:
        f.write(code)

    # Step 4: Launch process on random port
    port = 8500 + (hash(app_id) % 1000)
    process = subprocess.Popen(
        [
            "streamlit", "run", dashboard_file,
            "--server.port", str(port),
            "--server.address", "0.0.0.0"
        ],
        env={**os.environ, "PORT": str(port)},  # Pass port to Streamlit
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    running_dashboards[process.pid] = {
        "file": dashboard_file,
        "port": port
    }

    return JSONResponse({
        "message": "✅ Dashboard launched",
        "dashboard_url": f"http://localhost:{port}",
        "process_id": process.pid,
        "file": dashboard_file,
        "generated_code": code
    })


@router.post("/stop")
async def stop_dashboard(pid: int = Query(..., description="Process ID of the dashboard to stop")):
    """
    Stop a running dashboard by PID.
    Example: /dashboard/stop?pid=1234
    """
    if pid not in running_dashboards:
        return JSONResponse({"error": f"No process found with PID {pid}"}, status_code=404)

    try:
        os.kill(pid, signal.SIGTERM)  # Graceful stop
        info = running_dashboards.pop(pid, None)
        return JSONResponse({
            "message": f"🛑 Dashboard stopped (PID {pid})",
            "info": info
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
