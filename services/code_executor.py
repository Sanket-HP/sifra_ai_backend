from __future__ import annotations

import base64
import contextlib
import io
import traceback
import subprocess
import sqlite3
import tempfile
import uuid
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

# Headless plotting
import matplotlib
matplotlib.use("Agg")  # ensures plots render in headless envs
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Plotly
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    _PLOTLY_AVAILABLE = True
except Exception:
    _PLOTLY_AVAILABLE = False


# -----------------------------
# Visualization capture
# -----------------------------
def _capture_matplotlib_images() -> List[str]:
    images: List[str] = []
    try:
        for num in plt.get_fignums():
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            images.append(base64.b64encode(buf.read()).decode("utf-8"))
    finally:
        plt.close("all")
    return images


def _capture_plotly_figures(globs: Dict[str, Any]) -> List[str]:
    if not _PLOTLY_AVAILABLE:
        return []
    images: List[str] = []
    for v in globs.values():
        try:
            if isinstance(v, go.Figure):
                img_bytes = pio.to_image(v, format="png")
                images.append(base64.b64encode(img_bytes).decode("utf-8"))
        except Exception:
            pass
    return images


# -----------------------------
# Dataset loader
# -----------------------------
def _load_dataset(dataset_url: str) -> Optional[pd.DataFrame]:
    if not dataset_url:
        return None
    try:
        if dataset_url.endswith(".csv"):
            return pd.read_csv(dataset_url)
        elif dataset_url.endswith(".json"):
            return pd.read_json(dataset_url)
        elif dataset_url.endswith(".xlsx"):
            return pd.read_excel(dataset_url)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_url}")
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")


# -----------------------------
# Dashboard helpers
# -----------------------------
def _is_interactive_dashboard_code(code: str) -> bool:
    lowered = code.lower()
    keywords = [
        "dash.dash(", "app.run_server(", "from dash", "import dash",
        "streamlit", "st.", "streamlit.", "app.run("
    ]
    return any(k in lowered for k in keywords)


def _looks_like_preview_code(code: str) -> bool:
    lowered = code.lower()
    return any(
        x in lowered for x in ["df.head", "df.tail", "df.sample", "print(df.head", "show rows"]
    )


def _launch_interactive_app(code: str) -> Dict[str, Any]:
    app_id = uuid.uuid4().hex[:8]
    dashboard_file = os.path.join(tempfile.gettempdir(), f"dashboard_{app_id}.py")

    with open(dashboard_file, "w", encoding="utf-8") as f:
        f.write(code)

    port = 8500 + (hash(app_id) % 1000)
    env = {**os.environ, "PORT": str(port)}

    log_file = os.path.join(tempfile.gettempdir(), f"dashboard_{app_id}.log")
    log_f = open(log_file, "ab")

    process = subprocess.Popen(
        ["python", dashboard_file],
        env=env,
        stdout=log_f,
        stderr=log_f,
        close_fds=True
    )

    return {
        "input": code,
        "output": f"Dashboard launched at http://localhost:{port}",
        "error": None,
        "visualizations": [],
        "dashboard": {"url": f"http://localhost:{port}", "pid": process.pid},
    }


# -----------------------------
# ML pipeline detection
# -----------------------------
def _is_ml_pipeline(code: str) -> bool:
    """Detect ML workflow code (sklearn)."""
    keywords = [
        "LabelEncoder", "DecisionTreeClassifier", "RandomForestClassifier",
        "LogisticRegression", "train_test_split", "accuracy_score", "fit(", "predict("
    ]
    return any(k in code for k in keywords)


# -----------------------------
# Core executor
# -----------------------------
def execute_code_blocks(
    code: str,
    language: str = "python",
    dataset_url: Optional[str] = None,
    dataset_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    # Dashboard
    if language.lower() == "python" and _is_interactive_dashboard_code(code) and not _looks_like_preview_code(code):
        try:
            results.append(_launch_interactive_app(code))
            return results
        except Exception as e:
            return [{"input": code, "output": "", "error": f"Failed to launch dashboard: {e}", "visualizations": []}]

    # Block splitting
    if _is_ml_pipeline(code):
        blocks = [code]  # run ML pipeline in ONE block
    else:
        blocks = re.split(r"(?m)^#\s*Cell.*$", code)
        if len(blocks) <= 1:
            blocks = [b for b in code.split("\n\n") if b.strip()]
        else:
            blocks = [b.strip() for b in blocks if b.strip()]

    # Python execution
    if language.lower() == "python":
        exec_globals: Dict[str, Any] = {
            "pd": pd,
            "np": __import__("numpy"),
            "sns": sns,
            "plt": plt,
        }
        try:
            from sklearn.preprocessing import LabelEncoder
            from sklearn.model_selection import train_test_split
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.metrics import accuracy_score

            exec_globals.update({
                "LabelEncoder": LabelEncoder,
                "train_test_split": train_test_split,
                "DecisionTreeClassifier": DecisionTreeClassifier,
                "accuracy_score": accuracy_score,
            })
        except Exception:
            pass

        if dataset_url:
            try:
                df = _load_dataset(dataset_url)
                if df is not None:
                    exec_globals["df"] = df
                    exec_globals["DATASET"] = df
            except Exception as e:
                results.append({"input": "", "output": "", "error": str(e), "visualizations": []})

        for block in blocks:
            stdout_buffer = io.StringIO()
            error_text, output_text = None, ""
            images: List[str] = []
            plt.close("all")
            try:
                with contextlib.redirect_stdout(stdout_buffer):
                    exec(block, exec_globals)
                images.extend(_capture_matplotlib_images())
                images.extend(_capture_plotly_figures(exec_globals))
                output_text = stdout_buffer.getvalue().strip()
                if not output_text:
                    if "df.head" in block:
                        output_text = str(exec_globals.get("df", pd.DataFrame()).head(10))
                    elif "df.tail" in block:
                        output_text = str(exec_globals.get("df", pd.DataFrame()).tail(10))
                    elif "df.sample" in block:
                        output_text = str(exec_globals.get("df", pd.DataFrame()).sample(5))
                    elif "import " in block:
                        output_text = "Libraries imported successfully."
                    elif any(cmd in block for cmd in ["read_csv", "read_excel", "read_json"]):
                        output_text = "Dataset loaded successfully."
                    else:
                        output_text = "Code executed successfully."
            except Exception:
                error_text = traceback.format_exc()
            results.append({"input": block, "output": output_text, "error": error_text, "visualizations": images})

    # R execution
    elif language.lower() == "r":
        for block in blocks:
            try:
                proc = subprocess.run(["Rscript", "-e", block], capture_output=True, text=True)
                results.append({
                    "input": block,
                    "output": proc.stdout.strip(),
                    "error": proc.stderr if proc.stderr else None,
                    "visualizations": [],
                })
            except Exception:
                results.append({"input": block, "output": "", "error": traceback.format_exc(), "visualizations": []})

    # SQL execution
    elif language.lower() == "sql":
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        if dataset_url:
            try:
                df = _load_dataset(dataset_url)
                if df is not None:
                    df.to_sql("dataset", conn, if_exists="replace", index=False)
            except Exception as e:
                results.append({"input": "", "output": "", "error": str(e), "visualizations": []})
        for block in blocks:
            try:
                cursor.execute(block)
                rows = cursor.fetchall()
                col_names = [d[0] for d in cursor.description] if cursor.description else []
                output_text = str([dict(zip(col_names, row)) for row in rows]) if rows else "Query executed successfully."
                results.append({"input": block, "output": output_text, "error": None, "visualizations": []})
            except Exception:
                results.append({"input": block, "output": "", "error": traceback.format_exc(), "visualizations": []})
        conn.close()

    else:
        results.append({"input": code, "output": "", "error": f"Unsupported language: {language}", "visualizations": []})

    return results
