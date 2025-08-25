# services/code_executor.py
from __future__ import annotations

import base64
import contextlib
import io
import traceback
import subprocess
import sqlite3
from typing import Any, Dict, List, Optional

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Optional: Plotly capture
try:
    import plotly.graph_objects as go  # type: ignore
    import plotly.io as pio           # type: ignore
    _PLOTLY_AVAILABLE = True
except Exception:
    _PLOTLY_AVAILABLE = False


# -----------------------------
# Visualization capture helpers
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
            if isinstance(v, go.Figure):  # type: ignore
                img_bytes = pio.to_image(v, format="png")  # requires kaleido
                images.append(base64.b64encode(img_bytes).decode("utf-8"))
        except Exception:
            pass
    return images


# -----------------------------
# Core Executor
# -----------------------------
def execute_code_blocks(
    code: str,
    language: str = "python",
    dataset_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Execute code cell-wise. Supports Python, R, and SQL.
    Returns a list of results with stdout, errors, and visualizations.
    """
    blocks = [b for b in code.split("\n\n") if b.strip()]
    results: List[Dict[str, Any]] = []

    if language.lower() == "python":
        exec_globals: Dict[str, Any] = {}
        if dataset_url:
            # Inject dataset_url into Python context
            exec_globals["DATASET_URL"] = dataset_url

        for block in blocks:
            stdout_buffer = io.StringIO()
            error_text = None
            images: List[str] = []

            plt.close("all")

            try:
                with contextlib.redirect_stdout(stdout_buffer):
                    exec(block, exec_globals)  # nosec

                images.extend(_capture_matplotlib_images())
                images.extend(_capture_plotly_figures(exec_globals))

            except Exception:
                error_text = traceback.format_exc()
                try:
                    images.extend(_capture_matplotlib_images())
                    images.extend(_capture_plotly_figures(exec_globals))
                except Exception:
                    pass

            results.append(
                {
                    "input": block,
                    "output": stdout_buffer.getvalue(),
                    "error": error_text,
                    "visualizations": images,
                }
            )

    elif language.lower() == "r":
        for block in blocks:
            stdout_buffer = io.StringIO()
            error_text = None
            output_text = ""

            try:
                # Run R code via subprocess
                proc = subprocess.run(
                    ["Rscript", "-e", block],
                    capture_output=True,
                    text=True
                )
                output_text = proc.stdout
                if proc.stderr:
                    error_text = proc.stderr
            except Exception:
                error_text = traceback.format_exc()

            results.append(
                {
                    "input": block,
                    "output": output_text,
                    "error": error_text,
                    "visualizations": [],  # TODO: support R plots (png capture)
                }
            )

    elif language.lower() == "sql":
        # Use in-memory SQLite for MVP
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()

        for block in blocks:
            output_text = ""
            error_text = None
            try:
                cursor.execute(block)
                rows = cursor.fetchall()
                col_names = [desc[0] for desc in cursor.description] if cursor.description else []
                if rows:
                    output_text = str([dict(zip(col_names, row)) for row in rows])
                else:
                    output_text = "Query executed successfully."
            except Exception:
                error_text = traceback.format_exc()

            results.append(
                {
                    "input": block,
                    "output": output_text,
                    "error": error_text,
                    "visualizations": [],
                }
            )

        conn.close()

    else:
        results.append(
            {
                "input": code,
                "output": "",
                "error": f"Unsupported language: {language}",
                "visualizations": [],
            }
        )

    return results
