# routes/run_code.py
from fastapi import APIRouter
from pydantic import BaseModel
import io
import contextlib
import traceback
import base64

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Optional Plotly capture
try:
    import plotly.graph_objects as go  # type: ignore
    import plotly.io as pio           # type: ignore
    _PLOTLY_AVAILABLE = True
except Exception:
    _PLOTLY_AVAILABLE = False

router = APIRouter()


class CodeInput(BaseModel):
    code: str


def _capture_matplotlib_images():
    images = []
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


def _capture_plotly_figures(globs):
    if not _PLOTLY_AVAILABLE:
        return []
    images = []
    for v in globs.values():
        try:
            if isinstance(v, go.Figure):  # type: ignore
                img_bytes = pio.to_image(v, format="png")  # requires kaleido
                images.append(base64.b64encode(img_bytes).decode("utf-8"))
        except Exception:
            pass
    return images


# Optional: global variable context to maintain session (careful for concurrency!)
exec_globals = {}


@router.post("/run_code")
async def run_code(data: CodeInput):
    try:
        # Reset figures for this run
        plt.close("all")

        stdout_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer):
            exec(data.code, exec_globals)  # nosec - trusted environment assumed

        images = []
        images.extend(_capture_matplotlib_images())
        images.extend(_capture_plotly_figures(exec_globals))

        return {
            "output": stdout_buffer.getvalue(),
            "error": None,
            "visualizations": images,  # <-- list of base64 PNG strings
        }

    except Exception:
        return {
            "output": "",
            "error": traceback.format_exc(),
            "visualizations": [],
        }
