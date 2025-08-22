# services/code_executor.py
from __future__ import annotations

import base64
import contextlib
import io
import sys
import traceback
from typing import Any, Dict, List

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Optional: capture Plotly figures if your generated code uses Plotly
try:
    import plotly.graph_objects as go  # type: ignore
    import plotly.io as pio           # type: ignore
    _PLOTLY_AVAILABLE = True
except Exception:
    _PLOTLY_AVAILABLE = False


def _capture_matplotlib_images() -> List[str]:
    """Return a list of PNG images (base64) for any open Matplotlib figures."""
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
    """Scan globals for Plotly figures and capture them as PNG (base64)."""
    if not _PLOTLY_AVAILABLE:
        return []
    images: List[str] = []
    for v in globs.values():
        try:
            if isinstance(v, go.Figure):  # type: ignore
                img_bytes = pio.to_image(v, format="png")  # requires kaleido
                images.append(base64.b64encode(img_bytes).decode("utf-8"))
        except Exception:
            # If kaleido is missing or to_image fails, just skip
            pass
    return images


def execute_code_blocks(code: str) -> List[Dict[str, Any]]:
    """
    Split code into simple blocks and execute them sequentially,
    capturing stdout and any generated visualizations.
    """
    # Naive split: double newlines separate "blocks"
    blocks = [b for b in code.split("\n\n") if b.strip()]

    exec_globals: Dict[str, Any] = {}
    results: List[Dict[str, Any]] = []

    for block in blocks:
        stdout_buffer = io.StringIO()
        error_text = None
        images: List[str] = []

        # Reset figures for this block to avoid leaking cross-block plots
        plt.close("all")

        try:
            with contextlib.redirect_stdout(stdout_buffer):
                exec(block, exec_globals)  # nosec - trusted environment assumed

            # Capture any Matplotlib figures
            images.extend(_capture_matplotlib_images())

            # Capture Plotly figures if present
            images.extend(_capture_plotly_figures(exec_globals))

        except Exception:
            error_text = traceback.format_exc()
            # Even on error, try to capture figures that may have been produced
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
                # One or more images as base64 PNGs
                "visualizations": images,
            }
        )

    return results
