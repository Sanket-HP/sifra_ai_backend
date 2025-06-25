from fastapi import FastAPI
from routes import generate_code, upload, run_code
import logging

# Optional: Add startup log (for Azure log stream)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("🚀 SifraAI Backend starting...")

app = FastAPI(
    title="SifraAI Backend",
    version="1.0.0",
    description="Backend API for SifraAI platform.",
)

@app.get("/", tags=["Root"])
def root():
    return {"message": "Welcome to SifraAI Backend"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}

# Attach routes
app.include_router(upload.router, prefix="", tags=["Upload"])
app.include_router(generate_code.router, prefix="", tags=["Code"])
app.include_router(run_code.router, prefix="", tags=["Run"])
