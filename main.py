from fastapi import FastAPI
from routes import generate_code, upload, run_code, dashboard, report
import logging

# Azure log stream friendly logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("🚀 SifraAI Backend starting...")

# Initialize FastAPI app
app = FastAPI(
    title="SifraAI Backend",
    version="1.0.0",
    description="Backend API for the SifraAI platform.",
)

# Root route
@app.get("/", tags=["Root"])
def root():
    return {"message": "Welcome to SifraAI Backend"}

# Health check route (used by Azure for status monitoring)
@app.get("/health", status_code=200, tags=["Health"])
def health():
    return {"status": "healthy"}

# Include route modules
app.include_router(upload.router, prefix="", tags=["Upload"])
app.include_router(generate_code.router, prefix="", tags=["Code"])
app.include_router(run_code.router, prefix="", tags=["Run"])
app.include_router(dashboard.router)
app.include_router(report.router)
