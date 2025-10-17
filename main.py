from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # <-- ADD THIS IMPORT
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

# --- ADD THIS CORS MIDDLEWARE SECTION ---
# This allows your frontend to communicate with your backend.
# The "*" allows all origins, which is fine for development.
# For production, you should restrict this to your actual frontend domain.
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# ----------------------------------------

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
