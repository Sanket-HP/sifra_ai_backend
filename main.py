from fastapi import FastAPI
from dotenv import load_dotenv
import logging

# Load .env
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="SifraAI Backend", version="1.0.0", description="Backend API for SifraAI platform.")

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Welcome to SifraAI Backend!"}

# Mount routers
try:
    from routes import upload, generate_code, run_code
    app.include_router(upload.router, tags=["Upload"])
    app.include_router(generate_code.router, tags=["Code"])
    app.include_router(run_code.router, tags=["Run"])
except Exception as e:
    print(f"❌ Error loading routers: {e}")
