# main.py
from fastapi import FastAPI
from dotenv import load_dotenv
import logging
import os
import traceback

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ Set up basic logging
logging.basicConfig(level=logging.INFO)

# ✅ Create the FastAPI app
app = FastAPI(
    title="SifraAI Backend",
    version="1.0.0",
    description="Backend API for SifraAI platform."
)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Welcome to SifraAI Backend!"}

# ✅ Mount all routes
try:
    from routes import upload, generate_code, run_code

    app.include_router(upload.router, tags=["Upload"])
    app.include_router(generate_code.router, tags=["Generate Code"])
    app.include_router(run_code.router, tags=["Run Code"])

    logging.info("✅ All routes mounted successfully!")

except Exception as e:
    logging.error(f"❌ Error loading routers: {str(e)}")
    traceback.print_exc()
