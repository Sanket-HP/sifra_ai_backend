# main.py
from fastapi import FastAPI
from dotenv import load_dotenv
import os
import logging

# Load .env
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)

# Create app
app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Welcome to SifraAI Backend!"}

# Mount routes
try:
    from routes import upload, generate_code, run_code
    app.include_router(upload.router)
    app.include_router(generate_code.router)
    app.include_router(run_code.router)
    print("✅ All routes mounted successfully!")
except Exception as e:
    import traceback
    print("❌ Error mounting routes:", str(e))
    traceback.print_exc()
