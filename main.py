from fastapi import FastAPI
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Welcome to SifraAI Backend!"}

# ✅ Import & include routers
try:
    from routes import upload, generate_code, run_code
    app.include_router(upload.router)
    app.include_router(generate_code.router)
    app.include_router(run_code.router)
except Exception as e:
    print(f"❌ Error loading routers: {e}")
