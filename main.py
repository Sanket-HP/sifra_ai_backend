from fastapi import FastAPI
from routes import generate_code, upload, run_code

app = FastAPI(
    title="SifraAI Backend",
    version="1.0.0",
    description="Backend API for SifraAI platform.",
)

@app.get("/")
def root():
    return {"message": "Welcome to SifraAI Backend"}

@app.get("/health")
def health():
    return {"status": "healthy"}

app.include_router(upload.router, tags=["Upload"])
app.include_router(generate_code.router, tags=["Code"])
app.include_router(run_code.router, tags=["Run"])
