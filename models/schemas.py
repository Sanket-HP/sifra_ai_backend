from pydantic import BaseModel

class GenerateCodeInput(BaseModel):
    prompt: str
    language: str
    dataset_url: str
