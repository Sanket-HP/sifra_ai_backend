# utils/azureopenai_api.py
import os
import openai

openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. https://…openai.azure.com/
openai.api_type = "azure"
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")  # e.g. "2025-04-14"

def generate_code(prompt: str, language: str) -> str:
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")  # e.g. "sifra-gpt41-2"
    if not deployment:
        raise ValueError("Missing AZURE_OPENAI_DEPLOYMENT_NAME")
    resp = openai.ChatCompletion.create(
        engine=deployment,
        messages=[
            {"role": "system", "content": f"You are a helpful AI assistant who writes {language} code."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1500
    )
    return resp.choices[0].message.content.strip()
