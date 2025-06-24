# utils/openai_api.py
import os
from openai import AzureOpenAI

# Load from environment variables
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint
)

def generate_code(prompt: str, language: str) -> str:
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": f"You are a helpful assistant who writes {language} code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("🔴 Azure OpenAI error:", str(e))
        raise RuntimeError("Code generation failed.")
