from azure.ai.openai import OpenAIClient
from azure.core.credentials import AzureKeyCredential
import os

def generate_code(prompt: str, language: str) -> str:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not all([endpoint, deployment_id, api_key, api_version]):
        raise ValueError("Azure environment variables are missing.")

    client = OpenAIClient(endpoint=endpoint, credential=AzureKeyCredential(api_key), api_version=api_version)

    response = client.chat.completions.create(
        deployment_id=deployment_id,
        messages=[
            {"role": "system", "content": "You are an expert Python data analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content
