# utils/azureopenai_api.py
import os
import requests

def generate_code(prompt: str, language: str) -> str:
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not all([api_key, endpoint, deployment, api_version]):
        raise ValueError("❌ Missing Azure OpenAI environment variables")

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    body = {
        "messages": [
            {"role": "system", "content": "You are an expert data scientist."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise Exception(f"OpenAI request failed: {response.status_code} - {response.text}")

    return response.json()["choices"][0]["message"]["content"]
