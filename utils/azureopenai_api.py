import os
from openai import AzureOpenAI

def generate_code(prompt: str, language: str) -> str:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not all([endpoint, deployment_id, api_key, api_version]):
        raise ValueError("Azure environment variables (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION) are missing. Please set them in your Azure App Settings.")

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
    )

    response = client.chat.completions.create(
        model=deployment_id,
        messages=[
            {"role": "system", "content": f"You are an expert {language} data analyst. Generate {language} code only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content