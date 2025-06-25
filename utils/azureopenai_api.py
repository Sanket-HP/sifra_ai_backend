import os
from azure.ai.openai import OpenAIClient
from azure.identity import DefaultAzureCredential

def generate_code(prompt: str, language: str) -> str:
    # Example logic — modify based on your setup
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not all([endpoint, deployment, api_version]):
        raise ValueError("Missing Azure OpenAI environment variables")

    credential = DefaultAzureCredential()
    client = OpenAIClient(endpoint=endpoint, credential=credential)

    response = client.chat.completions.create(
        deployment_id=deployment,
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert data scientist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content
