import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def generate_code_with_output(prompt: str, language: str, dataset_url: str) -> str:
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not deployment:
        raise ValueError("Missing AZURE_OPENAI_DEPLOYMENT_NAME")

    full_prompt = (
        f"You are a coding assistant. Generate {language} code to:\n"
        f"{prompt}\n\n"
        f"Dataset URL: {dataset_url}\n"
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": f"You are a helpful assistant that writes {language} code."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.3,
        max_tokens=1500
    )
    return response.choices[0].message.content.strip()
