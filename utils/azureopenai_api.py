import os
from openai import AzureOpenAI

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),  # e.g. "2024-05-01-preview"
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")   # e.g. https://your-endpoint.openai.azure.com/
)

def generate_code_with_output(prompt: str, language: str, dataset_url: str) -> str:
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not deployment:
        raise ValueError("Missing AZURE_OPENAI_DEPLOYMENT_NAME")

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": f"You are a helpful assistant that writes {language} code using the dataset at {dataset_url}."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1500
    )

    return response.choices[0].message.content.strip()
