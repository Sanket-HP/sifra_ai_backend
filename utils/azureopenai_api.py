import os
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.ai.openai import OpenAIClient

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def generate_code_with_output(prompt: str, language: str, dataset_url: str) -> str:
    client = OpenAIClient(
        endpoint=api_base,
        credential=DefaultAzureCredential()
    )

    system_prompt = f"""
You are a {language} data science assistant.
Generate CLEAN {language} code only.
DO NOT return markdown (no ```), no explanations.
Only valid code.
The dataset is available at: {dataset_url}
"""

    user_prompt = f"{prompt}\nReturn only runnable {language} code."

    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=1000,
    )

    return response.choices[0].message.content.strip()
