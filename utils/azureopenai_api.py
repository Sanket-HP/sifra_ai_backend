from azure.ai.openai import OpenAIClient
from azure.identity import DefaultAzureCredential
import os

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_base = os.getenv("AZURE_OPENAI_ENDPOINT")

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
