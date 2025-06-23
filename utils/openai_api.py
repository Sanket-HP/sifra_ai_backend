import os
from azure.ai.openai import OpenAIClient
from azure.core.credentials import AzureKeyCredential

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

client = OpenAIClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))

def generate_code(prompt: str, language: str) -> str:
    try:
        response = client.get_chat_completions(
            deployment_name=deployment_name,
            messages=[
                {"role": "system", "content": f"You are a helpful assistant that writes {language} code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("🔴 Azure OpenAI error:", str(e))
        raise RuntimeError("Code generation failed.")
