import openai
import os

openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_type = "azure"
openai.api_version = "2025-01-01-preview"

deployment_id = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")

def generate_code(prompt, language):
    response = openai.ChatCompletion.create(
        deployment_id=deployment_id,
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"You are an expert {language} data scientist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=800
    )
    return response["choices"][0]["message"]["content"]
