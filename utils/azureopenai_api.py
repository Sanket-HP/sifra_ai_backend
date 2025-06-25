# utils/azureopenai_api.py

import os
import openai
from typing import Optional

# Load environment variables
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_type = "azure"
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION") 

def generate_code_with_output(prompt: str, language: str, dataset_url: str = None) -> str:
   
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") 
    
    if not deployment:
        raise ValueError("Missing AZURE_OPENAI_DEPLOYMENT_NAME environment variable.")
    if not openai.api_key:
        raise ValueError("Missing AZURE_OPENAI_API_KEY environment variable.")
    if not openai.api_base or not openai.api_version:
        raise ValueError("Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_VERSION environment variables.")

    try:
        response = openai.ChatCompletion.create(
            engine=deployment,
            messages=[
                {"role": "system", "content": f"You are a helpful assistant that writes {language} code."},
                {"role": "user", "content": f"Prompt: {prompt}\nDataset URL (if any): {dataset_url or 'None'}"}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"# Code generation failed: {str(e)}"
