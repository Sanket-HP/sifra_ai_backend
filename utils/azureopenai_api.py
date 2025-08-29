import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def generate_code_with_output(prompt: str, language: str, dataset_url: str) -> str:
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    full_prompt = (
        f"You are a helpful coding assistant.\n"
        f"Generate clean and executable {language} code only.\n"
        f"Prompt: {prompt}\nDataset URL: {dataset_url}"
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": f"You generate only clean {language} code."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.2,
        max_tokens=1500
    )

    return response.choices[0].message.content.strip()
