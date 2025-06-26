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
        f"You are a helpful coding assistant.\n"
        f"Generate clean and executable {language} code based on the following request:\n"
        f"{prompt}\n"
        f"Dataset URL: {dataset_url}\n\n"
        f"⚠️ Output ONLY valid code. No markdown (e.g., ```), no comments, no explanations, "
        f"no text like 'Sure!' or 'Make sure...'.\n"
        f"Only provide code, line-by-line. Include all required imports.\n"
        f"You are a Python coding assistant. Write only executable code in blocks "
        f"for this task:\n{prompt}\n"
        f"Dataset URL: {dataset_url}\n"
        f"Return code in clean python format only, no markdown, no explanations."
    )

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant that writes clean, executable {language} code only."
            },
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.2,
        max_tokens=1500
    )
    
    return response.choices[0].message.content.strip()
