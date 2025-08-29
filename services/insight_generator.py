# insight_generator.py

import json
from generate_code import generate_code  # Reuse your AI code generation function

def generate_insights(user_data: dict, prompt: str = None) -> dict:
    """
    Generates AI-driven insights from structured user data.
    
    Args:
        user_data (dict): Input data (can be analytics, subscriptions, usage logs, etc.)
        prompt (str): Optional custom prompt for tailored insights.
        
    Returns:
        dict: Insights containing summary, key findings, and recommendations.
    """
    # Convert user data to JSON for AI readability
    data_json = json.dumps(user_data, indent=2)

    # Default prompt if not provided
    if not prompt:
        prompt = f"""
        You are a Data Science Insight Generator.
        Analyze the following user data and provide insights:
        1. A short summary (2–3 lines).
        2. Key findings (bullet points).
        3. Actionable recommendations (bullet points).
        
        Data:
        {data_json}
        """

    # Call AI engine from generate_code.py
    response = generate_code(prompt)

    # Structure output (best-effort JSON parsing)
    insights = {
        "summary": "",
        "key_findings": [],
        "recommendations": []
    }

    try:
        # If model returned JSON directly
        parsed = json.loads(response)
        insights.update(parsed)
    except Exception:
        # Fallback: parse text response heuristically
        lines = response.strip().split("\n")
        section = None
        for line in lines:
            line = line.strip()
            if "summary" in line.lower():
                section = "summary"
                continue
            elif "key finding" in line.lower():
                section = "key_findings"
                continue
            elif "recommendation" in line.lower():
                section = "recommendations"
                continue
            
            if section == "summary":
                insights["summary"] += line + " "
            elif section == "key_findings":
                if line: insights["key_findings"].append(line.lstrip("-• "))
            elif section == "recommendations":
                if line: insights["recommendations"].append(line.lstrip("-• "))

    return insights


# Example usage
if __name__ == "__main__":
    sample_user_data = {
        "username": "Sanket",
        "subscriptions": ["AI Tools", "Data Science Bootcamp"],
        "activity": {
            "courses_completed": 5,
            "projects_submitted": 2,
            "daily_logins": 12
        }
    }

    insights = generate_insights(sample_user_data)
    print(json.dumps(insights, indent=2))
