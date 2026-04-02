"""
chatbot.py
CardioAI Health Chatbot — powered by Groq (FREE API)
Model : llama-3.3-70b-versatile
Deps  : requests  (pip install requests)

Get your FREE Groq API key at https://console.groq.com
No credit card required.
"""

import os
import requests
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are CardioAI, a professional medical AI assistant
specialising in cardiovascular health and heart disease prevention.

You are embedded in a clinical decision-support platform trained on
70,000 patient records using Gradient Boosting (99.31% accuracy).

Top risk factors by ML importance:
  1. Age                14.3%
  2. Cold Sweats/Nausea 11.6%
  3. Fatigue            11.5%
  4. Dizziness           9.6%
  5. Shortness of Breath 9.5%
  6. Pain Arms/Jaw/Back  9.4%
  7. Swelling            9.4%
  8. Chest Pain          9.0%

Rules:
- Always advise consulting a qualified doctor for personal medical decisions
- Never provide a definitive diagnosis
- Be warm, empathetic, and professional
- Keep replies concise (150-250 words)
- Use bullet points when listing items
"""

# ---------------------------------------------------------------------------

def get_chatbot_response(
    user_message: str,
    history: Optional[List[Dict]] = None
) -> str:
    """
    Send a message to Groq and return the AI reply.

    Parameters
    ----------
    user_message : str
        The latest question from the user.
    history : list of dicts, optional
        Previous turns: [{"role": "user"|"assistant", "content": "..."}]

    Returns
    -------
    str
        The assistant's reply text.
    """
    api_key = os.environ.get("GROQ_API_KEY")

    # No API key set — return a helpful fallback message
    if not api_key:
        return (
            "GROQ_API_KEY is not set.\n\n"
            "Steps to fix:\n"
            "1. Go to https://console.groq.com (free, no credit card)\n"
            "2. Create an API Key\n"
            "3. Add it to your .env file:\n"
            "   GROQ_API_KEY=gsk_your_key_here\n\n"
            "Quick heart health tips:\n"
            "- Stop smoking\n"
            "- Exercise 150 min/week\n"
            "- Eat a Mediterranean diet\n"
            "- Keep blood pressure below 120/80\n"
            "- Sleep 7-9 hours and manage stress"
        )

    # Build message list
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."

    except requests.exceptions.HTTPError:
        status = response.status_code
        if status == 401:
            return "Invalid GROQ_API_KEY. Check your key at https://console.groq.com"
        if status == 429:
            return "Rate limit reached. Wait a moment and try again (free limit: 30 req/min)."
        return f"Groq API returned error {status}. Please try again."

    except Exception as error:
        return f"Unexpected error: {str(error)}"


def explain_prediction(prediction: dict, patient_data: dict) -> str:
    """
    Generate a personalised explanation for a prediction result.

    Parameters
    ----------
    prediction   : dict with keys risk_level, risk_percent
    patient_data : dict with patient feature values
    """
    active_factors = [
        key for key, val in patient_data.items()
        if val == 1 and key not in ("Age", "Gender")
    ]

    prompt = (
        f"Patient details:\n"
        f"  Age    : {patient_data.get('Age')}\n"
        f"  Gender : {'Male' if patient_data.get('Gender') else 'Female'}\n"
        f"  Result : {prediction['risk_level']} risk ({prediction['risk_percent']}%)\n"
        f"  Active risk factors: {', '.join(active_factors) or 'None'}\n\n"
        "Please give a brief 3-4 sentence empathetic explanation of this result, "
        "then list 2-3 personalised prevention tips based on the patient's specific factors."
    )

    return get_chatbot_response(prompt)
