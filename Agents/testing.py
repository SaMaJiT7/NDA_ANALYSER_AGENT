from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

_deepseek_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Usage
response = _deepseek_client.chat.completions.create(
    model="qwen/qwen3-32b",
    messages=[
        {
            "role": "user",
            "content": """Analyze the following NDA clause under Section 27 of the Indian Contract Act: 
"The Employee shall not, for a period of 3 years after termination, engage in any business that competes with the Company within the territory of India."

Provide:
1. Risk Level
2. Violated Statute
3. Explanation of why this is void or voidable."""
        }
    ],
    max_tokens=1024,
)

ai_msg = response.choices[0].message
print(f"Verified Reasoning: {getattr(ai_msg, 'reasoning_content', 'No reasoning found.')}")
print(f"Response: {ai_msg.content}")