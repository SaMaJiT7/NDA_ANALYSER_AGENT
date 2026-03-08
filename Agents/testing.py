from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
load_dotenv()

client = InferenceClient(
    model = "Qwen/Qwen3-VL-235B-A22B-Instruct",
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

response = client.chat_completion(
    messages   = [{"role": "user", "content": "Return this JSON: {\"status\": \"ok\"}"}],
    max_tokens = 100
)
print(response.choices[0].message.content)