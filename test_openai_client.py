from openai import OpenAI

client = OpenAI(base_url="http://134.199.195.3:8000/v1", api_key="EMPTY")
r = client.chat.completions.create(
    model="/models/DeepSeek-R1-Distill-Qwen-32B",
    messages=[{"role": "user", "content": "Say hello in one word"}],
    max_tokens=10,
)
print("OK:", r.choices[0].message.content)
