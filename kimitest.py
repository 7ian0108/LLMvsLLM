import openai
import os

api_key = os.getenv("DEEP_KEY")

client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1"
)

completion = client.chat.completions.create(
    model="deepseek-chat",  # deepseek-chat, deepseek-coder, deepseek-math
    messages=[
        {
            "role": "system",
            "content": "You are DeepSeek, a powerful multilingual assistant. Provide clear, precise, and helpful answers to user queries."
        },
        {
            "role": "user",
            "content": "Say hello in Chinese."
        }
    ],
    temperature=0.3
)

print(completion.choices[0].message.content)
