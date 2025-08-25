from openai import OpenAI
import os
import base64
from PIL import Image
from io import BytesIO
import google.generativeai as genai
import re
from tqdm import tqdm
from cantor_prompt import handle_image_prompt, decision_stage_prompt, get_result_prompt

openai_key = os.getenv("OPENAI_API_KEY")
kimi_key = os.getenv("Kimi_key")
deep_key = os.getenv("DEEP_KEY")

openai_client = OpenAI(
    api_key=openai_key,
)

kimi_client = OpenAI(
    api_key=kimi_key,
    base_url="https://api.moonshot.cn/v1",
)

deep_client = OpenAI(
    api_key=deep_key,
    base_url="https://api.deepseek.com/v1"
)
genai.configure(api_key=os.getenv("GOOGLE_KEY"))


# Extract text information from images
def handle_image(image, cantor_input):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = handle_image_prompt + f"""Here is question and choice questions {cantor_input}."""
    response_image = model.generate_content([prompt, image])
    return response_image.text


# Decision-making stage
def decision_stage(input_msg):
    prompt_pin = decision_stage_prompt + f"{input_msg}."
    response_pin = deep_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_pin}
        ],
        temperature=0.2,
        max_tokens=800
    )

    return response_pin.choices[0].message.content


def get_final_result(decision_prompt, question_info):
    prompt_answer = get_result_prompt + f"""   Now, here is the case:
    
    Question: {question_info}
    
    Supplementary information:
    {decision_prompt}
    
    Only choose one of the given answer choices and copy it exactly. Do not make up anything."""

    response_answer = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_answer}
        ],
        temperature=0.3,
        max_tokens=512
    )

    return response_answer.choices[0].message.content, prompt_answer
