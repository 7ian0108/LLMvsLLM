from openai import OpenAI
import os
import ast
import time
import requests
import base64
from PIL import Image
from io import BytesIO
import google.generativeai as genai
import re
from tqdm import tqdm
from MLLMvMLLM_prompt import extract_prompt, question_prompt, support_prompt

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

#  Helper Functions
def clean_and_parse_list_string(s: str):
    cleaned = re.sub(r"^```json|```$", "", s.strip(), flags=re.IGNORECASE)
    return ast.literal_eval(cleaned)

# Extracting named entities
def extract_entities_from_rationale(rationale_text, model="gpt-3.5-turbo"):
    prompt = extract_prompt + rationale_text

    response = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# MLLM对抗
def generate_global_verification_questions(name_entities, cantor_input, reasoning_chain):
    """
    Generate a set of validation questions for multiple named entities
    """
    prompt = f"""You are a reasoning verification assistant.
    Given the following information:
    - Original user question and Visual-textual context: "{cantor_input}"
    - Reasoning chain: "{reasoning_chain}"
    - Named entities extracted from the reasoning: 
    {name_entities}
    Please generate 3–5 verification questions that help assess whether these named entities ({name_entities}) are semantically correct and consistent with the reasoning and input. Each question should help test the validity of one or more entities.
    """ + question_prompt

    response = kimi_client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return clean_and_parse_list_string(response.choices[0].message.content)


def answer_with_context(question, cantor_input, reasoning_chain, answer):
    """
    GPT-3.5 answers questions based on the image and text input and the first round of reasoning chain
    """
    prompt = f"""
    You are an expert reasoning assistant.

    Here is the original question and visual-textual context:
    "{cantor_input}"
    
    Initial reasoning chain:
    "{reasoning_chain}"
    
    Final answer from the initial reasoning:
    "{answer}"
    
    Now consider the following verification question:
    "{question}"
    
    Please answer this verification question carefully and thoroughly, using the full context provided above.
    
    Avoid vague or incomplete answers. Do not just restate the original answer—justify it with reasoning.
    """

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

def kimi_evaluate_answer(question, answer, cantor_input, first_answer):
    """
    Let Kimi comment and rate GPT-3.5's answers (using keywords to guide the evaluation)
    """
    prompt = support_prompt + f"""\nNow judge the following:
    Original question: {cantor_input}
    first original question answer: {first_answer}
    Question: {question}  
    Answer: {answer}
    """
    response = deep_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

def get_reasoning_score_from_answer(answer):
    """
    Map natural language responses to plausibility scores [0, 1]
    """
    answer = answer.lower()
    if any(k in answer for k in ["inappropriate", "incorrect", "wrong", "conflict", "replace"]):
        return 0.2
    elif any(k in answer for k in ["unclear", "ambiguous", "maybe", "not sure", "consider"]):
        return 0.5
    elif any(k in answer for k in ["appropriate", "correct", "reasonable", "yes", "accurate"]):
        return 0.9
    else:
        return 0.6

def answer_correction(command, threshold=0.6):
    """
    Based on each question's score, determine whether the inference conclusion needs to be revised.

    Parameters:
    command: The dictionary output from the second stage {question: {'answer':..., 'comment':..., 'score':...}}
    threshold: If the score is less than the threshold, the question is considered unreasonable and needs to be revised.

    Returns:
    Contains whether the inference conclusion needs to be revised, a list of low-scoring questions, and the constructed new prompt word.
    """
    low_conf_qas = []
    for question, qa in command.items():
        score = qa.get("score", 1.0)
        if score < threshold:
            low_conf_qas.append((question, qa["answer"], qa["comment"]))

    if not low_conf_qas:
        return ''

    # Tips for constructing corrections
    prompt_parts = [f"Question: {q}\nAnswer: {a}\nComment: {c}" for q, a, c in low_conf_qas]
    full_prompt = (
            "Some of the sub-questions and answers in the original reasoning were found to be unreliable.\n\n"
            "Please read the following list of verification questions, their original answers, and expert review comments:\n\n"
            + "\n\n".join(prompt_parts) + "\n\n"
                                          "Your task is to do the following:\n"
                                          "1. Identify the specific mistakes or flaws mentioned in the comments.\n"
                                          "2. Use the valid sub-answers, discard the unreliable ones, and revise the final answer accordingly.\n"
                                          "3. If needed, re-derive part of the reasoning based on the original context and the corrected logic.\n\n"
                                          "Finally, provide a corrected final answer. Be sure your new answer fixes the issues raised and aligns with the question's context."
    )

    return full_prompt










