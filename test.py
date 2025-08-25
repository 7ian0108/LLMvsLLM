from IPython.core.debugger import prompt
from datasets import load_dataset
from pycparser.ply.yacc import token

from MLLMvMLLM_prompt import question_prompt
from cantor import cantor
from openai import OpenAI
from cantor_prompt import get_result_prompt
import os
from IPython.display import display
from collections import defaultdict
from MLLMvMLLM_function import (extract_entities_from_rationale,
                                generate_global_verification_questions,
                                answer_with_context,
                                kimi_evaluate_answer,
                                get_reasoning_score_from_answer,
                                answer_correction)
openai_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(
    api_key=openai_key,
)

data = load_dataset('derek-thomas/ScienceQA', split='test')
sample = data[4]
categories = ["NAT", "SOC", "LAN", "TXT", "IMG", "NO", "G1-6", "G7-12"]
correct_per_category = defaultdict(int)
total_per_category = defaultdict(int)

def get_category_tags(sample):
    tags = []
    if sample["subject"] == "natural science":
        tags.append("NAT")
    elif sample["subject"] == "social science":
        tags.append("SOC")
    elif sample["subject"] == "language science":
        tags.append("LAN")

    if sample["hint"]:
        tags.append("TXT")
    elif sample["image"]:
        tags.append("IMG")
    else:
        tags.append("NO")

    grade = sample["grade"]
    if grade.startswith("grade1") or grade.startswith("grade2") or \
       grade.startswith("grade3") or grade.startswith("grade4") or \
       grade.startswith("grade5") or grade.startswith("grade6"):
        tags.append("G1-6")
    else:
        tags.append("G7-12")
    return tags

def regenerate_corrected_answer(correction_prompt, o_prompt):
    """
    Send the constructed correction prompt to the GPT model to generate the corrected answer.

    Parameters:
    correction_prompt: The prompt to be passed to the model (a string)
    model_name: The name of the model to be called (e.g., gpt-3.5)

    Returns:
    Corrected answer (a string)
    """
    prompt = correction_prompt + o_prompt + f"Your first answer {first_answer} is wrong, please do it again follow the comment."

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
        {"role": "system", "content": "You are an expert scientific reasoning assistant. Based on the user's review prompt, generate a corrected final answer."},
        {"role": "user", "content": prompt}
    ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()

# # stage01：cantor
# res, cantor_input, first_prompt = cantor(image=None, sample=sample)
# predicted_text = res.split("Answer:")[1].strip().split("\n")[0].strip().rstrip(".").lower()
# print(predicted_text)
#
#
# name_entities = extract_entities_from_rationale(rationale_text=res)
#
#
# question = generate_global_verification_questions(name_entities, cantor_input, res)
#
# print(question)

# results = {}
#
# for q in question:
#     gpt_answer = answer_with_context(q, cantor_input, res, predicted_text)
#     kimi_comment = kimi_evaluate_answer(q, gpt_answer)
#     score = get_reasoning_score_from_answer(kimi_comment)
#
#     results[q] = {
#         "answer": gpt_answer,
#         "comment": kimi_comment,
#         "score": score
#     }
#
# print(results)
#
# command = answer_correction(command=results)
#
# print(command)
#
# corrected_answer = regenerate_corrected_answer(command, first_prompt)
#
# print(corrected_answer)
correct_num = 0
for i in range(0, len(data)):
    sample = data[i]
    tags = get_category_tags(sample)
    chain, cantor_input, first_prompt = '', '', ''
    # stage01：cantor
    if sample["image"]:
        chain, cantor_input, first_prompt = cantor(image=sample["image"], sample=sample)
    else:
        chain, cantor_input, first_prompt = cantor(image=None, sample=sample)
    first_answer = chain.split("Answer:")[1].strip().split("\n")[0].strip().rstrip(".").lower()

    # Extracting named entities
    name_entities = extract_entities_from_rationale(rationale_text=chain)

    # Conduct MLLM confrontation
    question = generate_global_verification_questions(name_entities, cantor_input, chain)

    results = {}

    for q in question:
        gpt_answer = answer_with_context(q, cantor_input, chain, first_answer)
        kimi_comment = kimi_evaluate_answer(q, gpt_answer, cantor_input, first_answer)
        score = get_reasoning_score_from_answer(kimi_comment)

        results[q] = {
            "answer": gpt_answer,
            "comment": kimi_comment,
            "score": score
        }

    command = answer_correction(command=results)

    corrected_answer = regenerate_corrected_answer(command, first_prompt)
    final_answer = corrected_answer.split("Answer:")[1].strip().split("\n")[0].strip().strip("'").rstrip(".").lower()
    answer_index = sample['answer']
    ture_answer = sample['choices'][answer_index].rstrip(".").lower()


    if final_answer == ture_answer:
        correct_num += 1
        for tag in tags:
            correct_per_category[tag] += 1

    for tag in tags:
        total_per_category[tag] += 1
    print(f"question describe: {data[i]}")
    print(f"name_entities {name_entities}")
    print(f"questions {question}")
    print(f"command {results}")
    print(f"first_answer" )
    print(f"pred_answer: {final_answer}")
    print(f"ture_answer: {ture_answer}")
    print(f"current correct number is: {correct_num}")
    print(f"current process is: {i+1}/{len(data)}")
    print("----------------------------------------")
    break



acc = correct_num / len(data)
print(acc)
for tag in categories:
    acc = correct_per_category[tag] / total_per_category[tag] if total_per_category[tag] > 0 else 0
    print(f"{tag} Accuracy: {acc:.4f} ({correct_per_category[tag]}/{total_per_category[tag]})")
