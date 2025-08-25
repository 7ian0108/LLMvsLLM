from cantor_function import handle_image, decision_stage, get_result_prompt, get_final_result

# Organize cantor's input format
def build_cantor_input(sample):
    """
    Converts a ScienceQA sample to the cantor() input format.

    Parameters:
    - sample: dict, a sample from the ScienceQA dataset
    - image: PIL.Image or None if the sample contains an image, otherwise None

    Returns:
    - image: passed directly to cantor
    - questions: dict, passed as the second argument to cantor
    """
    question_text = sample.get("question", "")
    choices = sample.get("choices", [])
    task = sample.get("task", "")
    subject = sample.get("subject", "")
    topic = sample.get("topic", "")
    grade = sample.get("grade", "")
    hint = sample.get("hint", "")
    lecture = sample.get("lecture", "")
    solution = sample.get("solution", "")

    # Construct a unified context description string (for prompt)
    context = f"""
    Task type: {task}
    Subject: {subject}, Topic: {topic}, Grade: {grade}

    Hint: {hint}
    Lecture: {lecture}
    """

    questions = f"""question: {question_text},\nchoices: {choices},\n"""

    return questions
def cantor(image = None, sample=None):
    image_text = ""
    cantor_input = build_cantor_input(sample)

    if image:
        image_text = handle_image(image, cantor_input)
        cantor_input += f"Here is the image information {image_text}"

    # Decision-making stage
    decision_answer = decision_stage(cantor_input)

    # Answer integration stage
    final_result, o_prompt = get_final_result(decision_answer, cantor_input)

    return final_result, cantor_input, o_prompt


