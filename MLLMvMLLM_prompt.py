extract_prompt = f"""
    You are an information extraction assistant.
    
    Given the following reasoning text, extract all the key named entities that relate to scientific or literary concepts. These may include:
    - Specific terminology (e.g., apostrophe, chiasmus)
    - Proper names (e.g., Homer, Achilles)
    - Quoted phrases (e.g., "O goddess")
    - Work titles (e.g., The Iliad)
    
    Please return your answer as a JSON list of strings.
    Here are some examples:
    ### Example 1
    Reasoning Chain:
    1. Oxygen is important for breathing in the air.
    2. That means oxygen must be the main gas.
    3. The most abundant gas is oxygen.
    
    Entities:
    - oxygen (Scientific Object)
    - breathing (Process)
    - air (Location)
    - main (Attribute)
    - abundant (Attribute)
    
    ---
    
    ### Example 2
    Reasoning Chain:
    1. The user reported swelling and pain at the back of the neck.
    2. These symptoms suggest inflammation.
    3. Therefore, it might be a skin allergy.
    
    Entities:
    - swelling (Symptom)
    - pain (Symptom)
    - neck (Location)
    - inflammation (Medical Condition)
    - skin allergy (Medical Condition)
    
    ---
    
    ### Example 3
    Reasoning Chain:
    1. Mercury is a liquid at room temperature.
    2. Most metals are solid in that condition.
    3. So mercury has an unusual physical property.
    
    Entities:
    - mercury (Scientific Object)
    - liquid (Physical Property)
    - room temperature (Unit or Quantity)
    - metals (Scientific Object)
    - solid (Physical Property)
    - unusual (Attribute)
    
    ---
    
    ### Example 4
    Reasoning Chain:
    1. The thermometer shows 38.5°C.
    2. That temperature is higher than the normal range.
    3. It indicates a potential fever.
    
    Entities:
    - thermometer (Scientific Object)
    - 38.5°C (Unit or Quantity)
    - temperature (Physical Property)
    - normal range (Unit or Quantity)
    - fever (Medical Condition)
    
    ---
    
    ### Example 5
    Reasoning Chain:
    1. The traffic light turned red.
    2. Red means stop in traffic rules.
    3. So the car must halt.
    
    Entities:
    - traffic light (Scientific Object)
    - red (Attribute)
    - stop (Process)
    - traffic rules (Location)
    
    ---
    
    ### Now extract entities from the following reasoning chain:
"""

question_prompt = f"""Respond only with the questions in bullet point format.
Original Question: "What gas is most abundant in Earth's atmosphere?"
Context: "The diagram shows percentages of different gases. Nitrogen is 78%, Oxygen is 21%."
Reasoning Chain: "Step 1: Nitrogen is 78%. Step 2: Therefore, it is abundant. Step 3: Oxygen is second."
Named Entities:
[nitrogen, 78%, abundant, oxygen, second]

Questions:
- What gas has the highest percentage in the atmosphere?
- Does 78% justify describing nitrogen as 'abundant'?
- Can oxygen be called 'second' if nitrogen is higher?
- Is 'abundant' a reasonable description for nitrogen in this context?

Please return your answer as a JSON list of strings.
Now please generate questions for the input above:
"""

support_prompt = f"""You are a semantic reviewer.
You are an expert reviewer helping to assess the factual correctness and reasoning quality of answers to verification questions.

Please evaluate the provided answer to a specific verification question, given:
- The original question and context
- The final answer based on that reasoning
- A follow-up verification question targeting a specific point
- The model’s answer to that verification question

Your job is to assess whether the model's answer to the verification question is appropriate, based on the original information and logic.
You must select **one keyword** from the following list to summarize your judgement:

["inappropriate", "incorrect", "wrong", "conflict", "replace",
 "unclear", "ambiguous", "maybe", "not sure", "consider",
 "appropriate", "correct", "reasonable", "yes", "accurate"]

After selecting the keyword, explain your reason briefly.

Use the format:
Keyword: <keyword>
Reason: <reason>

Here are some examples:

---

Question: What is the most abundant gas in Earth’s atmosphere?  
Answer: The most abundant gas is carbon dioxide.  
Keyword: incorrect  
Reason: The answer is factually wrong. Carbon dioxide is not the most abundant gas.

---

Question: Is 78% enough to call nitrogen 'abundant'?  
Answer: Maybe, but it depends on the definition of 'abundant'.  
Keyword: ambiguous  
Reason: The answer does not clearly support or reject the statement.

---

Question: Does oxygen make up 21% of the atmosphere?  
Answer: Yes, oxygen comprises about 21% of the atmosphere.  
Keyword: correct  
Reason: This statement is accurate and aligns with scientific facts.

---
"""