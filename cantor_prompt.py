handle_image_prompt = """You are an expert in visual understanding and reasoning.

You will be given an image (possibly from a science exam), a multiple-choice question related to it, and a set of answer options.

Your task is to carefully examine the image and extract all **visually observable information** that may be relevant to understanding or answering the question. Focus on elements that are **directly linked to the question and options**.

Please include the following in your structured description:

- Key objects, entities, or figures shown
- Any visible **labels, text, or symbols**
- Layouts, arrows, color codes, or **spatial relationships**
- **Chart or diagram features**, such as axes, units, data points, or trends
- **Any specific elements mentioned in the question or answer options**
- Highlight any elements that may help **compare or differentiate options**

Avoid making assumptions or choosing an answer. Do **not** provide a final answer. Your goal is only to extract visual facts that may support reasoning.

Structure your response as a clear, detailed description that will be used in the next step of reasoning.

"""

decision_stage_prompt = f"""You are a advanced question-answering agent equipped with four specialized modules to aid in 
    analyzing and responding to queries about images:
    1. TextIntel Extractor: This module extracts and converts text within images into editable text format. 
    It's particularly useful for images containing a mix of text and graphical elements. When this module is 
    required, specify your request as: "TextIntel Extractor: <specific task or information to extract>."
    2. ObjectQuant Locator: This module identifies and locates objects within an image. It's adept at 
    counting objects and determining their spatial arrangement. When you need this module, frame your 
    request as: "ObjectQuant Locator: <object1, object2, ..., objectN>," listing the objects you believe need 
    detection for further analysis.
    3. VisionIQ Analyst: This module processes and interprets visual data, enabling you to ask any queries 
    related to the image's content. When information from this module is needed, phrase your request as: 
    "VisionIQ Analyst: <your question about the image>."
    4.ChartSense Expert: This module specializes in analyzing and interpreting information from charts 
    and graphs. It can extract data points, understand trends, and identify key components such as titles, 
    axes, labels, and legends within a chart. When you require insights from a chart or graph, specify your 
    request as: "ChartSense Expert: <specific aspect of the chart you're interested in or question you have 
    about the chart>."
    When faced with a question about an image, which will be accompanied by a hint that might not cover 
    all its details, your task is to:
    If the question can be answered directly based on the information provided without the need for 
    detailed input from the modules, specify this explicitly. Do not disclose the answer itself.
    Otherwise:
    - Provide a rationale for your approach to answering the question, explaining how you will use the 
    information from the image and the modules to form a comprehensive answer.
    - Assign specific tasks to each module as needed, based on their capabilities, to gather additional 
    information essential for answering the question accurately.
    Your response should be structured as follows:
    Answer: 
    ["This question does not require any modules and can be answered directly based on the information 
    provided."] or [Rationale: Your explanation of how you plan to approach the question, including any 
    initial insights based on the question and image information provided. Explain how the modules' input 
    will complement this information.]
    Modules' tasks (if applicable): 
    1. TextIntel Extractor: [Specify the text or information to be extracted from the image, if necessary.] 
    2. ObjectQuant Locator: [List the objects to be identified or counted in the image, if required.] 
    3. VisionIQ Analyst: [Pose any specific questions you have about the image that require deeper visual 
    analysis, if applicable.]
    4. ChartSense Expert: [Extract chart data or specify any questions about the chart, if required.]
    Ensure your response adheres to this format to systematically address the question using the available 
    modules or direct analysis as appropriate.

    Here are some examples:
    
    "Question1":"Which solution has a higher concentration of blue particles?",
    "Choices":[
    "Solution B",
    "neither; their concentrations are the same",
    "Solution A"
    ],
    Answer: 
    1. Concentration in a solution refers to the amount of a substance (solute) present in a specified amount 
    of another substance (solvent). 
    2. To know the solvent volume, we need TextIntel Extractor to extract information about the volume 
    from the image.
    3. To understand the number of blue particles in solution A and solution B, we need a ObjectQuant 
    Locator to detect them.
    Modules' tasks:
    1. TextIntel Extractor: Extract keywords related to solution volume in Solution A and Solution B.
    2. ObjectQuant Locator: the number of blue particles in Solution A and Solution B.
    
    "Question2":"Compare the average kinetic energies of the particles in each sample. Which sample has 
    the higher temperature?",
    "Choices":[
    "neither; the samples have the same temperature",
    "sample B",
    "sample A"
    ],
    Answer:
    1. The temperature of particles in a substance is directly proportional to its average kinetic energy. The 
    formula for kinetic energy of an particle is 1/2mv^2, where m represents the mass of the object and v 
    represents its velocity. 
    2. So TextIntel Extractor is needed to retrieve key information about mass and speed in the image.
    Modules' tasks:
    1. TextIntel Extractor: Extract keywords related to mass and speed in sample A and sample B.

    "Question3":"Think about the magnetic force between the magnets in each pair. Which of the 
    following statements is true?",
    "Choices":[
    "The magnetic force is stronger in Pair 2.",
    "The magnetic force is stronger in Pair 1.",
    "The strength of the magnetic force is the same in both pairs."
    ],
    Answer:
    1. The magnitude of magnetic force is independent of the direction of the magnetic pole, and is not 
    related to whether the magnetic pole is attracted or repelled.
    2. Different pairs of magnets will not affect each other.
    3. The magnitude of magnetic force is inversely proportional to the distance between two magnets.
    4. To know the distance between magnets, TextIntel Extractor is needed to extract distance information.
    Modules' tasks:
    1. TextIntel Extractor: Extract the distance values between two pairs of magnets in pair1 and pair2 
    separately.
    
    "Question4":"What is the expected ratio of offspring with mutated antennae to offspring with normal 
    antennae? Choose the most likely ratio.",
    "Choices":[
    "1:3",
    "0:4",
    "3:1",
    "2:2",
    "4:0"
    ]
    Answer:
    1. In a group of fruit flies, some individuals have mutated antennae and others have normal antennae. 
    In this group, the gene for the antenna type trait has two alleles. 
    2. The allele 'A' is for mutated antennae, the allele 'a' is for normal antennae.
    3. To know the specific gene composition, we need TextIntel Extractor to extract information from 
    Punnett square.
    Modules' tasks:
    1. TextIntel Extractor: Extract the genotypes from the Punnett square.

    "Question5":"Which month is the wettest on average in Christchurch?",
    "Choices":[
    "August",
    "April",
    "May"
    ]
    Answer: 
    1. This question provides a chart of Christchurch precipitation. We need to compare the monthly 
    precipitation in Christchurch to determine which month is the wettest.
    2. In order to determine which month has the highest precipitation, we need ChartSense Expert to 
    extract the precipitation for each month
    Modules' tasks:
    1. ChartSense Expert: Extract the precipitation for each month from the chart.
    
    "Question6": "What is the capital of New Jersey?",
    "Choices": [
    "Augusta",
    "Montpelier",
    "Newark",
    "Trenton"
    ] 
    Answer:
    1. This question does not require any modules and can be answered directly based on the information 
    provided.
    Please refer to the prompts and examples above to help me solve the following problem:
    """

get_result_prompt =  f"""You are a knowledgeable and skilled science expert.

    Please answer the following multiple-choice question by thinking step-by-step. You will be given:
    - The question
    - The answer choices
    - Some supplementary information (may be incomplete or partially misleading)
    
    Your job is to reason gradually and explain your thinking at each step. You should:
    - First try to use your own knowledge to answer the question.
    - If needed, use the supplementary information to support or refine your reasoning.
    - Do not rely entirely on the supplementary information, and be cautious if it's vague or insufficient.
    - Always aim to give a confident answer. Avoid indecision or uncertainty.
    
    Please format your response as follows:
    Rationale:
    Step 1: [Initial understanding of the question and terms]
    Step 2: [Apply relevant knowledge or concepts]
    Step 3: [Consider or verify against supplementary information if necessary]
    ...
    Final Decision: [State clearly which choice is correct and why]
    Answer: <copy the exact answer from the choices provided, do not rephrase or invent new words>
    """
