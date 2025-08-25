# MLLM-vs-LLM Fine-grained Verification (ScienceQA)

This repository implements a two-stage framework for multimodal reasoning with fine-grained entity verification.  
Stage 1 produces an initial answer with reasoning steps. Stage 2 performs adversarial verification across multiple models to check named entities and revise the final answer.

---

## File Structure

```
.
├── cantor.py                # Main entry for Stage 1: input assembly, decision, and integration
├── cantor_function.py       # Helper functions for image handling, decision, and answer generation
├── cantor_prompt.py         # Prompt templates for Stage 1 modules
├── MLLLMvMLLM_function.py   # Core functions for Stage 2: entity extraction, verification, correction
├── MLLLMvMLLM_prompt.py     # Prompt templates for Stage 2 modules
├── kimitest.py              # Minimal DeepSeek/Kimi API test
├── test.py                  # Example evaluation loop for ScienceQA
└── error case.py            # Simple API call example for debugging
```

---

## Dependencies

- Python >= 3.9
- Required libraries:
  ```bash
  pip install openai google-generativeai datasets pillow tqdm
  ```
- Optional:  
  - `ipython` (for display)  
  - `pycparser` (for parsing tasks)

---

## Environment Variables

You must configure API keys before running:

```bash
export OPENAI_API_KEY="sk-..."     # GPT-3.5 for answering and correction
export Kimi_key="sk-..."           # Moonshot (Kimi) for adversarial question generation
export DEEP_KEY="sk-..."           # DeepSeek for decision and review
export GOOGLE_KEY="..."            # Gemini 1.5 Flash for image information extraction
```

---

## Quick Start

### 1. Run a Single Sample

```python
from datasets import load_dataset
from cantor import cantor

data = load_dataset('derek-thomas/ScienceQA', split='test')
sample = data[4]

chain, cantor_input, first_prompt = cantor(
    image=sample["image"] if sample["image"] else None,
    sample=sample
)

print(chain)  # Outputs rationale and answer
```

### 2. End-to-End with Verification

```bash
python test.py
```

This pipeline will:
1. Generate an initial answer with reasoning.  
2. Extract named entities from the rationale.  
3. Generate verification questions.  
4. Answer verification questions using GPT-3.5.  
5. Review and score answers with DeepSeek/Kimi.  
6. Aggregate low-score cases and build correction prompts.  
7. Revise the answer using GPT-3.5 to produce the final output.  

---

## Core Scripts

### `cantor.py`
Stage 1 entry. Combines question, options, subject, grade, and hints. Calls:
- Gemini for extracting visual facts
- DeepSeek for decision planning
- OpenAI GPT-3.5 for rationale and final answer

### `cantor_function.py`
- `handle_image`: Extracts visual elements using Gemini  
- `decision_stage`: Decides necessary modules with DeepSeek  
- `get_final_result`: Generates reasoning and answer with GPT-3.5  

### `cantor_prompt.py`
Contains prompts for:
- Image handling  
- Decision stage  
- Final answer generation  

### `MLLMvMLLM_function.py`
- `extract_entities_from_rationale`: Extracts named entities  
- `generate_global_verification_questions`: Creates verification questions  
- `answer_with_context`: Answers questions with GPT-3.5  
- `kimi_evaluate_answer`: Reviews and scores with DeepSeek  
- `answer_correction`: Aggregates low scores and requests corrections  

### `MLLMvMLLM_prompt.py`
Prompt templates for:
- Entity extraction  
- Verification question generation  
- Review and scoring  

### `test.py`
Provides an example evaluation script with accuracy calculation and optional category-level statistics.

### `kimitest.py`
Minimal test script for DeepSeek/Kimi API.

### `error case.py`
Simple DeepSeek API call for debugging and connectivity testing.

---

## Notes

- Dataset: `derek-thomas/ScienceQA` from HuggingFace `datasets`  
- Multiple APIs are required; monitor usage, cost, and rate limits  
- Ensure compliance with privacy and API service terms before deployment  

---

## License

This project is provided for demonstration purposes only. Use APIs and datasets according to their respective licenses and terms.
