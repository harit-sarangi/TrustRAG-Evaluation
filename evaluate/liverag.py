import json
from loguru import logger
from typing import Dict, Any

from generation.backend import GenerationBackend
from utils.strings import clean_str

LIVERAG_EVAL_PROMPT = """
You are an expert evaluator assessing the quality of an answer to a given question based on retrieved passages.
Your evaluation must measure `correctness` and `faithfulness` according to predefined criteria.
"""

LIVERAG_EVAL_QEURY = """
Please evaluate the **generated answer** based on these specific metrics:

1. Correctness. Combines elements of:
- **coverage**: portion of vital information, in the ground truth answer which is covered by the generated answer.
- **relevance**: portion of the generated response which is directly addressing the question, regardless its factual correctness.  

Graded on a continuous scale with the following representative points:
- **2:** Correct and relevant (no irrelevant information)
- **1:** Correct but contains irrelevant information
- **0:** No answer provided (abstention)
- **-1:** Incorrect answer

2. Faithfulness. Assesses whether the response is **grounded in the retrieved passages**.

Graded on a continuous scale with the following representative points:
- **1:** Full support. All answer parts are grounded
- **0:** Partial support. Not all answer parts are grounded
- **-1:** No support. All answer parts are not grounded


The question that was asked:
{question}

Ground truth answer:
{gold_answer}

Ground truth passage from a document:
{gold_document}

Generated answer:
{generated_answer}

Retrieved passages:
{retrieved_passages}

**Return the result strictly in JSON format:**
{{
  "correctness": <int>,
  "faithfulness": <int>
}}
"""


def liveRag_eval(
    be: GenerationBackend,
    question: str,
    answer: str,
    ret_documents: list[str],
    correct_answer: str = None,
    gold_documents: list[str] = None,
    retries: int = 10,
    temperature: float = 0.1,
) -> Dict[str, Any]:
    clean_question = clean_str(question)
    clean_answer = clean_str(answer)
    clean_ret_documents = clean_str(ret_documents)
    clean_correct_answer = clean_str(correct_answer) if correct_answer else ""
    clean_gold_documents = clean_str(gold_documents) if gold_documents else ""

    eval_prompt = LIVERAG_EVAL_QEURY.format(
        question=clean_question,
        generated_answer=clean_answer,
        gold_answer=clean_correct_answer,
        gold_document=clean_gold_documents,
        retrieved_passages=clean_ret_documents,
    )

    correctness, faithfulness = None, None
    for attempt in range(retries):
        try:
            response = be.chat_completions(prompt=eval_prompt, system_prompt=LIVERAG_EVAL_PROMPT, temperature=temperature, format="json_object")
            
            # --- FIX: Handle Dictionary Response (Semantic Confidence Format) ---
            if isinstance(response, dict):
                # The actual JSON string we want is inside the 'answer' key
                response_text = response.get("answer", "{}")
            else:
                response_text = response
            # ------------------------------------------------------------------

            try:
                eval_data = json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response: {response_text}")
                eval_data = {}

            logger.trace(f"Evaluation succeeded after {attempt + 1} attempt.")
            correctness = eval_data.get("correctness", None)
            faithfulness = eval_data.get("faithfulness", None)
            logger.debug(f"Evaluated: correctness={correctness}, faithfulness={faithfulness}")
            break
        except Exception as e:
            if attempt < retries - 1:
                logger.opt(exception=e).warning(f"Evaluation failed on attempt {attempt + 1}/{retries}, retrying...")
            else:
                logger.opt(exception=e).error("Max retries reached. Evaluation failed.")

    return {
        "correctness": correctness,
        "faithfulness": faithfulness,
    }