from loguru import logger
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Any, Tuple

from .liverag import liveRag_eval
from generation.backend import GenerationBackend


def eval_batch(be: GenerationBackend, answers, data):
    evaluations = []
    for i, answerobj in enumerate(tqdm(answers, total=len(answers), desc="Evaluating answers", unit="answer")):
        evaluation = liveRag_eval(
            be,
            question=answerobj["question"],
            answer=answerobj["answer"],
            ret_documents=answerobj["passages"],
            correct_answer=data[i].get("answer", None) if data else None,
            gold_documents=data[i].get("context", None) if data else None,
        )
        evaluations.append(evaluation)

    return evaluations


def summarize_eval(results: List) -> Tuple[Dict[str, Any], List[str]]:
    collected_errors = []
    aggregated_scores = defaultdict(lambda: {"counts": defaultdict(int), "distribution_percentage": {}})

    # Step 1: Count occurrences of each score
    for eval_result in tqdm(results, desc="Summarizing evaluations", unit="question", leave=False, delay=1):
        if not isinstance(eval_result, dict):
            collected_errors.append(f"Evaluation result is not a dictionary: {eval_result}")
            continue

        for score_name, score_value in eval_result.items():
            # Handle None values so they don't break the dict keys
            if score_value is None:
                score_value = "None"
            
            aggregated_scores[score_name]["counts"][str(score_value)] += 1

    # Step 2: Calculate distribution percentages
    for score_name, data in aggregated_scores.items():
        total_occurrences_of_score = sum(data["counts"].values())
        if total_occurrences_of_score > 0:
            for value_str, count in data["counts"].items():
                percentage = (count / total_occurrences_of_score) * 100
                data["distribution_percentage"][value_str] = f"{percentage:.2f}"

        if not data["distribution_percentage"] and data["counts"]:
            for value_str in data["counts"].keys():
                data["distribution_percentage"][value_str] = "0.00"

    # Step 3: Calculate average scores
    for score_name, data in aggregated_scores.items():
        # Safe math loop that ignores "None" strings to prevent crashes
        total_score = 0
        valid_count = 0
        for value_str, count in data["counts"].items():
            if value_str != "None":
                try:
                    total_score += int(value_str) * count
                    valid_count += count
                except ValueError:
                    pass
        
        if valid_count > 0:
            average_score = total_score / valid_count
            data["average"] = average_score
        else:
            data["average"] = None
        

    # Step 4: Convert defaultdict to normal dict
    final_aggregated_scores = {
        key: {
            "counts": dict(sorted(value["counts"].items())),
            "distribution_percentage": dict(sorted(value["distribution_percentage"].items())),
            "average": value["average"],
        }
        for key, value in aggregated_scores.items()
    }

    return final_aggregated_scores, collected_errors


def print_eval_summary(aggregated_scores: Dict[str, Any], collected_errors: List[str]):
    logger.debug(f"Evaluation errors: {collected_errors}")
    logger.debug(f"Aggregated scores: {aggregated_scores}")

    logger.info(f"Correctness: {aggregated_scores['correctness']['distribution_percentage']}")
    logger.info(f"Correctness average: {aggregated_scores['correctness']['average']}")
    logger.info(f"Faithfulness: {aggregated_scores['faithfulness']['distribution_percentage']}")
    logger.info(f"Faithfulness average: {aggregated_scores['faithfulness']['average']}")