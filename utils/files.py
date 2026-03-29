import os
import json
import numpy as np
import jsonlines
from loguru import logger

from .validate import validate_answers


RESULTS_DIR = os.path.join("output", "results")
OUTPUTS_DIR = os.path.join("output", "cache")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def save_outputs(outputs, dir, file_name):
    json_dict = json.dumps(outputs, cls=NpEncoder)
    outputs = json.loads(json_dict)

    output_dir = os.path.join(OUTPUTS_DIR, dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{file_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=4)


def save_results(results, dir, file_name="answers"):
    results_dir = os.path.join(RESULTS_DIR, dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, f"{file_name}.jsonl")
    with jsonlines.open(results_path, "w") as writer:
        writer.write_all(results)

    logger.info(f"Results saved to {results_path}")

    has_errors = validate_answers(os.path.abspath(results_path))
    if not has_errors:
        logger.info("Results are valid.")
    else:
        logger.error("Results ARE NOT valid. Please check the results file!")


def load_results(dir, file_name="answers"):
    results_path = os.path.join(RESULTS_DIR, dir, f"{file_name}.jsonl")
    return load_jsonl(results_path)


def load_jsonl(file_path):
    results = []
    with jsonlines.open(file_path, "r") as reader:
        for obj in reader:
            results.append(obj)
    return results


def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results
