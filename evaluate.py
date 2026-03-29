import argparse
import time
from loguru import logger
from dotenv import load_dotenv, find_dotenv

from evaluate import eval_batch, summarize_eval, print_eval_summary
from generation import get_backend
from utils.logging import setup_logging
from utils.files import load_jsonl, load_results, save_outputs
from main import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--answers_path", type=str, help="Path to the answers file, if not specified, will use output/results/$log_name/answers.jsonl")

    parser.add_argument("--dataset_path", type=str, help="Path to the dataset file, if not specified, will use the database")
    parser.add_argument("--limit", type=int, default=-1, help="Number of questions to process, -1 to process all")
    parser.add_argument("--skip", type=int, default=-1, help="Number of questions to skip, -1 to skip none")
    parser.add_argument("--random", action="store_true", help="Randomly sample questions from the dataset")

    parser.add_argument("--eval_backend", type=str, default="openai", choices=["local", "openai", "ollama"], help="Backend to use for evaluation LLM")
    parser.add_argument("--eval_model", type=str, default="gemma3:27b-it-fp16", help="Model to use for evaluation LLM, no evaluation if not specified")

    parser.add_argument("--log_name", type=str, help="Name of log and result")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (Debug level)")
    parser.add_argument("-vv", "--very-verbose", action="store_true", help="Enable very verbose logging (Trace level)")

    args = parser.parse_args()
    return args


def main():
    main_time = time.perf_counter()
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
    load_dotenv(dotenv_path, verbose=True, override=True)

    args = parse_args()
    setup_logging(console_level="TRACE" if args.very_verbose else "DEBUG" if args.verbose else "INFO")
    logger.debug("Found .env file at: " + dotenv_path)
    logger.info(args)

    data = None
    if args.dataset_path:
        data, _ = load_dataset(args)

    if args.answers_path:
        answers = load_jsonl(args.answers_path)
    else:
        answers = load_results(args.log_name, "answers")

    if args.limit > 0:
        answers = answers[: args.limit]

    evalBackend = get_backend(args.eval_backend, args.eval_model)
    logger.info(f"Using {args.eval_backend} backend with model {args.eval_model} for evaluation")

    evaluations = eval_batch(evalBackend, answers, data)

    save_outputs(evaluations, args.log_name, "evaluations")
    aggregated_scores, errors = summarize_eval(evaluations)
    print_eval_summary(aggregated_scores, errors)

    logger.info(f"Evaluation took (seconds): {time.perf_counter() - main_time}")


if __name__ == "__main__":
    main()
