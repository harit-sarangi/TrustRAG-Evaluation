import argparse
from functools import cache
import os
import random
import sys
import time
import json
from typing import List, Tuple, Dict, Any

import jsonlines
import numpy as np
import torch
from tqdm import tqdm, trange
from utils.logging import setup_logging, compound_log_name
from utils.time import time_measurement, time_summarize
from utils.files import save_results, save_outputs
import evaluate
import reranking
import retrieval
import generation
from loguru import logger
from dotenv import load_dotenv, find_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rag")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file, if not specified, will use the database")
    parser.add_argument("--limit", type=int, default=-1, help="Number of questions to process, -1 to process all")
    parser.add_argument("--skip", type=int, default=-1, help="Number of questions to skip, -1 to skip none")
    parser.add_argument("--random", action="store_true", help="Randomly sample questions from the dataset")

    parser.add_argument("--ret_backend", type=str, default="opensearch", choices=["none", "opensearch", "pinecone"], help="Backend to use for retrieval")
    parser.add_argument("--ret_method", type=str, default="top_k", choices=["top_k"], help="Method to retrieve documents")
    parser.add_argument("--ret_top_k", type=int, default=10, help="Number of documents to retrieve")
    parser.add_argument("--ret_rephrase", action="store_true", help="Rephrase the question before retrieval")

    parser.add_argument("--rerank_backend", type=str, default="none", choices=["none", "local"], help="Backend to use for reranking")
    parser.add_argument("--rerank_model", type=str, help="Reramker model (Transformers sequence classification model)")
    parser.add_argument("--rerank_top_k", type=int, default=5, help="Number of documents to keep after reranking")

    parser.add_argument("--invert", action="store_true", help="Invert the order of the retrieved documents (applied after reranking, if any)")

    parser.add_argument("--query_backend", type=str, default="openai", choices=["none", "local", "openai", "ollama"], help="Backend to use for RAG LLM")
    parser.add_argument("--query_method", type=str, default="instruct", choices=["simple", "trustrag", "astute", "instruct"], help="RAG Method to query the LLM")
    parser.add_argument("--query_model", type=str, default="falcon3:7b", help="Model to use for RAG LLM")

    parser.add_argument("--eval_backend", type=str, default="none", choices=["none", "local", "openai", "ollama"], help="Backend to use for evaluation LLM")
    parser.add_argument("--eval_model", type=str, help="Model to use for evaluation LLM, no evaluation if not specified")

    parser.add_argument("--retry_times", type=int, default=5, help="Retry this many times if the query fails. If reached, empty ansswer will be returned.")
    parser.add_argument("--repeat_times", type=int, default=1, help="Repeat several times to compute average. Results will be overridden with the last run, logs are persisted.")
    parser.add_argument("--seed", type=int, default=int(time.time()), help="Random seed, as default, current time in seconds is used to have sortable results")
    parser.add_argument("--log_name", type=str, help="Name of log and result")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (Debug level)")
    parser.add_argument("-vv", "--very_verbose", action="store_true", help="Enable very verbose logging (Trace level)")

    args = parser.parse_args()
    if not args.log_name:
        args.log_name = compound_log_name(args)
    return args


@cache
def get_query_func(method: str) -> generation.QueryFunction:
    if method == "simple":
        return generation.simple_query
    elif method == "trustrag":
        return generation.trustrag_query
    elif method == "astute":
        return generation.astute_query
    elif method == "instruct":
        return generation.instructrag_query
    else:
        raise ValueError(f"Invalid query method: {method}")


@cache
def get_retrieval_func(method: str) -> retrieval.RetrievalFunction | None:
    if not method or method == "none":
        return None
    elif method == "top_k":
        return retrieval.top_k
    else:
        raise ValueError(f"Invalid retrieval method: {method}")


def load_dataset(args: argparse.Namespace) -> Tuple[List[dict], int]:
    dataset_path = os.path.normpath(args.dataset_path)
    logger.info("Loading dataset from " + dataset_path)
    dataset = []
    with jsonlines.open(dataset_path) as reader:
        for obj in reader:
            dataset.append(obj)

    # warnings here, because it's expected to be used with caution
    if args.skip > 0:
        logger.warning(f"Skipping first {args.skip} questions")
        dataset = dataset[args.skip :]

    if args.limit > 0 and args.limit < len(dataset):
        logger.warning(f"Limiting dataset to {args.limit} questions")
        dataset = dataset[: args.limit]

    if args.random:
        logger.warning("Randomly sampling questions from the dataset")
        random.shuffle(dataset)

    return dataset, len(dataset)


def apply_rag(
    args: argparse.Namespace, question: str, retBackend: retrieval.RetrievalBackend, genBackend: generation.GenerationBackend, rerankBackend: reranking.RerankingBackend = None
) -> Tuple[str, str, List[str], List[str], Dict[str, float]]:
    perf_stats = {}
    query_func = get_query_func(args.query_method)
    retrieval_func = get_retrieval_func(args.ret_method)

    context = ""
    doc_ids = []
    doc_passages = []
    if retrieval_func:
        ret_question = question
        if args.ret_rephrase:
            with time_measurement(perf_stats, "rephrase"):
                ret_question, ignore = generation.rephrase(genBackend, question)
                logger.debug(f"Rephrased question: {ret_question}")

        with time_measurement(perf_stats, "retrieval"):
            doc_ids, doc_passages = retrieval_func(retBackend, ret_question, args.ret_top_k)

        with time_measurement(perf_stats, "reranking"):
            if rerankBackend:
                doc_ids, doc_passages = reranking.rerank_docs(rerankBackend, question, doc_ids, doc_passages, top_k=args.rerank_top_k)

        # Invert ordering of retrieved context based on
        # The Power of Noise: Redefining Retrieval for RAG Systems. SIGIR 2024
        if args.invert:
            logger.debug("Inverting document order")
            doc_ids.reverse()
            doc_passages.reverse()

        for index, document in enumerate(doc_passages):
            context += f"Externally Retrieved Document{index}:" + document + "\n"
    else:
        logger.warning("No retrieval method specified, using empty context")

    with time_measurement(perf_stats, "rag_query"):
        #Original line ->final_answer, final_prompt = query_func(genBackend, question, context)
        #Modification by Harit
        query_output = query_func(genBackend, question, context)
        
        if isinstance(query_output, tuple) and len(query_output) == 3:
            final_answer, final_prompt, confidence = query_output
        else:
            final_answer, final_prompt = query_output
            confidence = None
            
        perf_stats["confidence"] = confidence
        #Till here
    
    return final_answer, final_prompt, doc_ids, doc_passages, perf_stats


def _flatten_passages(doc_ids: List[str], doc_passages: List[str]) -> List[dict]:
    structured_passages = []
    for i in range(len(doc_ids)):
        structured_passages.append(
            {
                "doc_IDs": [doc_ids[i]],  # Wrap in list as requested
                "passage": doc_passages[i],
            }
        )
    return structured_passages


def process_question(args: argparse.Namespace, qobj: dict, **kwargs) -> Tuple[Dict[str, Any], str | None, Dict[str, float]]:
    id = qobj.get("id")
    question = qobj.get("question")
    request_id = qobj.get("request_id", None)

    final_answer, final_prompt, doc_ids, doc_passages, perf_stats = apply_rag(args, question, **kwargs)

    answer = {
        "id": id,  # only numeric id is expected
        "question": question,
        "answer": final_answer,
        "final_prompt": final_prompt,
        "passages": _flatten_passages(doc_ids, doc_passages),
        #Modification by Harit
        "confidence": perf_stats.get("confidence", None),
        #Till here
    }

    logger.info(" " + final_answer)
    return answer, request_id, perf_stats


def retry_question(args: argparse.Namespace, qobj: dict, **kwargs) -> Tuple[Dict[str, Any], str | None, Dict[str, float]]:
    logger.info(f"Question: {qobj['question']}")
    for attempt in range(args.retry_times):
        try:
            return process_question(args, qobj, **kwargs)
        except Exception as e:
            if attempt < args.retry_times - 1:
                logger.opt(exception=e).warning(f"Attempt {attempt + 1}/{args.retry_times} failed, retrying...")
            else:
                logger.opt(exception=e).error("Out of attempts to process the question, falling back to empty answer")

    # If all attempts fail, return "I don't know" as the answer
    return (
        {
            "id": qobj.get("id", None),
            "question": qobj.get("question", None),
            "passages": [],
            "final_prompt": "The system is not responding, answer with 'I don't know'",
            "answer": "I don't know",
        },
        qobj.get("request_id", None),
        {},
    )


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    main_time = time.perf_counter()
    dotenv_path = find_dotenv(raise_error_if_not_found=True)
    load_dotenv(dotenv_path, verbose=True, override=True)

    args = parse_args()
    setup_seeds(args.seed)
    setup_logging(args.log_name, file_level="DEBUG", console_level="TRACE" if args.very_verbose else "DEBUG" if args.verbose else "INFO")
    logger.debug("Found .env file at: " + dotenv_path)
    logger.info(args)

    for it in trange(args.repeat_times, desc="Running iterations", unit="iteration", disable=args.repeat_times <= 1):
        
        # --- PATHS AND SUMMARY LOADING FIX ---
        results_dir = os.path.join("output", "results", args.log_name)
        summary_path = os.path.join(results_dir, "summary.json")
        answers_path = os.path.join(results_dir, "answers.jsonl")
        perf_stats_path = os.path.join(results_dir, "perf_stats.json")
        
        existing_summary = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    existing_summary = json.load(f)
                logger.info("Found existing summary.json. Data will be merged.")
            except Exception as e:
                logger.warning(f"Could not load existing summary: {e}")
        # --- END PATHS FIX ---

        answers = []
        request_ids = []
        times = []
        data, total = load_dataset(args)
        aggregated_scores = None

        # --- GENERATION VS EVALUATION SPLIT FIX ---
        if args.query_backend != "none":
            logger.info(f"Using {args.ret_backend} backend for retrieval")
            retBackend = retrieval.get_backend(args.ret_backend)

            logger.info(f"Using {args.query_backend} backend with model {args.query_model} for generation")
            genBackend = generation.get_backend(args.query_backend, args.query_model)

            logger.info(f"Using {args.rerank_backend} backend with model {args.rerank_model} for reranking")
            rerankBackend = reranking.get_backend(args.rerank_backend, args.rerank_model)

            for qobj in tqdm(data, total=total, desc="Generating answers", unit="question"):
                start_time = time.perf_counter()
                with logger.contextualize(question_id=qobj["id"]):
                    answer, request_id, perf_stats = retry_question(args, qobj, retBackend=retBackend, genBackend=genBackend, rerankBackend=rerankBackend)
                perf_stats["rag_total"] = time.perf_counter() - start_time

                answers.append(answer)
                times.append(perf_stats)
                if request_id is not None:
                    request_ids.append(request_id)

                save_results(answers, args.log_name, "answers")
                save_outputs(times, args.log_name, "perf_stats")
                if len(request_ids) > 0:
                    save_outputs(request_ids, args.log_name, "request_ids")

            del retBackend, genBackend, rerankBackend
        else:
            logger.info("Query backend is 'none'. Loading existing answers for evaluation...")
            if os.path.exists(answers_path):
                with jsonlines.open(answers_path) as reader:
                    answers = [obj for obj in reader]
            else:
                logger.error(f"Cannot evaluate: No answers file found at {answers_path}")
                continue
            
            if os.path.exists(perf_stats_path):
                try:
                    with open(perf_stats_path, 'r') as f:
                        times = json.load(f)
                except Exception:
                    pass
        # --- END SPLIT FIX ---

        if args.eval_model:
            evalBackend = generation.get_backend(args.eval_backend, args.eval_model)
            logger.info(f"Using {args.eval_backend} backend with model {args.eval_model} for evaluation")
            
            # --- START FIX: The WSL Network Pause ---
            logger.info("Pausing for 5 seconds to clear WSL network sockets...")
            time.sleep(5)
            # --- END FIX ---
            
            # Run the evaluation
            evaluations = evaluate.eval_batch(evalBackend, answers, data)
            del evalBackend

            # Save raw evaluation logs
            save_outputs(evaluations, args.log_name, "evaluations")
            
            # --- START FIX: Merge Scores & Save ---
            logger.info("Merging evaluation scores into results file...")
            for i, score in enumerate(evaluations):
                if score and i < len(answers):
                    answers[i]["correctness"] = score.get("correctness")
                    answers[i]["faithfulness"] = score.get("faithfulness")
            
            # Overwrite the answers file with the scores included
            save_results(answers, args.log_name, "answers")
            # --- END FIX ---

            aggregated_scores, errors = evaluate.summarize_eval(evaluations)
            evaluate.print_eval_summary(aggregated_scores, errors)

        avg_times, total_times = time_summarize(times)
        
        # 6. Construct the Master Summary Dictionary
        # --- SUMMARY MERGE FIX ---
        merged_params = existing_summary.get("parameters", {})
        current_params = vars(args)
        
        # If this is an evaluation run, only update the eval-specific parameters
        if current_params.get("query_backend") == "none":
            merged_params["eval_backend"] = current_params.get("eval_backend")
            merged_params["eval_model"] = current_params.get("eval_model")
        else:
            # If this is a generation run, save all parameters
            merged_params.update(current_params)

        experiment_summary = {
            "parameters": merged_params, 
            "time_average": avg_times if avg_times else existing_summary.get("time_average"),
            "time_total": total_times if total_times else existing_summary.get("time_total"),
            "total_process_time_seconds": (time.perf_counter() - main_time) + existing_summary.get("total_process_time_seconds", 0),
            "evaluation_summary": aggregated_scores if aggregated_scores is not None else existing_summary.get("evaluation_summary", "No evaluation run")
        }
        # --- END SUMMARY MERGE FIX ---

        # 7. Save to summary.json
        logger.info("Saving full experiment summary to summary.json...")
        # We use save_outputs which handles JSON saving. 
        # Passing a dict (instead of a list) usually saves as a standard .json file.
        save_outputs(experiment_summary, args.log_name, "summary")

        # 8. Final Console Output
        logger.info(f"Average times (seconds): {avg_times}")
        logger.info(f"Total times (seconds): {total_times}")
        logger.info(f"Total process time (seconds): {experiment_summary['total_process_time_seconds']}")


if __name__ == "__main__":
    main()