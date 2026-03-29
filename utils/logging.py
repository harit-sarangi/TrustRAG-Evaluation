import argparse
import os

from tqdm import tqdm
from loguru import logger

LOGS_DIR = "output/logs"


def compound_log_name(args: argparse.Namespace):
    log_name = f"{args.query_method}-"
    log_name += args.query_model.replace("/", "_").replace("-", "_").lower()
    log_name += f"-{args.query_backend}"
    if args.limit > 0:
        log_name += f"-{args.limit}x{args.repeat_times}"
    if args.ret_method and args.ret_method != "none":
        log_name += f"-{args.ret_method}-t{args.ret_top_k}"
    if args.rerank_model and args.rerank_model != "none":
        log_name += f"-{args.rerank_model.replace('/', '_').replace('-', '_').lower()}-t{args.rerank_top_k}"
    if args.invert:
        log_name += "-invert"
    if args.seed:
        log_name += f"-{args.seed}"
    return log_name


def file_formatter(record):
    record["fn_line"] = record["name"] + ":" + record["function"] + ":" + str(record["line"])
    format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {fn_line: <32} | "

    if record["extra"]:
        format += "{extra} "

    format += "{message}\n"

    if record["exception"]:
        format += "{exception}\n"

    return format


def setup_logging(experiment_name=None, log_dir=LOGS_DIR, file_level="INFO", console_level="INFO"):
    """
    Configure logging for experiments with both console and file output.

    Args:
        experiment_name: Name of the experiment for the log file
        log_dir: Directory to store log files
    """
    # Remove any existing handlers
    logger.remove()

    # Add console handler with a simple format
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format="<level>{level: <7}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> {extra} - <level>{message}</level>",
        level=console_level,
        colorize=True,
    )

    # Add file handler if experiment_name is provided
    if experiment_name and file_level is not None:
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, f"{experiment_name}.log")
        if os.path.exists(log_file):
            os.remove(log_file)

        logger.add(log_file, format=file_formatter, level=file_level)

    return logger
