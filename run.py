import sys
from main import main


def build_sys_argv(test_params, original_argv):
    argv = ["main.py"]
    for key, value in test_params.items():
        arg_key = f"--{key}"
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                argv.append(arg_key)
        else:
            argv.extend([arg_key, str(value)])
    argv.extend(original_argv[1:])  # Append any additional arguments passed to the script
    return argv


def run(test_params):
    original_argv = sys.argv.copy()
    sys.argv = build_sys_argv(test_params, original_argv)
    main()
    sys.argv = original_argv


test_params = {
    # model parameters
    # "query_method": "simple",
    # "query_model": "tiiuae/Falcon3-10B-Instruct",
    # "query_backend": "local",
    # retriever methods
    # "ret_method": "top_k",
    "ret_top_k": 5,
    # reranker methods
    # "rerank_model": "BAAI/bge-reranker-v2-m3",
    # "rerank_top_k": 5,
    # "invert": True,
    # other parameters
    "limit": 10,  # number of queries
    "eval_backend": "openai",
    "eval_model": "gemma3:27b",
    "log_name": None,  # will override the generated log name
}

# Make sure to enable only one of them at the same time
# Otherwise, they are going to be run in sequence for a long time

# Option 1 - Single run with default parameters defined in test_params
# run(test_params)

# Option 2 - Override some parameters
test_params["dataset_path"] = "datasets/dev.jsonl"
# test_params["query_method"] = "instruct"
test_params["rerank_backend"] = "none"  # no reranking
test_params["very_verbose"] = True
# test_params["eval_model"] = None  # but also automatically switched off when there is no `answer` in the dataset
run(test_params)

# Option 3 - Run multiple methods sequentially (each of them will write results and logs into separate files)
# for ret_method in ["none", "top_k"]:
#     for query_method in ["none", "trustrag", "astute", "instruct"]:
#         test_params["ret_method"] = ret_method
#         test_params["query_method"] = query_method
#         run(test_params)
