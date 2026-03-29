# TrustRAG: Multi-Sample Semantic Verification and Confidence Scoring
*(Built upon the AI Lab RAG Baseline authored by Oleh Astappiev/L3S - https://git.l3s.uni-hannover.de/astappiev/labrag)*

Author: Harit Sarangi

Mentor: Oleh Astappiev

This repository contains a reference RAG implementation, heavily modified to evaluate the **TrustRAG** architecture—a pipeline designed to reduce hallucinations in Retrieval-Augmented Generation by calculating an internal semantic confidence score via multi-sample verification and token probability analysis.

## Overview

Standard generative models often force a hallucinated consensus when presented with noisy or conflicting retrieved context. TrustRAG addresses this by replacing standard zero-shot generation with a three-step verification process:
1. **Multi-Sampling:** Generates three independent answer samples based on the retrieved context.
2. **Semantic Consensus:** Cross-examines the samples to find semantic overlap.
3. **Confidence Scoring:** Assigns a confidence score by evaluating semantic variance and extracting average token log-probabilities. High variance or dropping log-probabilities flag the context as adversarial or insufficient.

### Evaluated Models
* **Baseline:** Falcon3:7B (Local)
* **SOTA (State of the Art) 1:** Llama 3.1-8B (API)
* **SOTA (State of the Art) 2:** GLM 4.7 Flash (API) - *Achieved highest faithfulness and correctness.*

## System Architecture (AI Lab Baseline + TrustRAG)

The pipeline consists of three main parts:
* **Retrieval:** OpenSearch index (`fineweb-10b`), fetching `top_k=10`.
* **Reranking:** Local transformers (e.g., `BAAI/bge-reranker-v2-m3`), filtering to `top_k=3`.
* **Generation & Evaluation:** Hosted models via API (Neumann server) or local models via transformers. Evaluated via LLM-as-a-judge (`gemma3:27b-it-fp16`).

## Repository Structure

```text
labrag-main/
├── datasets/            # Contains evaluation datasets (e.g., comp200.jsonl, dev.jsonl)
├── evaluation/          # LLM-as-a-judge scoring logic (Correctness & Faithfulness)
├── generation/          # TrustRAG multi-sample and confidence scoring algorithms
├── output/              # Summary, Execution logs and final answers.jsonl results
├── reranking/           # Cross-encoder reranking logic
├── retrieval/           # OpenSearch backend integration
├── utils/               # Helper functions (time tracking, validation, I/O)
├── .env.example         # Template for required environment variables
├── env.yml              # Conda environment specifications
├── main.py              # Primary execution script
└── run.py               # Alternative execution/helper script
```

## Setup Environment
```bash
conda env create -n raglab -f env.yml
conda activate raglab
```
## Secrets
Make sure you have .env in the directory from which you run the code. Copy .env.example to .env and fill in the values.

## Usage
All supported parameters can be found in main.py or printed in terminal by python main.py -h.

## Using run.py
Modify parameters in run.py to your liking. Run it as follows:
```bash
python run.py
# or on cluster with GPU
srun --pty python run.py
```
## Using main.sh / main.py directly
You can also run the script directly. But you need to specify all the parameters as arguments.
For a basic dev run:
```bash
python main.py --dataset_path datasets/dev.jsonl
```
## For the full TrustRAG Evaluation run:
To run the full evaluation pipeline using the Baseline or SOTA (Falcon3:7B, Llama 3.1-8B, GLM 4.7 Flash) configuration with built-in retry logic for API timeouts, execute:
```bash
python main.py \
  --dataset_path datasets/comp200.jsonl \
  --limit 100 \
  --ret_backend opensearch \
  --ret_top_k 10 \
  --rerank_backend local \
  --rerank_model BAAI/bge-reranker-v2-m3 \ #You can change Reranker model name here
  --rerank_top_k 3 \
  --query_method trustrag \
  --query_model glm-4.7-flash:latest \ #You can change Generator model name here (Falcon3:7B, Llama 3.1-8B, GLM 4.7 Flash)
  --eval_backend openai \
  --eval_model gemma3:27b-it-fp16 \ #You can change Evaluator model name here (Gemma2:2b, Gemma3:27b etc.)
  --retry_times 5 \
  --verbose True
  ```
  (Note: Due to the rigorous multi-sample consensus required by GLM 4.7 Flash, queries with highly conflicting context may trigger extended reasoning loops (900s - 3600s per query) and utilize the retry mechanism to bypass API timeouts.)

  ## References & Theoretical Foundations
This implementation builds upon the following research:

-> Token Probability Confidence: DRAGIN: Dynamic Retrieval Augmented Generation based on the Real-time Information Needs of LLMs (arXiv:2403.10081).

-> LLM-as-a-Judge Evaluation: SIGIR 2025 – LiveRAG Challenge Report (arXiv:2507.04942).

-> Semantic Verification: Survey on Confidence Scoring and Semantic Verification in LLMs (arXiv:2508.15437).
