from typing import List, Tuple
from loguru import logger

from .backend import RerankingBackend


def get_backend(backend_type: str, model_name: str) -> RerankingBackend | None:
    if backend_type == "local":
        from .backend_transformers import TransformersBackend

        return TransformersBackend(model_name)
    elif not model_name or model_name == "none":
        return None
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}. Supported types are 'local'.")


def rerank_docs(be: RerankingBackend, query: str, doc_ids: List[str], doc_passages: List[str], top_k: int = None) -> Tuple[List[str], List[str]]:
    scores = be.rerank_passages(query, doc_passages)
    logger.debug(f"Received scores: {scores}")

    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    if top_k:  # Limit to top_k results
        sorted_indices = sorted_indices[:top_k]

    logger.debug(f"Selected indices: {sorted_indices}")
    doc_ids = [doc_ids[i] for i in sorted_indices]
    doc_passages = [doc_passages[i] for i in sorted_indices]
    return doc_ids, doc_passages
