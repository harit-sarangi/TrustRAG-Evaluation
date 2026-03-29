from typing import Callable, List, Tuple
from loguru import logger

from .backend import RetrievalBackend


def get_backend(backend_type: str) -> RetrievalBackend:
    if backend_type == "opensearch":
        from .backend_opensearch import OpenSearchBackend

        return OpenSearchBackend()
    elif backend_type == "pinecone":
        from .backend_pinecone import PineconeBackend

        return PineconeBackend()
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}. Supported types are 'opensearch' and 'pinecone'.")


RetrievalFunction = Callable[[RetrievalBackend, str, int], Tuple[List[str], List[str]]]


def empty(bg: RetrievalBackend, query: str, ret_top_k: int) -> Tuple[List[str], List[str]]:
    """
    This function does not perform any retrieval and returns empty lists.
    The purpose of this function is to provide a reference for the retrieval function.
    """
    return [], []


def top_k(bg: RetrievalBackend, query: str, top_k: int) -> Tuple[List[str], List[str]]:
    return bg.search(query, top_k=top_k)
