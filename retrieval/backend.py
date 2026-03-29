from abc import ABC, abstractmethod
from typing import List, Tuple


class RetrievalBackend(ABC):
    """Base class for retrieval backends"""

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> Tuple[List[str], List[str]]:
        """Query the backend to retrieve `topK` documents to the input query"""
        pass

    @abstractmethod
    def search_batch(self, queries: List[str], top_k: int = 10) -> List[Tuple[List[str], List[str]]]:
        """Query the backend to retrieve `topK` documents to the input queries"""
        pass

    def __del__(self):
        """Clean up resources"""
        pass
