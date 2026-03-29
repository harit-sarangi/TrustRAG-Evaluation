from abc import ABC, abstractmethod
from typing import List


class RerankingBackend(ABC):
    """Base class for reranking backends"""

    @abstractmethod
    def __init__(self, model_name: str):
        """Initialize the backend with the model"""
        pass

    @abstractmethod
    def rerank_passages(
        self,
        query: str,
        passages: List[str],
    ) -> List[float]:
        """Rerank passages based on the query and return scores"""
        pass

    @abstractmethod
    def __del__(self):
        """Clean up resources"""
        pass
