import os
from multiprocessing.pool import ThreadPool
from typing import List, Tuple
from loguru import logger
from pinecone import Pinecone

from backend import RetrievalBackend
from encoder import TransformersEncoder


class PineconeBackend(RetrievalBackend):
    def __init__(self):
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "fineweb-10bt")
        self.namespace = os.getenv("PINECONE_NAMESPACE", "default")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_APIKEY"))
        self.index = self.pc.Index(name=self.index_name)
        self.encoder = TransformersEncoder(model_name="intfloat/e5-base-v2", max_tokens=512)

    def search(self, query: str, top_k: int = 10) -> Tuple[List[str], List[str]]:
        embedded_query = self.encoder.encode_query(query)

        response = self.index.query(
            vector=embedded_query,
            top_k=top_k,
            include_values=False,
            namespace=self.namespace,
            include_metadata=True,
        )
        logger.trace(f"Retrieved {len(response['matches'])} documents from Pinecone")

        doc_ids = []
        ret_docs = []
        for matches in response["matches"]:
            doc_ids.append(matches["metadata"]["doc_id"])
            ret_docs.append(matches["metadata"]["text"])
        return doc_ids, ret_docs

    def search_batch(self, queries: List[str], top_k: int = 10, n_parallel: int = 10) -> List[Tuple[List[str], List[str]]]:
        """Batch query a Pinecone index and return the results."""
        embeds = self.encoder.encode_queries(queries)

        pool = ThreadPool(n_parallel)
        responses = pool.map(lambda x: self.index.query(vector=x, top_k=top_k, include_values=False, namespace=self.namespace, include_metadata=True), embeds)

        results = []
        for response in responses:
            logger.trace(f"Retrieved {len(response['hits']['hits'])} documents from OpenSearch")

            doc_ids = []
            ret_docs = []
            for matches in response["matches"]:
                doc_ids.append(matches["_source"]["doc_id"])
                ret_docs.append(matches["_source"]["text"])
            results.append((doc_ids, ret_docs))
        return results
