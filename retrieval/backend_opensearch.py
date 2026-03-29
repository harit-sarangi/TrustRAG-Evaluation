import os
from typing import List, Tuple
from loguru import logger
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection

from .embeddings import embed_query
from .backend import RetrievalBackend


class OpenSearchBackend(RetrievalBackend):
    def __init__(self):
        opensearch_host = os.getenv("OPENSEARCH_HOST", "labrag.kbs.uni-hannover.de")
        opensearch_port = os.getenv("OPENSEARCH_PORT", 443)
        opensearch_index_name = os.getenv("OPENSEARCH_INDEX_NAME", "fineweb-10bt")

        if "amazonaws.com" in opensearch_host:
            import boto3

            aws_access_key_id = os.getenv("AWS_ACCESS_KEY", os.getenv("AWS_ACCESS_KEY_ID"))
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = os.getenv("AWS_REGION_NAME", "us-east-1")

            credentials = boto3.Session(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key).get_credentials()
            auth = AWSV4SignerAuth(credentials, region=aws_region)
        else:
            opensearch_user = os.getenv("OPENSEARCH_USER", "readall")
            opensearch_password = os.getenv("OPENSEARCH_PASSWORD", "readall")
            auth = (opensearch_user, opensearch_password)

        self.index_name = opensearch_index_name
        self.client = OpenSearch(
            hosts=[{"host": opensearch_host, "port": opensearch_port}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            connection_class=RequestsHttpConnection,
        )

    def search(self, query: str, top_k: int = 10) -> Tuple[List[str], List[str]]:
        response = self.client.search(index=self.index_name, body={"query": {"match": {"text": query}}, "size": top_k})
        logger.trace(f"Retrieved {len(response['hits']['hits'])} documents from OpenSearch")

        doc_ids = []
        ret_docs = []
        for matches in response["hits"]["hits"]:
            doc_ids.append(matches["_source"]["doc_id"])
            ret_docs.append(matches["_source"]["text"])
        return doc_ids, ret_docs

    def search_embed(self, query, top_k: int = 10, timeout: str = "30s") -> dict:
        """Make a query to OpenSearch using embeddings."""
        query_vector = embed_query(query, query_prefix="query: ")

        results = self.client.search(
            index=self.index_name,
            body={
                "query": {
                    "knn": {"embedding": {"vector": query_vector, "k": top_k}},
                },
            },
            timeout=timeout,
        )
        return results

    def search_batch(self, queries: List[str], top_k: int = 10) -> List[Tuple[List[str], List[str]]]:
        """Sends a list of queries to OpenSearch and returns the results. Configuration of Connection Timeout might be needed for serving large batches of queries"""
        request = []
        for query in queries:
            req_head = {"index": self.index_name}
            req_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text"],
                    }
                },
                "size": top_k,
            }
            request.extend([req_head, req_body])

        responses = self.client.msearch(body=request)

        results = []
        for response in responses["responses"]:
            logger.trace(f"Retrieved {len(response['hits']['hits'])} documents from OpenSearch")

            doc_ids = []
            ret_docs = []
            for matches in response["hits"]["hits"]:
                doc_ids.append(matches["_source"]["doc_id"])
                ret_docs.append(matches["_source"]["text"])
            results.append((doc_ids, ret_docs))
        return results

    def find_by_doc_id(self, doc_id: str) -> List[str]:
        response = self.client.search(index=self.index_name, body={"query": {"term": {"doc_id": doc_id}}})
        logger.trace(f"Retrieved {len(response['hits']['hits'])} passages to the doc_id {doc_id} from OpenSearch")

        return [matches["_source"]["text"] for matches in response["hits"]["hits"]]
