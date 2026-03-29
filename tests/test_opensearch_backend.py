import unittest
from dotenv import load_dotenv
from loguru import logger

from retrieval.backend_opensearch import OpenSearchBackend


class TestOpenSearchBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = OpenSearchBackend()

    @classmethod
    def tearDownClass(cls):
        del cls.backend

    def test_search(self):
        doc_ids, doc_texts = self.backend.search("What is a second brain?")

        for i, (id, text) in enumerate(zip(doc_ids, doc_texts)):
            logger.info(f"{str(i + 1).zfill(2)}: {id} - {text}")

        self.assertEqual(len(doc_ids), len(doc_texts))

    def test_search_embed(self):
        doc_ids, doc_texts = self.backend.search_embed("What is a second brain?")

        for i, (id, text) in enumerate(zip(doc_ids, doc_texts)):
            logger.info(f"{str(i + 1).zfill(2)}: {id} - {text}")

        self.assertEqual(len(doc_ids), len(doc_texts))

    def test_search_batch(self):
        queries = [
            "What is a second brain?",
            "What is the purpose of a second brain?",
            "How to build a second brain?",
        ]
        results = self.backend.search_batch(queries, top_k=3)

        for i, (doc_ids, doc_texts) in enumerate(results):
            logger.info(f"Query: {queries[i]}")
            for j, (id, text) in enumerate(zip(doc_ids, doc_texts)):
                logger.info(f"{str(j + 1).zfill(2)}: {id} - {text}")

        self.assertEqual(len(results), len(queries))

    def test_find_by_doc_id(self):
        chunks = self.backend.find_by_doc_id("<urn:uuid:a919e400-a474-4eb7-a7dd-916efdc35616>")

        logger.info(f"Found {len(chunks)} chunks with doc_id <urn:uuid:a919e400-a474-4eb7-a7dd-916efdc35616>")
        for i, chunk in enumerate(chunks):
            logger.info(f"{str(i + 1).zfill(2)}: {chunk}")

        self.assertEqual(len(chunks), 3)


if __name__ == "__main__":
    load_dotenv(verbose=True)
    unittest.main(verbosity=2)
