import unittest
from loguru import logger

from .backend_transformers import TransformersBackend

MODEL_NAME = "tiiuae/Falcon3-10B-Instruct"
# MODEL_NAME = "google/gemma-3-27b-it"


class TestTransformersBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = TransformersBackend(MODEL_NAME)
        logger.info(f"Loaded model: {MODEL_NAME}")

    @classmethod
    def tearDownClass(cls):
        del cls.backend

    def test_chat_completions(self):
        answer = self.backend.chat_completions(prompt="What is a capital of Germany? Answer only with a city name.")
        logger.info(f"Answer: {answer}")

        self.assertEqual(answer, "Berlin")

    def test_chat_completions_json(self):
        answer = self.backend.chat_completions(prompt='What is a capital of Germany? Answer in a format {"city": "name_of_the_city"}.', format="json_object")
        logger.info(f"Answer: {answer}")

        self.assertEqual(answer["city"], "Berlin")


if __name__ == "__main__":
    unittest.main(verbosity=2)
