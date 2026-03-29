from typing import Literal
from ollama import Client
from loguru import logger
import os

from .backend import GenerationBackend


class OllamaBackend(GenerationBackend):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = Client(host=os.environ.get("OLLAMA_HOST", "http://gpunode07.kbs:11434"))

    def chat_completions(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.1,
        format: Literal["text", "json_object"] = "text",
        retries: int = 5,
        fail_after_retries: bool = False,
        response_retries_exceeded: str = "I apologise, I can't help you there.",
    ) -> str:
        for attempt in range(retries):  # this retries is only protecting us from an empty answer
            response = self.client.chat(
                model=self.get_model(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                format="json" if format == "json_object" else "",
                options={
                    "temperature": temperature,
                    "num_ctx": 31000,
                },
            )

            answer = response["message"]["content"]
            if answer:
                return answer

            logger.debug(f"Empty answer received on attempt {attempt + 1}/{retries}, retrying...")

        if fail_after_retries:
            raise Exception("Could not generate a valid response after maximum retries.")

        logger.warning("Could not generate completion. Continue with default response.")
        return response_retries_exceeded
