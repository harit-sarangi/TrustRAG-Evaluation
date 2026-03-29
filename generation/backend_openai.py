import os
from typing import Literal
from loguru import logger
from openai import OpenAI

from .backend import GenerationBackend


class OpenAIBackend(GenerationBackend):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.openai = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/"),
            api_key=os.environ.get("OPENAI_API_KEY", None),
        )

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
        if "mini" in self.get_model():  # OpenAI's mini models does not support custom temperature
            temperature = 1.0

        for attempt in range(retries):
            completion = self.openai.chat.completions.create(
                model=self.get_model(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                response_format={"type": format},
                max_tokens=8_000,
                #Modifications by Harit
                logprobs=True,
                top_logprobs=1
                #till here
            )

            answer = completion.choices[0].message.content
            #Modifications by Harit
            logprobs = completion.choices[0].logprobs
            confidence = None
            if logprobs and hasattr(logprobs, "content") and logprobs.content:
                token_logprobs = [t.logprob for t in logprobs.content if t.logprob is not None]
                if token_logprobs:
                    confidence = sum(token_logprobs) / len(token_logprobs)
            #till here
            if answer:
                return {
                    "answer": answer, "confidence": confidence #here added the confidence
                }
            

            logger.debug(f"Empty answer received on attempt {attempt + 1}/{retries}, retrying...")

        if fail_after_retries:
            raise Exception("Could not generate a valid response after maximum retries.")

        logger.warning("Could not generate completion. Continue with default response.")
        return response_retries_exceeded
