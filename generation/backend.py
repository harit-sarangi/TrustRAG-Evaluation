from abc import ABC, abstractmethod
import json
import re
from loguru import logger
from typing import Literal


class GenerationBackend(ABC):
    """Base class for generation backends"""

    @abstractmethod
    def __init__(self, model_name: str):
        """Initialize the backend with the model"""
        self.model_name = model_name

    def get_model(self) -> str:
        return self.model_name

    @abstractmethod
    def chat_completions(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.1,
        format: Literal["text", "json_object"] = "text",
        **kwargs,
    ) -> str:
        """Generate chat completion with the given parameters"""
        pass

    def _validate_json(self, response: str, raise_on_error=True) -> str | None:
        try:
            if response.startswith("```"):
                code_block_match = re.search(r"^```(?:json)?(.+?)```", response, re.DOTALL)
                if code_block_match:
                    response = code_block_match.group(1).strip()
                    logger.trace(f"Extracted JSON from code block: {response}")

            _ = json.loads(response)
            return response
        except json.JSONDecodeError:
            if raise_on_error:
                raise ValueError(f"Failed to parse JSON object in response: {response}")

            logger.warning(f"Failed to parse JSON object in response: {response}")
            return None

    def __del__(self):
        """Clean up resources"""
        pass
