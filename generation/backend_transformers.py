import gc
from typing import Literal
from loguru import logger

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .backend import GenerationBackend


class TransformersBackend(GenerationBackend):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()

        device_type = next(iter(self.model.parameters())).device.type
        self.device = torch.device(device_type)
        if device_type != "cuda":
            logger.warning("No GPU available. Using CPU for LLM!")

    def chat_completions(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        temperature: float = 0.1,  # High for creativity, low for technical/precise answer
        format: Literal["text", "json_object"] = "text",
    ) -> str:
        if format == "json_object":
            system_prompt += " Please return the answer strictly in valid JSON format, without additional explanations."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        with torch.inference_mode():
            inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)

            outputs = self.model.generate(
                inputs,
                temperature=temperature,
                max_length=31000,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            prompt_length = inputs.shape[1]
            response = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

            if format == "json_object":
                return self._validate_json(response)
            else:
                return response

    def __del__(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if hasattr(self, "device"):
            del self.device
        torch.cuda.empty_cache()  # Try to release GPU memory
        gc.collect()
