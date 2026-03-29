import gc
from typing import List
from loguru import logger

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .backend import RerankingBackend


class TransformersBackend(RerankingBackend):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, device_map="auto", torch_dtype="auto").eval()

        device_type = next(iter(self.model.parameters())).device.type
        self.device = torch.device(device_type)
        if device_type != "cuda":
            logger.warning("No GPU available. Using CPU for LLM!")

    def rerank_passages(
        self,
        query: str,
        passages: List[str],
    ) -> List[float]:
        pairs = [(query, p) for p in passages]

        with torch.inference_mode():
            inputs = self.tokenizer([q for q, _ in pairs], [p for _, p in pairs], padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
            return scores.cpu().tolist()

    def __del__(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if hasattr(self, "device"):
            del self.device
        torch.cuda.empty_cache()  # Try to release GPU memory
        gc.collect()
