from typing import List, Literal
import torch
from transformers import AutoModel, AutoTokenizer


class TransformersEncoder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",    #model_name: str = "intfloat/e5-base-v2",
        max_tokens: int = 512,
        pooling: Literal["cls", "avg"] = "avg",
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.pooling = pooling
        self.normalize = normalize

        self.tokenizer = self.get_tokenizer(self.model_name)
        self.model = self.get_model(self.model_name)

    def encode_query(self, query: str) -> List[float]:
        return self.encode(query, prefix="query: ")

    def encode_queries(self, queries: List[str]) -> List[float]:
        return self.encode_batch(queries, prefix="query: ")

    def encode_passages(self, passages: List[str]) -> List[List[float]]:
        return self.encode_batch(passages, prefix="passage: ")

    def encode(self, passage: str, prefix: str = None) -> List[float]:
        return self.encode_batch([passage], prefix=prefix)[0]

    def encode_batch(self, passages: List[str], prefix: str = None) -> List[List[float]]:
        if prefix:
            passages = [f"{prefix} {passage}" for passage in passages]

        with torch.no_grad():
            # FIX: Changed 'max_length=max_tokens' to 'max_length=self.max_tokens'
            encoded = self.tokenizer(passages, padding=True, return_tensors="pt", truncation="longest_first", max_length=self.max_tokens)
            encoded = encoded.to(self.model.device)
            model_out = self.model(**encoded)

            match self.pooling:
                case "cls":
                    embeddings = model_out.last_hidden_state[:, 0]
                case "avg":
                    embeddings = self.average_pool(model_out.last_hidden_state, encoded["attention_mask"])

            if self.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()

    @staticmethod
    def get_tokenizer(model_name):
        return AutoTokenizer.from_pretrained(model_name)

    @staticmethod
    def get_model(model_name):
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if torch.backends.mps.is_available():
            model = model.to("mps")
        elif torch.cuda.is_available():
            model = model.to("cuda")
        else:
            model = model.to("cpu")
        return model

    @staticmethod
    def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]