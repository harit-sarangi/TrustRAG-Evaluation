from typing import List, Literal
import torch
from functools import cache
from transformers import AutoModel, AutoTokenizer

EMBEDDINGS_MODEL_NAME = "intfloat/e5-base-v2"
EMBEDDINGS_DIMENSION = 768
CHUNK_TOKEN_SIZE = 512


@cache
def has_mps():
    return torch.backends.mps.is_available()


@cache
def has_cuda():
    return torch.cuda.is_available()


@cache
def get_tokenizer(model_name: str = EMBEDDINGS_MODEL_NAME):
    print(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


@cache
def get_model(model_name: str = EMBEDDINGS_MODEL_NAME):
    print(f"Loading model for {model_name}")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    if has_mps():
        model = model.to("mps")
    elif has_cuda():
        model = model.to("cuda")
    else:
        model = model.to("cpu")
    return model


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def batch_embed(
    passages: List[str],
    query_prefix: str = "query: ",
    model_name: str = EMBEDDINGS_MODEL_NAME,
    pooling: Literal["cls", "avg"] = "avg",
    normalize: bool = True,
    max_tokens: int = CHUNK_TOKEN_SIZE,
) -> List[List[float]]:
    with_prefixes = [" ".join([query_prefix, passage]) for passage in passages]
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)

    with torch.no_grad():
        encoded = tokenizer(with_prefixes, padding=True, return_tensors="pt", truncation="longest_first", max_length=max_tokens)
        encoded = encoded.to(model.device)
        model_out = model(**encoded)
        match pooling:
            case "cls":
                embeddings = model_out.last_hidden_state[:, 0]
            case "avg":
                embeddings = average_pool(model_out.last_hidden_state, encoded["attention_mask"])

        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()


def embed_query(query: str, *args, **kwargs) -> list[float]:
    return batch_embed([query], *args, **kwargs)[0]
