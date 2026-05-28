"""Local embedding backend for medical factor memory retrieval."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


DEFAULT_LOCAL_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def _str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class LocalEmbeddingModel:
    def __init__(self) -> None:
        self.model_name = os.environ.get(
            "MEDICAL_LOCAL_EMBEDDING_MODEL",
            DEFAULT_LOCAL_EMBEDDING_MODEL,
        )
        device = os.environ.get("MEDICAL_LOCAL_EMBEDDING_DEVICE", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_length = int(os.environ.get("MEDICAL_LOCAL_EMBEDDING_MAX_LENGTH", "2048"))
        self.batch_size = int(os.environ.get("MEDICAL_LOCAL_EMBEDDING_BATCH_SIZE", "8"))
        self.query_instruction = os.environ.get(
            "MEDICAL_LOCAL_EMBEDDING_QUERY_INSTRUCTION",
            (
                "Instruct: Given a clinical factor-mining hypothesis, retrieve "
                "relevant prior medical symbolic factor memories, including "
                "successful and failed attempts.\nQuery: "
            ),
        )
        trust_remote_code = _str_to_bool(
            os.environ.get("MEDICAL_LOCAL_EMBEDDING_TRUST_REMOTE_CODE", "1")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer.padding_side = "left"
        dtype = torch.float32 if self.device.type == "cpu" else torch.float16
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        if is_query:
            texts = [self.query_instruction + text for text in texts]
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = self.model(**encoded)
                pooled = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
                pooled = F.normalize(pooled, p=2, dim=1)
            embeddings.extend(pooled.detach().cpu().float().numpy().tolist())
        return embeddings


@lru_cache(maxsize=1)
def get_local_embedding_model() -> LocalEmbeddingModel:
    return LocalEmbeddingModel()


def local_embed(texts: list[str], *, is_query: bool = False) -> list[list[float]]:
    return get_local_embedding_model().encode(texts, is_query=is_query)


def cosine_scores(query_embedding: list[float], doc_embeddings: list[list[float]]) -> np.ndarray:
    query = np.asarray(query_embedding, dtype=np.float32)
    docs = np.asarray(doc_embeddings, dtype=np.float32)
    query = query / max(float(np.linalg.norm(query)), 1e-12)
    doc_norms = np.linalg.norm(docs, axis=1, keepdims=True)
    docs = docs / np.maximum(doc_norms, 1e-12)
    return docs @ query
