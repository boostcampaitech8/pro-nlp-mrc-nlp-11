
import os
import json
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


class DenseRetrieval:
    """
    Dense Passage Retrieval (DPR) with dual encoders.
    Loads trained question/passsage encoders and performs dot-product search.
    """

    def __init__(
        self,
        model_name_or_path: str,
        data_path: Optional[str] = "../data",
        context_path: Optional[str] = "wikipedia_documents.json",
        q_encoder_path: Optional[str] = None,
        p_encoder_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 64,
        max_q_length: int = 64,
        max_p_length: int = 256,
    ):
        self.model_name_or_path = model_name_or_path
        self.data_path = data_path
        self.q_encoder_path = q_encoder_path or model_name_or_path
        self.p_encoder_path = p_encoder_path or model_name_or_path
        self.batch_size = batch_size
        self.max_q_length = max_q_length
        self.max_p_length = max_p_length

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load contexts
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print(f"Loaded {len(self.contexts)} unique contexts for DPR")

        # Tokenizer and encoders
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        
        # Load question encoder
        print(f"Loading question encoder from: {self.q_encoder_path}")
        # Verify the model path exists (for local paths)
        if os.path.isdir(self.q_encoder_path):
            if not os.path.exists(os.path.join(self.q_encoder_path, "config.json")):
                print(f"WARNING: No config.json found in {self.q_encoder_path}")
                print(f"Falling back to base model: {self.model_name_or_path}")
                self.q_encoder_path = self.model_name_or_path
        
        self.q_encoder = AutoModel.from_pretrained(self.q_encoder_path).to(self.device)
        
        # Load passage encoder
        print(f"Loading passage encoder from: {self.p_encoder_path}")
        # Verify the model path exists (for local paths)
        if os.path.isdir(self.p_encoder_path):
            if not os.path.exists(os.path.join(self.p_encoder_path, "config.json")):
                print(f"WARNING: No config.json found in {self.p_encoder_path}")
                print(f"Falling back to base model: {self.model_name_or_path}")
                self.p_encoder_path = self.model_name_or_path
        
        self.p_encoder = AutoModel.from_pretrained(self.p_encoder_path).to(self.device)
        
        self.q_encoder.eval()
        self.p_encoder.eval()
        print("Dense retrievers loaded successfully")

        # Passage embeddings cache
        self.p_embedding: Optional[np.ndarray] = None

    def _encode_passages(self, passages: List[str]) -> np.ndarray:
        """Encode passages with the passage encoder."""
        all_embs = []
        with torch.no_grad():
            for start in tqdm(range(0, len(passages), self.batch_size), desc="Encode passages"):
                batch = passages[start : start + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_p_length,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.p_encoder(**inputs)
                cls = outputs.last_hidden_state[:, 0, :]
                cls = F.normalize(cls, p=2, dim=-1)
                all_embs.append(cls.cpu().numpy().astype(np.float32))
        return np.concatenate(all_embs, axis=0)

    def _encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries with the question encoder."""
        all_embs = []
        with torch.no_grad():
            for start in tqdm(range(0, len(queries), self.batch_size), desc="Encode queries"):
                batch = queries[start : start + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_q_length,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.q_encoder(**inputs)
                cls = outputs.last_hidden_state[:, 0, :]
                cls = F.normalize(cls, p=2, dim=-1)
                all_embs.append(cls.cpu().numpy().astype(np.float32))
        return np.concatenate(all_embs, axis=0)

    def get_dense_embedding(self):
        """Pre-compute and cache passage embeddings for fast retrieval."""
        if self.p_embedding is None:
            self.p_embedding = self._encode_passages(self.contexts)
            print(f"Built dense passage embeddings: {self.p_embedding.shape}")
        return self.p_embedding

    def get_relevant_doc(self, query: str, k: int = 1) -> Tuple[List[float], List[int]]:
        """Get top-k relevant documents for a single query."""
        if self.p_embedding is None:
            self.get_dense_embedding()

        q_emb = self._encode_queries([query])  # (1, dim)
        scores = np.dot(self.p_embedding, q_emb.squeeze())
        top_k_indices = np.argsort(scores)[::-1][:k]
        top_k_scores = scores[top_k_indices].tolist()
        return top_k_scores, top_k_indices.tolist()

    def get_relevant_doc_bulk(
        self,
        queries: List[str],
        k: int = 1,
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """Get top-k relevant documents for multiple queries."""
        if self.p_embedding is None:
            self.get_dense_embedding()

        q_embs = self._encode_queries(queries)  # (B, dim)
        scores = np.matmul(q_embs, self.p_embedding.T)  # (B, num_docs)

        doc_scores_list: List[List[float]] = []
        doc_indices_list: List[List[int]] = []

        for row in scores:
            top_k_indices = np.argsort(row)[::-1][:k]
            doc_indices_list.append(top_k_indices.tolist())
            doc_scores_list.append(row[top_k_indices].tolist())

        return doc_scores_list, doc_indices_list

    def retrieve(
        self,
        query_or_dataset: Union[str, Dataset],
        topk: int = 1,
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """Retrieve top-k passages using dense embeddings."""
        if isinstance(query_or_dataset, str):
            scores, indices = self.get_relevant_doc(query_or_dataset, k=topk)
            passages = [self.contexts[i] for i in indices]

            print(f"[Dense Search query]\n{query_or_dataset}\n")
            for i, (score, passage) in enumerate(zip(scores, passages)):
                print(f"Top-{i+1} passage with score {score:.4f}")
                print(f"{passage[:200]}...\n")

            return scores, passages

        elif isinstance(query_or_dataset, Dataset):
            total = []
            scores_list, indices_list = self.get_relevant_doc_bulk(
                query_or_dataset["question"], k=topk
            )

            for example, doc_indices in zip(query_or_dataset, indices_list):
                retrieved_context = " ".join([self.contexts[idx] for idx in doc_indices])
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": retrieved_context,
                }

                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]

                total.append(tmp)

            return pd.DataFrame(total)

        else:
            raise ValueError("query_or_dataset must be str or Dataset")
