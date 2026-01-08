import os
import json
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm


class SparseRetrieval:
    """
    Sparse Retrieval with BM25 algorithm.
    BM25 is a probabilistic retrieval function that ranks passages based on query term frequency and document length.
    """
    
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data",
        context_path: Optional[str] = "wikipedia_documents.json",
    ):
        """
        Initialize BM25 retrieval.
        
        Args:
            tokenize_fn: Tokenization function (e.g., tokenizer.tokenize)
            data_path: Path to data directory
            context_path: Wikipedia documents JSON filename
        """
        self.data_path = data_path
        self.tokenize_fn = tokenize_fn
        
        # Load contexts
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)
        
        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print(f"Loaded {len(self.contexts)} unique contexts")
        
        # Tokenize all contexts
        print("Tokenizing contexts for BM25...")
        self.tokenized_contexts = [
            self.tokenize_fn(context) for context in tqdm(self.contexts, desc="Tokenizing")
        ]
        
        # Initialize BM25
        print("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_contexts)
        print("BM25 index built successfully")
    
    def retrieve(
        self, 
        query_or_dataset: Union[str, Dataset], 
        topk: int = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Retrieve top-k passages for query/queries using BM25.
        
        Args:
            query_or_dataset: Single query string or HuggingFace Dataset
            topk: Number of top passages to retrieve
            
        Returns:
            If single query: Tuple of (scores, passages)
            If dataset: DataFrame with retrieved contexts
        """
        if isinstance(query_or_dataset, str):
            return self._retrieve_single(query_or_dataset, topk)
        elif isinstance(query_or_dataset, Dataset):
            return self._retrieve_dataset(query_or_dataset, topk)
        else:
            raise ValueError("query_or_dataset must be str or Dataset")
    
    def _retrieve_single(self, query: str, topk: int) -> Tuple[List[float], List[str]]:
        """Retrieve for a single query."""
        tokenized_query = self.tokenize_fn(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_indices = np.argsort(doc_scores)[::-1][:topk]
        top_k_scores = doc_scores[top_k_indices].tolist()
        top_k_passages = [self.contexts[idx] for idx in top_k_indices]
        
        print(f"[Search query]\n{query}\n")
        for i, (score, passage) in enumerate(zip(top_k_scores, top_k_passages)):
            print(f"Top-{i+1} passage with score {score:.4f}")
            print(f"{passage[:200]}...\n")
        
        return top_k_scores, top_k_passages
    
    def _retrieve_dataset(self, dataset: Dataset, topk: int) -> pd.DataFrame:
        """Retrieve for a dataset of queries."""
        total = []
        
        for example in tqdm(dataset, desc="BM25 retrieval"):
            query = example["question"]
            tokenized_query = self.tokenize_fn(query)
            doc_scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k indices
            top_k_indices = np.argsort(doc_scores)[::-1][:topk]
            
            # Concatenate retrieved contexts
            retrieved_context = " ".join([self.contexts[idx] for idx in top_k_indices])
            
            tmp = {
                "question": query,
                "id": example["id"],
                "context": retrieved_context,
            }
            
            # Include original context and answers if available
            if "context" in example.keys() and "answers" in example.keys():
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            
            total.append(tmp)
        
        return pd.DataFrame(total)
    
    def get_relevant_doc(self, query: str, k: int = 1) -> Tuple[List[float], List[int]]:
        """
        Get top-k relevant document indices and scores for a single query.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            Tuple of (scores, indices)
        """
        tokenized_query = self.tokenize_fn(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_indices = np.argsort(doc_scores)[::-1][:k]
        top_k_scores = doc_scores[top_k_indices].tolist()
        
        return top_k_scores, top_k_indices.tolist()
    
    def get_relevant_doc_bulk(
        self, 
        queries: List[str], 
        k: int = 1
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """
        Get top-k relevant document indices and scores for multiple queries.
        
        Args:
            queries: List of query strings
            k: Number of documents to retrieve per query
            
        Returns:
            Tuple of (scores_list, indices_list)
        """
        doc_scores_list = []
        doc_indices_list = []
        
        for query in tqdm(queries, desc="BM25 bulk retrieval"):
            tokenized_query = self.tokenize_fn(query)
            doc_scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k indices
            top_k_indices = np.argsort(doc_scores)[::-1][:k]
            top_k_scores = doc_scores[top_k_indices].tolist()
            
            doc_scores_list.append(top_k_scores)
            doc_indices_list.append(top_k_indices.tolist())
        
        return doc_scores_list, doc_indices_list