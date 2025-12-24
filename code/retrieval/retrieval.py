import pandas as pd
from typing import List, Optional, Tuple, Union
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import numpy as np
from datasets import Dataset
from .Sparse_retrieval import SparseRetrieval
from .Dense_retrieval import DenseRetrieval

class Retrieval:
    """
    Base Retrieval class with hybrid retrieval support.
    Can combine Sparse (BM25) and Dense (DPR) retrieval methods.
    """
    
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data",
        context_path: Optional[str] = "wikipedia_documents.json",
        use_sparse: bool = True,
        use_dense: bool = False,
        sparse_weight: float = 0.5,
        dense_weight: float = 0.5,
        dense_model_path: Optional[str] = None,
        q_encoder_path: Optional[str] = None,
        p_encoder_path: Optional[str] = None,
    ):
        """
        Initialize Retrieval with optional hybrid support.
        
        Args:
            tokenize_fn: Tokenization function
            data_path: Path to data directory
            context_path: Wikipedia documents JSON filename
            use_sparse: Use BM25 sparse retrieval
            use_dense: Use DPR dense retrieval
            sparse_weight: Weight for sparse scores (used if both enabled)
            dense_weight: Weight for dense scores (used if both enabled)
            dense_model_path: Base model for DPR (e.g., "klue/bert-base")
            q_encoder_path: Path to trained question encoder
            p_encoder_path: Path to trained passage encoder
        """
        self.use_sparse = use_sparse
        self.use_dense = use_dense
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        
        if not use_sparse and not use_dense:
            raise ValueError("At least one of use_sparse or use_dense must be True")
        
        # Initialize sparse retriever
        self.sparse_retriever = None
        if use_sparse:
            self.sparse_retriever = SparseRetrieval(
                tokenize_fn=tokenize_fn,
                data_path=data_path,
                context_path=context_path,
            )
        
        # Initialize dense retriever (from dense_retrival.py)
        self.dense_retriever = None
        if use_dense:
            self.dense_retriever = DenseRetrieval(
                model_name_or_path=dense_model_path or "klue/bert-base",
                data_path=data_path,
                context_path=context_path,
                q_encoder_path=q_encoder_path,
                p_encoder_path=p_encoder_path,
            )
            # Build embeddings for dense retrieval
            self.dense_retriever.get_dense_embedding()
        
        # Get contexts from whichever retriever is available
        if self.sparse_retriever:
            self.contexts = self.sparse_retriever.contexts
        elif self.dense_retriever:
            self.contexts = self.dense_retriever.contexts
        
        # Determine mode
        if use_sparse and use_dense:
            self.mode = "hybrid"
            print(f"Initialized Hybrid Retrieval (Sparse weight={sparse_weight}, Dense weight={dense_weight})")
        elif use_sparse:
            self.mode = "sparse"
            print("Initialized Sparse Retrieval (BM25)")
        else:
            self.mode = "dense"
            print("Initialized Dense Retrieval (DPR)")
    
    def retrieve(
        self, 
        query_or_dataset: Union[str, Dataset], 
        topk: int = 1,
        rerank_topk: Optional[int] = None,
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Retrieve top-k passages using configured method(s).
        
        Args:
            query_or_dataset: Single query string or HuggingFace Dataset
            topk: Final number of passages to retrieve
            rerank_topk: Number of candidates to retrieve before fusion (hybrid only)
            
        Returns:
            If single query: Tuple of (scores, passages)
            If dataset: DataFrame with retrieved contexts
        """
        if self.mode == "sparse":
            return self.sparse_retriever.retrieve(query_or_dataset, topk)
        elif self.mode == "dense":
            return self.dense_retriever.retrieve(query_or_dataset, topk)
        else:  # hybrid
            return self._retrieve_hybrid(query_or_dataset, topk, rerank_topk)
    
    def _retrieve_hybrid(
        self,
        query_or_dataset: Union[str, Dataset],
        topk: int,
        rerank_topk: Optional[int] = None,
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """Hybrid retrieval combining sparse and dense methods."""
        if rerank_topk is None:
            rerank_topk = max(100, topk * 10)  # Default: 10x topk, min 100
        
        if isinstance(query_or_dataset, str):
            return self._retrieve_hybrid_single(query_or_dataset, topk, rerank_topk)
        elif isinstance(query_or_dataset, Dataset):
            return self._retrieve_hybrid_dataset(query_or_dataset, topk, rerank_topk)
        else:
            raise ValueError("query_or_dataset must be str or Dataset")
    
    def _retrieve_hybrid_single(
        self, 
        query: str, 
        topk: int,
        rerank_topk: int,
    ) -> Tuple[List[float], List[str]]:
        """Retrieve for a single query using hybrid approach."""
        # Get candidates from both retrievers
        sparse_scores, sparse_indices = self.sparse_retriever.get_relevant_doc(
            query, k=rerank_topk
        )
        dense_scores, dense_indices = self.dense_retriever.get_relevant_doc(
            query, k=rerank_topk
        )
        
        # Combine scores
        combined_scores = self._combine_scores(
            sparse_scores, sparse_indices,
            dense_scores, dense_indices
        )
        
        # Get top-k from combined
        top_k_indices = np.argsort(combined_scores)[::-1][:topk]
        top_k_scores = [combined_scores[idx] for idx in top_k_indices]
        top_k_passages = [self.contexts[idx] for idx in top_k_indices]
        
        print(f"[Hybrid Search query]\n{query}\n")
        for i, (score, passage) in enumerate(zip(top_k_scores, top_k_passages)):
            print(f"Top-{i+1} passage with score {score:.4f}")
            print(f"{passage[:200]}...\n")
        
        return top_k_scores, top_k_passages
    
    def _retrieve_hybrid_dataset(
        self, 
        dataset: Dataset, 
        topk: int,
        rerank_topk: int,
    ) -> pd.DataFrame:
        """Retrieve for a dataset using hybrid approach."""
        total = []
        queries = dataset["question"]
        
        # Bulk retrieval from both methods
        sparse_scores_list, sparse_indices_list = self.sparse_retriever.get_relevant_doc_bulk(
            queries, k=rerank_topk
        )
        dense_scores_list, dense_indices_list = self.dense_retriever.get_relevant_doc_bulk(
            queries, k=rerank_topk
        )
        
        for idx, example in enumerate(tqdm(dataset, desc="Hybrid retrieval")):
            # Combine scores for this query
            combined_scores = self._combine_scores(
                sparse_scores_list[idx], sparse_indices_list[idx],
                dense_scores_list[idx], dense_indices_list[idx]
            )
            
            # Get top-k
            top_k_indices = np.argsort(combined_scores)[::-1][:topk]
            retrieved_context = " ".join([self.contexts[i] for i in top_k_indices])
            
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
    
    def _combine_scores(
        self,
        sparse_scores: List[float],
        sparse_indices: List[int],
        dense_scores: List[float],
        dense_indices: List[int],
    ) -> np.ndarray:
        """
        Combine sparse and dense scores using weighted sum with normalization.
        
        Args:
            sparse_scores: Scores from sparse retriever
            sparse_indices: Document indices from sparse retriever
            dense_scores: Scores from dense retriever
            dense_indices: Document indices from dense retriever
            
        Returns:
            Combined scores for all documents
        """
        num_docs = len(self.contexts)
        combined = np.zeros(num_docs)
        
        # Normalize scores to [0, 1] range
        if len(sparse_scores) > 0:
            sparse_scores_norm = self._normalize(np.array(sparse_scores))
            for score, idx in zip(sparse_scores_norm, sparse_indices):
                combined[idx] += self.sparse_weight * score
        
        if len(dense_scores) > 0:
            dense_scores_norm = self._normalize(np.array(dense_scores))
            for score, idx in zip(dense_scores_norm, dense_indices):
                combined[idx] += self.dense_weight * score
        
        return combined
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1] range."""
        if len(scores) == 0:
            return scores
        min_score = scores.min()
        max_score = scores.max()
        if max_score - min_score == 0:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)
    
    def get_relevant_doc(self, query: str, k: int = 1) -> Tuple[List[float], List[int]]:
        """Get top-k relevant documents."""
        if self.mode == "sparse":
            return self.sparse_retriever.get_relevant_doc(query, k)
        elif self.mode == "dense":
            return self.dense_retriever.get_relevant_doc(query, k)
        else:  # hybrid
            scores, passages = self._retrieve_hybrid_single(query, k, max(100, k * 10))
            indices = [self.contexts.index(p) for p in passages]
            return scores, indices
    
    def get_relevant_doc_bulk(
        self, 
        queries: List[str], 
        k: int = 1
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """Get top-k relevant documents for multiple queries."""
        if self.mode == "sparse":
            return self.sparse_retriever.get_relevant_doc_bulk(queries, k)
        elif self.mode == "dense":
            return self.dense_retriever.get_relevant_doc_bulk(queries, k)
        else:  # hybrid
            rerank_topk = max(100, k * 10)

            sparse_scores_list, sparse_indices_list = self.sparse_retriever.get_relevant_doc_bulk(
                queries, k=rerank_topk
            )
            dense_scores_list, dense_indices_list = self.dense_retriever.get_relevant_doc_bulk(
                queries, k=rerank_topk
            )

            doc_scores_list: List[List[float]] = []
            doc_indices_list: List[List[int]] = []

            for sparse_scores, sparse_indices, dense_scores, dense_indices in zip(
                sparse_scores_list, sparse_indices_list, dense_scores_list, dense_indices_list
            ):
                combined_scores = self._combine_scores(
                    sparse_scores, sparse_indices,
                    dense_scores, dense_indices,
                )

                top_k_indices = np.argsort(combined_scores)[::-1][:k]
                doc_indices_list.append(top_k_indices.tolist())
                doc_scores_list.append(combined_scores[top_k_indices].tolist())

            return doc_scores_list, doc_indices_list


class HybridRetrieval:
    """
    Hybrid Retrieval combining Sparse (BM25) and Dense (DPR) retrieval.
    Uses weighted combination of both methods for improved retrieval quality.
    """
    
    def __init__(
        self,
        sparse_retriever: SparseRetrieval,
        dense_retriever: DenseRetrieval,
        sparse_weight: float = 0.5,
        dense_weight: float = 0.5,
        normalize_scores: bool = True,
    ):
        """
        Initialize hybrid retrieval.
        
        Args:
            sparse_retriever: Initialized SparseRetrieval instance
            dense_retriever: Initialized DenseRetrieval instance
            sparse_weight: Weight for sparse retrieval scores (default: 0.5)
            dense_weight: Weight for dense retrieval scores (default: 0.5)
            normalize_scores: Whether to normalize scores before combining
        """
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.normalize_scores = normalize_scores
        
        # Verify both retrievers use same contexts
        assert len(self.sparse_retriever.contexts) == len(self.dense_retriever.contexts), \
            "Sparse and Dense retrievers must use the same context corpus"
        
        self.contexts = self.sparse_retriever.contexts
        print(f"Initialized HybridRetrieval with {len(self.contexts)} contexts")
        print(f"Weights: Sparse={sparse_weight}, Dense={dense_weight}")
    
    def retrieve(
        self, 
        query_or_dataset: Union[str, Dataset], 
        topk: int = 1,
        rerank_topk: int = 100,
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Retrieve top-k passages using hybrid approach.
        
        Strategy:
        1. Get top-N candidates from both sparse and dense retrievers
        2. Combine scores with weighted sum
        3. Re-rank and return top-k
        
        Args:
            query_or_dataset: Single query string or HuggingFace Dataset
            topk: Final number of passages to retrieve
            rerank_topk: Number of candidates to retrieve from each method before fusion
            
        Returns:
            If single query: Tuple of (scores, passages)
            If dataset: DataFrame with retrieved contexts
        """
        if isinstance(query_or_dataset, str):
            return self._retrieve_single(query_or_dataset, topk, rerank_topk)
        elif isinstance(query_or_dataset, Dataset):
            return self._retrieve_dataset(query_or_dataset, topk, rerank_topk)
        else:
            raise ValueError("query_or_dataset must be str or Dataset")
    
    def _retrieve_single(
        self, 
        query: str, 
        topk: int,
        rerank_topk: int,
    ) -> Tuple[List[float], List[str]]:
        """Retrieve for a single query using hybrid approach."""
        # Get candidates from both retrievers
        sparse_scores, sparse_indices = self.sparse_retriever.get_relevant_doc(
            query, k=rerank_topk
        )
        dense_scores, dense_indices = self.dense_retriever.get_relevant_doc(
            query, k=rerank_topk
        )
        
        # Combine scores
        combined_scores = self._combine_scores(
            sparse_scores, sparse_indices,
            dense_scores, dense_indices
        )
        
        # Get top-k from combined
        top_k_indices = np.argsort(combined_scores)[::-1][:topk]
        top_k_scores = [combined_scores[idx] for idx in top_k_indices]
        top_k_passages = [self.contexts[idx] for idx in top_k_indices]
        
        print(f"[Hybrid Search query]\n{query}\n")
        for i, (score, passage) in enumerate(zip(top_k_scores, top_k_passages)):
            print(f"Top-{i+1} passage with score {score:.4f}")
            print(f"{passage[:200]}...\n")
        
        return top_k_scores, top_k_passages
    
    def _retrieve_dataset(
        self, 
        dataset: Dataset, 
        topk: int,
        rerank_topk: int,
    ) -> pd.DataFrame:
        """Retrieve for a dataset using hybrid approach."""
        total = []
        queries = dataset["question"]
        
        # Bulk retrieval from both methods
        sparse_scores_list, sparse_indices_list = self.sparse_retriever.get_relevant_doc_bulk(
            queries, k=rerank_topk
        )
        dense_scores_list, dense_indices_list = self.dense_retriever.get_relevant_doc_bulk(
            queries, k=rerank_topk
        )
        
        for idx, example in enumerate(tqdm(dataset, desc="Hybrid retrieval")):
            # Combine scores for this query
            combined_scores = self._combine_scores(
                sparse_scores_list[idx], sparse_indices_list[idx],
                dense_scores_list[idx], dense_indices_list[idx]
            )
            
            # Get top-k
            top_k_indices = np.argsort(combined_scores)[::-1][:topk]
            retrieved_context = " ".join([self.contexts[i] for i in top_k_indices])
            
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
    
    def _combine_scores(
        self,
        sparse_scores: List[float],
        sparse_indices: List[int],
        dense_scores: List[float],
        dense_indices: List[int],
    ) -> np.ndarray:
        """
        Combine sparse and dense scores using weighted sum.
        
        Args:
            sparse_scores: Scores from sparse retriever
            sparse_indices: Document indices from sparse retriever
            dense_scores: Scores from dense retriever
            dense_indices: Document indices from dense retriever
            
        Returns:
            Combined scores for all documents (length = num_documents)
        """
        num_docs = len(self.contexts)
        combined = np.zeros(num_docs)
        
        # Normalize scores if requested
        if self.normalize_scores and len(sparse_scores) > 0 and len(dense_scores) > 0:
            sparse_scores = self._normalize(np.array(sparse_scores))
            dense_scores = self._normalize(np.array(dense_scores))
        
        # Add weighted sparse scores
        for score, idx in zip(sparse_scores, sparse_indices):
            combined[idx] += self.sparse_weight * score
        
        # Add weighted dense scores
        for score, idx in zip(dense_scores, dense_indices):
            combined[idx] += self.dense_weight * score
        
        return combined
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1] range."""
        if len(scores) == 0:
            return scores
        min_score = scores.min()
        max_score = scores.max()
        if max_score - min_score == 0:
            return np.ones_like(scores)
        return (scores - min_score) / (max_score - min_score)
    
    def get_relevant_doc(self, query: str, k: int = 1) -> Tuple[List[float], List[int]]:
        """Get top-k relevant documents using hybrid approach."""
        scores, passages = self._retrieve_single(query, k, rerank_topk=100)
        # Convert passages back to indices
        indices = [self.contexts.index(p) for p in passages]
        return scores, indices
    
    def get_relevant_doc_bulk(
        self, 
        queries: List[str], 
        k: int = 1
    ) -> Tuple[List[List[float]], List[List[int]]]:
        """Get top-k relevant documents for multiple queries using hybrid approach."""
        from datasets import Dataset as HFDataset
        
        # Create temporary dataset
        temp_dataset = HFDataset.from_dict({
            "question": queries,
            "id": [f"temp_{i}" for i in range(len(queries))],
        })
        
        df = self._retrieve_dataset(temp_dataset, k, rerank_topk=100)
        
        # Extract scores and indices (approximate - actual implementation may vary)
        doc_scores_list = []
        doc_indices_list = []
        
        for _, row in df.iterrows():
            # Note: We lose individual scores in DataFrame format
            # This is a simplified return
            doc_scores_list.append([1.0] * k)  # Placeholder scores
            # Extract indices from contexts
            contexts_list = row["context"].split(" ")[:k]
            indices = [self.contexts.index(c) if c in self.contexts else 0 
                      for c in contexts_list]
            doc_indices_list.append(indices)
        
        return doc_scores_list, doc_indices_list


if __name__ == "__main__":
    from datasets import load_from_disk
    
    # Load dataset
    datasets = load_from_disk("../data/train_dataset")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", use_fast=True)
    
    # Initialize Sparse (BM25) retriever
    sparse_retriever = SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path="../data",
        context_path="wikipedia_documents.json",
    )
    
    # Test Sparse Retrieval
    print("\n" + "="*50)
    print("Testing Sparse Retrieval (BM25)")
    print("="*50)
    
    # Test single query
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    scores, passages = sparse_retriever.retrieve(query, topk=5)
    
    # Test on validation set
    val_dataset = datasets["validation"]
    results_df = sparse_retriever.retrieve(val_dataset, topk=10)
    
    # Calculate accuracy
    if "original_context" in results_df.columns:
        results_df["correct"] = results_df["original_context"] == results_df["context"]
        accuracy = results_df["correct"].sum() / len(results_df)
        print(f"Sparse Retrieval accuracy: {accuracy:.4f}")
    
    print(f"Retrieved {len(results_df)} examples")
    
    # Initialize Dense retriever (placeholder)
    print("\n" + "="*50)
    print("Dense Retrieval (DPR) - Not yet implemented")
    print("="*50)
    dense_retriever = DenseRetrieval(
        model_name_or_path="klue/bert-base",
        data_path="../data",
        context_path="wikipedia_documents.json",
        q_encoder_path="./models/dpr/q_encoder",
        p_encoder_path="./models/dpr/p_encoder",
    )
    print("DenseRetrieval initialized (structure only)")
    
    # Hybrid Retrieval (placeholder - will work once DPR is implemented)
    print("\n" + "="*50)
    print("Hybrid Retrieval - Structure ready")
    print("="*50)
    print("To use HybridRetrieval:")
    print("  hybrid = HybridRetrieval(sparse_retriever, dense_retriever)")
    print("  results = hybrid.retrieve(dataset, topk=10)")
    print("  # Combines BM25 + DPR with weighted fusion")