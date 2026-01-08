import os
import sys
import json
import random
import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List, Tuple, Dict
import pandas as pd
import pickle
import re

# Add parent directory to path to import retrieval
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval import SparseRetrieval

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SetHardNegatives:
    """
    Set hard negatives for DPR training using BM25 sparse retrieval.
    Hard negatives are top-ranked passages from BM25 that are NOT the positive passage.
    """
    
    def __init__(
        self,
        data_path: str,
        context_path: str,
        max_context_len: int,
        max_question_len: int,
        neg_num: int,
        tokenizer,
        top_k: int = 100,
    ):
        """
        Initialize hard negative miner.
        
        Args:
            data_path: Path to training dataset
            context_path: Path to Wikipedia documents JSON
            max_context_len: Max length for context
            max_question_len: Max length for question
            neg_num: Number of hard negatives per question
            tokenizer: Tokenizer for BM25
            top_k: Number of candidates to retrieve from BM25
        """
        self.data_path = data_path
        self.context_path = context_path
        self.max_context_len = max_context_len
        self.max_question_len = max_question_len
        self.neg_num = neg_num
        self.tokenizer = tokenizer
        self.top_k = top_k

        # Load dataset
        print(f"Loading dataset from {data_path}")
        self.dataset = load_from_disk(self.data_path)
        
        # Load Wikipedia contexts
        data_dir = os.path.dirname(data_path)
        with open(os.path.join(data_dir, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)
        
        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print(f"Loaded {len(self.contexts)} unique contexts")
        
        # Initialize BM25 sparse retriever
        print("Initializing BM25 retriever for hard negative mining...")
        self.sparse_retriever = SparseRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path=data_dir,
            context_path=context_path,
        )
        print("BM25 retriever initialized")
    
    def mine_hard_negatives(self, split: str = "train") -> List[List[int]]:
        """
        Mine hard negatives using BM25 sparse retrieval.
        
        For each question:
        1. Retrieve top-k passages using BM25
        2. Remove the positive passage
        3. Select top neg_num as hard negatives
        
        Args:
            split: Dataset split to use ('train' or 'validation')
            
        Returns:
            List of lists containing hard negative indices for each question
        """
        print(f"\nMining hard negatives from {split} split...")
        
        dataset_split = self.dataset[split]
        questions = dataset_split["question"]
        
        # Build positive passage indices
        print("Building positive passage indices...")
        pos_indices = self._build_positive_indices(dataset_split)
        
        # Get BM25 rankings for all questions
        print(f"Retrieving top-{self.top_k} passages from BM25...")
        _, doc_indices_list = self.sparse_retriever.get_relevant_doc_bulk(
            questions, k=self.top_k
        )
        
        # Mine hard negatives
        hard_neg_indices = []
        num_found = 0
        skipped = 0
        
        for i, (pos_idx, bm25_indices) in enumerate(
            tqdm(zip(pos_indices, doc_indices_list), total=len(pos_indices), desc="Mining negatives")
        ):
            # Skip if no valid positive passage
            if pos_idx == -1:
                hard_neg_indices.append([])
                skipped += 1
                continue
            
            # Remove positive passage from BM25 results
            hard_negs = [idx for idx in bm25_indices if idx != pos_idx]
            
            # Take top neg_num as hard negatives
            hard_negs = hard_negs[:self.neg_num]
            
            if len(hard_negs) > 0:
                num_found += 1
            
            hard_neg_indices.append(hard_negs)
        
        print(f"Found hard negatives for {num_found}/{len(pos_indices)} questions")
        print(f"Skipped {skipped} questions without valid positive passages")
        valid_negs = [h for h in hard_neg_indices if len(h) > 0]
        if valid_negs:
            print(f"Average hard negatives per valid question: {sum(len(h) for h in valid_negs) / len(valid_negs):.2f}")
        
        return hard_neg_indices
    
    def _build_positive_indices(self, dataset_split) -> List[int]:
        """
        Build positive passage indices by matching contexts to corpus.
        
        Args:
            dataset_split: Dataset split to process
            
        Returns:
            List of positive passage indices
        """
        pos_indices = []
        
        # Build context-to-index mapping for O(1) lookup
        context_to_idx = {ctx: idx for idx, ctx in enumerate(self.contexts)}
        
        for example in tqdm(dataset_split, desc="Building positive indices"):
            if "context" not in example:
                pos_indices.append(-1)
                continue
            
            gold_context = example["context"]
            
            # Try exact match first (O(1) lookup)
            if gold_context in context_to_idx:
                pos_indices.append(context_to_idx[gold_context])
                continue
            
            # If no exact match, try to find by answer
            found = False
            if "answers" in example and len(example["answers"]["text"]) > 0:
                answer_text = example["answers"]["text"][0]
                
                # Search through contexts for answer
                for ctx_idx, ctx in enumerate(self.contexts):
                    if answer_text in ctx:
                        pos_indices.append(ctx_idx)
                        found = True
                        break
            
            if not found:
                pos_indices.append(-1)
        
        # Count valid positive indices
        valid_count = sum(1 for idx in pos_indices if idx != -1)
        print(f"Found {valid_count}/{len(pos_indices)} valid positive passages")
        
        return pos_indices
    
    def create_training_dataset(
        self, 
        split: str = "train",
        hard_neg_indices: List[List[int]] = None,
        tokenizer=None,
        max_q_length: int = 64,
        max_p_length: int = 256,
        num_hard_negatives: int = 1,
    ):
        """
        Create DPR training dataset with hard negatives.
        
        Args:
            split: Dataset split to use
            hard_neg_indices: Pre-computed hard negative indices (if None, will mine them)
            tokenizer: Tokenizer for dataset (optional)
            max_q_length: Max question length
            max_p_length: Max passage length
            num_hard_negatives: Number of hard negatives per example
            
        Returns:
            DPRDatasetWithNegatives instance
        """
        dataset_split = self.dataset[split]
        
        if hard_neg_indices is None:
            hard_neg_indices = self.mine_hard_negatives(split)
        
        # Build positive indices
        pos_indices = self._build_positive_indices(dataset_split)
        
        # Filter to only valid examples (those with positive passages)
        questions = []
        valid_pos_indices = []
        valid_hard_neg_indices = []
        
        for i, (q, pos_idx, hard_negs) in enumerate(
            zip(dataset_split["question"], pos_indices, hard_neg_indices)
        ):
            if pos_idx != -1:  # Only include if we found a positive passage
                questions.append(q)
                valid_pos_indices.append(pos_idx)
                valid_hard_neg_indices.append(hard_negs)
        
        print(f"\nCreated training dataset:")
        print(f"  Total questions: {len(questions)}")
        print(f"  Total contexts: {len(self.contexts)}")
        print(f"  Questions with valid positives: {len(questions)}/{len(dataset_split)}")
        if valid_hard_neg_indices:
            avg_negs = sum(len(h) for h in valid_hard_neg_indices) / len(valid_hard_neg_indices)
            print(f"  Avg hard negatives per question: {avg_negs:.2f}")
        
        # Return DPRDatasetWithNegatives instance
        return self.DPRDatasetWithNegatives(
            questions=questions,
            passages=self.contexts,
            pos_indices=valid_pos_indices,
            neg_indices=valid_hard_neg_indices,
            tokenizer=tokenizer,
            max_q_length=max_q_length,
            max_p_length=max_p_length,
            num_hard_negatives=num_hard_negatives,
        )
    
    class DPRDatasetWithNegatives(Dataset):
        """
        Enhanced DPR Dataset with hard negative support.
        Each example has:
        - question
        - positive passage (correct answer)
        - hard negatives (challenging wrong passages from BM25)
        """

        def __init__(
            self,
            questions: list,
            passages: list,
            pos_indices: list,
            neg_indices: list = None,
            tokenizer=None,
            max_q_length: int = 64,
            max_p_length: int = 256,
            num_hard_negatives: int = 1,
        ):
            self.questions = questions
            self.passages = passages
            self.pos_indices = pos_indices
            self.neg_indices = neg_indices or [[]] * len(questions)
            self.tokenizer = tokenizer
            self.max_q_length = max_q_length
            self.max_p_length = max_p_length
            self.num_hard_negatives = num_hard_negatives

        def __len__(self):
            return len(self.questions)

        def __getitem__(self, idx):
            question = self.questions[idx]
            pos_passage = self.passages[self.pos_indices[idx]]
            
            # Get hard negatives (or empty if not available)
            hard_negs = self.neg_indices[idx] if idx < len(self.neg_indices) else []
            hard_neg_passages = [
                self.passages[neg_idx] 
                for neg_idx in hard_negs[:self.num_hard_negatives]
            ] if hard_negs else []

            return {
                "question": question,
                "pos_passage": pos_passage,
                "hard_neg_passages": hard_neg_passages,
            }
    
    def save_hard_negatives(self, hard_neg_indices: List[List[int]], output_path: str):
        """Save hard negative indices to pickle file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(hard_neg_indices, f)
        print(f"Saved hard negatives to {output_path}")
    
    def load_hard_negatives(self, input_path: str) -> List[List[int]]:
        """Load hard negative indices from pickle file."""
        with open(input_path, "rb") as f:
            hard_neg_indices = pickle.load(f)
        print(f"Loaded hard negatives from {input_path}")
        return hard_neg_indices

