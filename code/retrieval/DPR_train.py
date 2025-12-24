"""
Train Dense Passage Retrieval (DPR) model on train_dataset.
Fine-tunes a dual-encoder architecture (question encoder + passage encoder) using in-batch negatives.
"""


import logging
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from DPR_util import SetHardNegatives, set_seed
from transformers import AdamW, AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from arguments import DataTrainingArguments, ModelArguments
from datasets import load_from_disk

logger = logging.getLogger(__name__)

class DPRDataset(Dataset):
    """
    Dataset for DPR training.
    Each example has a question, positive passage, and optionally negative passages.
    """

    def __init__(
        self,
        questions: list,
        passages: list,
        pos_indices: list,
        tokenizer,
        max_q_length: int = 64,
        max_p_length: int = 256,
    ):
        self.questions = questions
        self.passages = passages
        self.pos_indices = pos_indices
        self.tokenizer = tokenizer
        self.max_q_length = max_q_length
        self.max_p_length = max_p_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        pos_passage = self.passages[self.pos_indices[idx]]

        return {
            "question": question,
            "pos_passage": pos_passage,
        }




class DPRTrainer:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        learning_rate: float = 1e-5,
        warmup_steps: int = 1000,
        num_epochs: int = 5,
        batch_size: int = 16,
        max_q_length: int = 64,
        max_p_length: int = 256,
        output_dir: str = "./models/dpr",
    ):
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_q_length = max_q_length
        self.max_p_length = max_p_length
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        # Load tokenizer and models
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.q_encoder = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        self.p_encoder = AutoModel.from_pretrained(model_name_or_path).to(self.device)
       

    def encode_texts(self, texts: list, is_query: bool = False) -> np.ndarray:
        """Encode texts to embeddings."""
        max_length = self.max_q_length if is_query else self.max_p_length
        encoder = self.q_encoder if is_query else self.p_encoder

        all_embs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding"):
                batch_texts = texts[i : i + self.batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = encoder(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
                cls_emb = F.normalize(cls_emb, p=2, dim=-1)
                all_embs.append(cls_emb.cpu().numpy().astype(np.float32))

        return np.concatenate(all_embs, axis=0)

    def collate_fn(self, batch):
        """Collate batch for training with hard negatives support."""
        questions = [item["question"] for item in batch]
        pos_passages = [item["pos_passage"] for item in batch]
        
        # Check if hard negatives are present
        has_hard_negs = "hard_neg_passages" in batch[0] and len(batch[0]["hard_neg_passages"]) > 0

        q_inputs = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_q_length,
            return_tensors="pt",
        ).to(self.device)

        # Collect all passages: positive + hard negatives
        if has_hard_negs:
            all_passages = []
            for item in batch:
                all_passages.append(item["pos_passage"])
                all_passages.extend(item["hard_neg_passages"])
            
            p_inputs = self.tokenizer(
                all_passages,
                padding=True,
                truncation=True,
                max_length=self.max_p_length,
                return_tensors="pt",
            ).to(self.device)
            
            num_hard_negs = len(batch[0]["hard_neg_passages"])
            return {"q_inputs": q_inputs, "p_inputs": p_inputs, "num_hard_negs": num_hard_negs}
        else:
            p_inputs = self.tokenizer(
                pos_passages,
                padding=True,
                truncation=True,
                max_length=self.max_p_length,
                return_tensors="pt",
            ).to(self.device)
            
            return {"q_inputs": q_inputs, "p_inputs": p_inputs, "num_hard_negs": 0}
    
    def train(self, train_dataset: DPRDataset):
        """Train DPR model."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

        # Optimizer and scheduler
        optimizer = AdamW(
            list(self.q_encoder.parameters()) + list(self.p_encoder.parameters()),
            lr=self.learning_rate,
        )
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

        self.q_encoder.train()
        self.p_encoder.train()

        for epoch in range(self.num_epochs):
            total_loss = 0
            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}") as pbar:
                for batch in pbar:
                    optimizer.zero_grad()

                    # Encode questions and passages
                    q_outputs = self.q_encoder(**batch["q_inputs"])
                    p_outputs = self.p_encoder(**batch["p_inputs"])

                    q_emb = F.normalize(q_outputs.last_hidden_state[:, 0, :], p=2, dim=-1)  # (B, D)
                    p_emb = F.normalize(p_outputs.last_hidden_state[:, 0, :], p=2, dim=-1)  # (num_passages, D)

                    num_hard_negs = batch.get("num_hard_negs", 0)
                    batch_size = len(q_emb)
                    
                    if num_hard_negs > 0:
                        # With hard negatives: reshape p_emb to (B, 1+num_hard_negs, D)
                        # First passage is positive, rest are hard negatives
                        passages_per_question = 1 + num_hard_negs
                        p_emb = p_emb.view(batch_size, passages_per_question, -1)  # (B, 1+K, D)
                        
                        # Compute scores: (B, D) x (B, 1+K, D) -> (B, 1+K)
                        scores = torch.bmm(q_emb.unsqueeze(1), p_emb.transpose(1, 2)).squeeze(1)  # (B, 1+K)
                        
                        # Labels: first passage (index 0) is always positive
                        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                        
                        # Cross-entropy loss
                        loss = F.cross_entropy(scores, labels)
                    else:
                        # In-batch negatives only: compute similarities (B, B)
                        # scores[i, j] = sim(q_i, p_j)
                        scores = torch.matmul(q_emb, p_emb.T)  # (B, B)

                        # Labels: diagonal is positive (q_i matched with p_i)
                        labels = torch.arange(batch_size, device=self.device)

                        # Cross-entropy loss
                        loss = F.cross_entropy(scores, labels)

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    total_loss += loss.item()
                    pbar.set_postfix({"loss": loss.item()})

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

        # Save models
        self.q_encoder.save_pretrained(os.path.join(self.output_dir, "q_encoder"))
        self.p_encoder.save_pretrained(os.path.join(self.output_dir, "p_encoder"))
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"Models saved to {self.output_dir}")


def main():
    logging.basicConfig(level=logging.INFO)
    
    # Set seed for reproducibility
    seed = 2024
    set_seed(seed)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base", use_fast=True)
    
    # Initialize hard negative miner
    logger.info("Initializing hard negative miner...")
    miner = SetHardNegatives(
        data_path="../data/train_dataset",
        context_path="../data/wikipedia_documents.json",
        max_context_len=256,
        max_question_len=64,
        neg_num=1,
        tokenizer=tokenizer,
        top_k=100,
    )

    # Cache hard negatives to avoid re-mining on every DPR run
    hard_neg_cache = os.path.join("./models/dpr", "hard_negatives.pkl")
    if os.path.exists(hard_neg_cache):
        logger.info("Loading cached hard negatives from %s", hard_neg_cache)
        hard_neg_indices = miner.load_hard_negatives(hard_neg_cache)
    else:
        logger.info("Mining hard negatives (cache not found at %s)", hard_neg_cache)
        hard_neg_indices = miner.mine_hard_negatives(split="train")
        miner.save_hard_negatives(hard_neg_indices, hard_neg_cache)
    
    # Create training dataset with cached/mined hard negatives
    logger.info("Creating training dataset with hard negatives...")
    train_dataset = miner.create_training_dataset(
        split="train",
        hard_neg_indices=hard_neg_indices,
        tokenizer=tokenizer,
        max_q_length=64,
        max_p_length=256,
        num_hard_negatives=1,
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")

    # Create trainer
    trainer = DPRTrainer(
        model_name_or_path="klue/bert-base",
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=1e-5,
        warmup_steps=500,
        num_epochs=3,
        batch_size=32,
        output_dir="./models/dpr",
    )

    trainer.train(train_dataset)


if __name__ == "__main__":
    main()
