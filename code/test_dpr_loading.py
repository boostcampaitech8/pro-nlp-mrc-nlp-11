#!/usr/bin/env python3
"""
Simple test script to verify DPR encoder loading works correctly.
Helps debug issues before running full inference.
"""

import os
import sys

def test_dpr_loading():
    """Test if DPR encoders will load correctly."""
    print("=" * 70)
    print("DPR Loading Test")
    print("=" * 70)
    
    # Check if trained models exist
    q_encoder_path = "./models/dpr/q_encoder"
    p_encoder_path = "./models/dpr/p_encoder"
    
    print(f"\n1. Checking trained model paths:")
    print(f"   Question encoder path: {q_encoder_path}")
    print(f"   Passage encoder path: {p_encoder_path}")
    
    q_exists = os.path.exists(os.path.join(q_encoder_path, "config.json"))
    p_exists = os.path.exists(os.path.join(p_encoder_path, "config.json"))
    
    if q_exists:
        print(f"   ✓ Question encoder found at {q_encoder_path}")
    else:
        print(f"   ✗ Question encoder NOT found at {q_encoder_path}")
        print(f"     DPR training must be run first!")
    
    if p_exists:
        print(f"   ✓ Passage encoder found at {p_encoder_path}")
    else:
        print(f"   ✗ Passage encoder NOT found at {p_encoder_path}")
        print(f"     DPR training must be run first!")
    
    print(f"\n2. Checking Wikipedia documents:")
    wiki_path = "../data/wikipedia_documents.json"
    if os.path.exists(wiki_path):
        import json
        with open(wiki_path) as f:
            wiki = json.load(f)
        num_docs = len(wiki)
        print(f"   ✓ Wikipedia documents found: {num_docs} documents")
    else:
        print(f"   ✗ Wikipedia documents NOT found at {wiki_path}")
    
    print(f"\n3. Checking training dataset:")
    train_dataset_path = "../data/train_dataset"
    if os.path.exists(train_dataset_path):
        print(f"   ✓ Training dataset found at {train_dataset_path}")
    else:
        print(f"   ✗ Training dataset NOT found at {train_dataset_path}")
    
    print(f"\n4. Testing imports:")
    try:
        from retrieval.Dense_retrieval import DenseRetrieval
        print(f"   ✓ DenseRetrieval imported successfully")
    except Exception as e:
        print(f"   ✗ Failed to import DenseRetrieval: {e}")
        return False
    
    print(f"\n5. Testing model loading:")
    if q_exists and p_exists:
        try:
            from transformers import AutoModel, AutoTokenizer
            print(f"   Loading question encoder...")
            q_encoder = AutoModel.from_pretrained(q_encoder_path)
            print(f"   ✓ Question encoder loaded successfully")
            
            print(f"   Loading passage encoder...")
            p_encoder = AutoModel.from_pretrained(p_encoder_path)
            print(f"   ✓ Passage encoder loaded successfully")
            
        except Exception as e:
            print(f"   ✗ Failed to load models: {e}")
            return False
    else:
        print(f"   ⚠ Skipping model loading test (models not found)")
        print(f"   Run DPR training first: python retrieval/DPR_train.py")
        return False
    
    print(f"\n" + "=" * 70)
    print("✓ All checks passed! DPR should load correctly.")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_dpr_loading()
    sys.exit(0 if success else 1)
