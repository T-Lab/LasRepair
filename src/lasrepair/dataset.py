import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class Seq2SeqDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128, weights=None):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        self.weights = weights

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text, target_text = self.data[idx]
        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        targets = self.tokenizer.encode_plus(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        result = {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": targets["input_ids"].squeeze(0)
        }
        
        # Add weight if provided
        if self.weights is not None:
            result["weight"] = self.weights[idx]
        
        return result