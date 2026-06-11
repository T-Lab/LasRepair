"""
Used for seq2seq and mice based on dataset flights with single model approach
"""
import os
import sys
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import GPUtil
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.optim import AdamW, SGD, Adam
import gc
import numpy as np
from sklearn.model_selection import train_test_split
from .utils import F1_score, all_wrong_corrector, EDR
from .dataset import Seq2SeqDataset
from .confident_learning import uncertainty_matrix
import time
import argparse
from pathlib import Path

from .paths import DEFAULT_DATASETS_DIR, DEFAULT_RESULTS_DIR


class MultiModelIterativeGenerativeRepair():
    def __init__(self, dirty_path, clean_path, args:argparse.Namespace):
        """
        args: argparse.Namespace, see main function for more details
        """
        self.args = args
        self.dirty_df = pd.read_csv(dirty_path, na_values=["nan", "NaN", "N/A", "None", "null"], dtype='str').replace(np.nan, '')
        self.clean_df = pd.read_csv(clean_path, na_values=["nan", "NaN", "N/A", "None", "null"], dtype='str').replace(np.nan, '')
        self.clean_df.columns = self.dirty_df.columns
        if self.clean_df.shape != self.dirty_df.shape:
            raise ValueError('clean_df and dirty_df have different shape, set to lower shape')

        self.device = args.gpu
        self.max_iteration = args.max_iteration
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.name = args.experiment
        self.use_weight = args.use_weight
        self.sample_num = args.sample_num
        self.temperature = args.temperature
        self.correct_prop = args.correct_prop
        self.sml_model = args.sml_name

        self.error_df = (self.clean_df.ne(self.dirty_df))
        self.clean_df, self.dirty_df, self.error_df = all_wrong_corrector(self.clean_df, self.dirty_df, self.error_df, prop=self.correct_prop)
        self.res_df = self.dirty_df.copy()  # store result sync
        self.temp_df = self.dirty_df.copy()  # copy res_df to process async
        self.weight_df = self.error_df.astype(float)  # full uncertainty matrix
        self.weight_used = np.ones(len(self.clean_df))  # row-wise weights for training
        
        # Initialize single model for all columns
        self.model = None
        self.optimizer = None
        self.tokenizer = None

        self.flag = True

    def initialize_model(self):
        """Initialize a single model for all columns"""
        seq2seq_model = self.sml_model
        tokenizer = AutoTokenizer.from_pretrained(seq2seq_model)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(seq2seq_model)
        
        model = base_model
        model = model.to(self.device)
        optimizer = AdamW(model.parameters(), lr=1e-4)
        
        return model, optimizer, tokenizer

    def compute_weight_used(self):
        """Compute row-wise mean of weight_df and apply softmax to get weight_used"""
        # Compute row means across all columns. Actually the uncertainty matrix. Consider of reversing it.
        row_means = self.weight_df.mean(axis=1).values  # Convert to numpy array
        # temperature used here. default is 1.0, lower temperature means more focus on the error values
        row_means = row_means / self.temperature
        
        # Apply softmax to get normalized weights
        exp_means = np.exp(row_means - np.max(row_means))  # Subtract max for numerical stability
        self.weight_used = exp_means / np.sum(exp_means)

    def preprocess(self, target_index=0, iteration_time=1):
        """
        preprocess the data
        divide the data into train and test
        prepare the prompt and the target for the model
        calculate the weight for the training data
        """
        train_data, test_data, train_weights = [], [], []
        n_rows, n_cols = self.clean_df.shape
        column_names = list(self.clean_df.columns)
    
        # generate prompt and target
        # prompt: col_name: col_value <extra_id_1> .... <extra_id_2> target: target_name
        for row in range(n_rows):
            # Get input values
            inputs = []
            # Use all columns except target
            for col in range(n_cols):
                if col == target_index:
                    continue
                # mask the error values
                if iteration_time == 1 and self.error_df.iloc[row, col]:
                    inputs.append(str(column_names[col]) + ": " + "<extra_id_0>")
                else:
                    inputs.append(str(column_names[col]) + ": " + str(self.temp_df.iloc[row, col]))

            # Construct prompt, simple, just for slm
            target_content = str(self.clean_df.iloc[row, target_index])
            
            if self.error_df.iloc[row, target_index]:
                input_content = "<extra_id_1>".join(inputs)
            # contain dirty information now, comment out to go back to original version.
                input_content = input_content + f"<extra_id_2> {str(column_names[target_index])}: " + str(self.temp_df.iloc[row, target_index])
                # prefix the task
                input_content = "correct column " + str(column_names[target_index]) + ": " + input_content
                target_content = ''
                test_data.append((input_content, target_content))
            else:
                input_content = "<extra_id_1>".join(inputs)
                input_content = input_content + "<extra_id_2>" + str(column_names[target_index]) + ": "
                train_data.append((input_content, target_content))
                # Get weight for this training sample from weight_used (default to 1.0 for first iteration)
                weight = float(self.weight_used[row]) if iteration_time > 1 and self.use_weight else 1.0
                train_weights.append(weight)
            
            if self.flag:
                print(f"input_content: {input_content}, target_content: {target_content}")
                self.flag = False

        return train_data, test_data, train_weights

    def train_step(self, train_loader, epochs=3):
        model = self.model
        optimizer = self.optimizer
        device = self.device
        epochs = self.epochs

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # Check if weights are provided
                if "weight" in batch and self.use_weight:
                    weights = batch["weight"].to(device).float()
                    
                    # Get logits and compute weighted loss manually
                    outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits  # [batch, seq_len, vocab_size]
                    
                    # Compute cross-entropy loss per token
                    vocab_size = logits.size(-1)
                    loss_per_token = torch.nn.functional.cross_entropy(
                        logits.view(-1, vocab_size),
                        labels.view(-1),
                        ignore_index=-100,
                        reduction='none'
                    ).view(labels.size())  # [batch, seq_len]
                    
                    # Mask out padding tokens and compute per-sample loss
                    valid_mask = (labels != -100).float()
                    per_sample_loss = (loss_per_token * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
                    
                    # Apply sample weights and compute final loss
                    loss = (per_sample_loss * weights).sum() / weights.sum().clamp_min(1.0)
                else:
                    # Standard unweighted loss
                    outputs = model(input_ids=inputs, labels=labels, attention_mask=attention_mask)
                    loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    def test_step(self, test_loader, max_length=128, n=5):
        model = self.model
        device = self.device
        tokenizer = self.tokenizer
        n_samples = n  # number of samples to calculate the confident matrix

        model.eval()
        total_loss = 0
        predictions, targets, first_n_tokens, logits = [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=inputs, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss
                total_loss += loss.item()
                
                generated_ids = model.generate(input_ids=inputs, attention_mask=attention_mask, max_length=max_length)
                preds = [tokenizer.decode(g_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g_id in
                         generated_ids]
                predictions.extend(preds)
                targets.extend(
                    [tokenizer.decode(tgt, skip_special_tokens=True, clean_up_tokenization_spaces=True) for tgt in
                     labels])

                if self.use_weight:
                    first_pos_logits = outputs.logits[:, 0, :]  # [batch, vocab] - logits for first position
                    
                    # here is ids, not tokens!
                    top_n_logits, top_n_ids = torch.topk(first_pos_logits, k=n_samples, dim=-1)

                    first_n_tokens.extend(top_n_ids.cpu().tolist())  # List of [n] lists
                    logits.extend(top_n_logits.cpu().tolist())  # List of [n] lists

        return predictions, targets, first_n_tokens, logits

    def run(self):
        """
        Main training loop
        """
        now_time = time.time()
        start_time = time.time()
        self.last_f1 = 0

        # Initialize single model once
        print('Initializing model')
        self.model, self.optimizer, self.tokenizer = self.initialize_model()

        # column version
        for iteration in range(self.max_iteration):
            print(f'---------------start iteration {iteration + 1}---------------')
            # processing each column in the order of error rate()
            for ind in range(len(self.clean_df.columns)):
                column_name = self.clean_df.columns[ind]
                
                # skip columns without errors
                if self.error_df.iloc[:, ind].sum() == 0:
                    print(f'skip index {ind}\n')
                    continue
                
                # Process data and train
                train_data, test_data, train_weights = self.preprocess(target_index=ind, iteration_time=iteration + 1)
                
                # Sample here for efficiency
                if self.sample_num != 0:
                    sample_indices = np.random.choice(len(train_data), self.sample_num, replace=False)
                    train_data = [train_data[i] for i in sample_indices]
                    train_weights = [train_weights[i] for i in sample_indices]

                train_dataset = Seq2SeqDataset(train_data, self.tokenizer, weights=train_weights if self.use_weight else None)
                test_dataset = Seq2SeqDataset(test_data, self.tokenizer)
                # must not shuffle, or the order will be wrong
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

                self.train_step(train_loader, epochs=self.epochs)
                print(f'fine tune for column {column_name}: {(time.time() - now_time)/60:.2f} minutes')
                now_time = time.time()

                predictions, targets, first_n_tokens, logits = self.test_step(test_loader)
                print(f'inference for column {column_name}: {(time.time() - now_time)/60:.2f} minutes\n')
                now_time = time.time()

                # Update results
                if self.use_weight:
                    weights = uncertainty_matrix(first_n_tokens, logits)
                n_rows, n_cols = self.clean_df.shape
                pointer = 0
                for row in range(n_rows):
                    if self.error_df.iloc[row, ind]:
                        self.res_df.iloc[row, ind] = predictions[pointer]
                        if self.use_weight:
                            self.weight_df.iloc[row, ind] = weights[pointer]
                        pointer += 1
                # the repaired data is immediately used for the next columns.
                self.temp_df = self.res_df.copy()

            # detect errors after each iteration
            self.error_df = (self.res_df.ne(self.clean_df))
            
            # compute weight_used from weight_df after each iteration
            if self.use_weight:
                print(f'Computing weight_used after iteration {iteration + 1}\n')
                self.compute_weight_used()

            now_time = time.time()

            f1_score = self.get_f1()
            if self.use_weight:
                pass
            print(f'F1 score: {f1_score}')
            if f1_score - self.last_f1 < 0.005:
                break
            else:
                self.last_f1 = f1_score

        print('Done.\n')
        print(f'Time taken: {((time.time() - start_time)/60):.2f} minutes')

    def get_res(self):
        return self.res_df

    def get_f1(self):
        f1_score = F1_score(self.clean_df.astype(str), self.res_df.astype(str), self.dirty_df.astype(str))
        return f1_score

    def get_edr(self):
        edr = EDR(self.clean_df, self.dirty_df, self.res_df)
        return edr


if __name__ == "__main__":
    # Run with: python -m lasrepair.repair --experiment flight --gpu cuda:0
    args = argparse.ArgumentParser()
    args.add_argument("--experiment", type=str, default='flight')
    args.add_argument("--gpu", type=str, default='cuda:0')
    args.add_argument("--batch_size", type=int, default=8)
    args.add_argument("--epochs", type=int, default=3)
    args.add_argument("--max_iteration", type=int, default=10)
    args.add_argument("--use_weight", type=bool, default=True)
    args.add_argument("--sample_num", type=int, default=0)
    args.add_argument("--temperature", type=float, default=1.0)
    args.add_argument("--correct_prop", type=float, default=0.1)
    args.add_argument("--sml_name", type=str, default='google-t5/t5-large')
    args.add_argument("--llm_name", type=str, default='gpt-5')
    args.add_argument("--error_rate", type=float, default=0)
    args.add_argument("--data_dir", type=Path, default=DEFAULT_DATASETS_DIR)
    args.add_argument("--output_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = args.parse_args()

    # modify the path to run on your own dataset
    dirty_path = args.data_dir / args.experiment / 'dirty.csv'
    if args.error_rate != 0:
        dirty_path = args.data_dir / args.experiment / (args.experiment + '_' + str(int(args.error_rate*100)) + '_error.csv')
    clean_path = args.data_dir / args.experiment / 'clean.csv'
    print(f"arguments:\n weight: {args.use_weight}, sample_num: {args.sample_num}")
    print(f"batch_size: {args.batch_size}, epochs: {args.epochs}, max_iteration: {args.max_iteration}, temperature: {args.temperature}")

    a = MultiModelIterativeGenerativeRepair(dirty_path, clean_path, args)
    a.run()
    r = a.get_res()
    if args.error_rate == 0:
        path = args.output_dir / (args.experiment + '_repaired_original.csv')
    else: 
        path = args.output_dir / (args.experiment + '_' + str(int(args.error_rate*100)) + '_error_repaired.csv')
    r.to_csv(path, index=False)
    print(f"result saved to {path}")
    true = a.clean_df
    dirty = a.dirty_df
    f1_score = F1_score(true, r, dirty)
    print(args.experiment)
    print(r.head())
    print(f"F1 score: {f1_score}")
    print(f"running done at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
