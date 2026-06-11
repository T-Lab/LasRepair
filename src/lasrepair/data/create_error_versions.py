import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from ..paths import DEFAULT_DATASETS_DIR


def calculate_error_rate(error_df, correct_df):
    """计算错误率"""
    error_key = error_df.columns[0]
    correct_key = correct_df.columns[0]
    
    error_dict = {row[error_key]: row for _, row in error_df.iterrows()}
    correct_dict = {row[correct_key]: row for _, row in correct_df.iterrows()}
    
    error_count = 0
    total_cells = 0
    
    for key in error_dict.keys():
        if key in correct_dict:
            error_row = error_dict[key]
            correct_row = correct_dict[key]
            
            for col in error_df.columns[1:]:
                if col in correct_df.columns:
                    total_cells += 1
                    error_val = str(error_row[col]).strip() if pd.notna(error_row[col]) else ""
                    correct_val = str(correct_row[col]).strip() if pd.notna(correct_row[col]) else ""
                    if error_val != correct_val:
                        error_count += 1
    
    return error_count / total_cells if total_cells > 0 else 0


def repair_dataset(error_df, correct_df, target_error_rate):
    """修复数据集到目标错误率"""
    repaired_df = error_df.copy()
    error_key = error_df.columns[0]
    correct_key = correct_df.columns[0]
    
    error_dict = {row[error_key]: (idx, row) for idx, row in error_df.iterrows()}
    correct_dict = {row[correct_key]: row for _, row in correct_df.iterrows()}
    
    errors_to_fix = []
    for key in error_dict.keys():
        if key in correct_dict:
            idx, error_row = error_dict[key]
            correct_row = correct_dict[key]
            for col in error_df.columns[1:]:
                if col in correct_df.columns:
                    error_val = str(error_row[col]).strip() if pd.notna(error_row[col]) else ""
                    correct_val = str(correct_row[col]).strip() if pd.notna(correct_row[col]) else ""
                    if error_val != correct_val:
                        errors_to_fix.append((idx, col, correct_row[col]))
    
    current_error_rate = calculate_error_rate(repaired_df, correct_df)
    total_cells = len([c for c in error_df.columns[1:] if c in correct_df.columns]) * len(error_dict)
    target_error_count = int(target_error_rate * total_cells)
    current_error_count = int(current_error_rate * total_cells)
    fix_count = current_error_count - target_error_count
    
    if fix_count > 0:
        np.random.seed(42)
        selected = np.random.choice(len(errors_to_fix), min(fix_count, len(errors_to_fix)), replace=False)
        for i in selected:
            idx, col, val = errors_to_fix[i]
            repaired_df.at[idx, col] = val
    
    return repaired_df, calculate_error_rate(repaired_df, correct_df)


# 示例使用
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="hospital")
    parser.add_argument("--source_error_rate", type=int, default=52)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATASETS_DIR)
    args = parser.parse_args()

    # 读取数据集
    dataset_dir = args.data_dir / args.experiment
    error_df = pd.read_csv(dataset_dir / f"{args.experiment}_{args.source_error_rate}_error.csv")
    correct_df = pd.read_csv(dataset_dir / "clean.csv")
    error_df.columns = correct_df.columns
    
    # 1. 统计错误率
    error_rate = calculate_error_rate(error_df, correct_df)
    print(f"错误率: {error_rate:.4f} ({error_rate*100:.2f}%)")
    
    # 2. 修复到目标错误率（例如0.1，即10%）
    target_rates = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
    for target_rate in target_rates:
        repaired_df, final_rate = repair_dataset(error_df, correct_df, target_rate)
        print(f"修复后错误率: {final_rate:.4f} ({final_rate*100:.2f}%)")
        
        # 保存修复后的数据集
        repaired_df.to_csv(dataset_dir / f"{args.experiment}_{int(target_rate*100)}_error.csv", index=False)
