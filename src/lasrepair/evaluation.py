import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from .paths import DEFAULT_DATASETS_DIR, DEFAULT_RESULTS_DIR
from .utils import F1_score, all_wrong_corrector, levenshtein_distance, edit_distance


def show_diff(repaired_path, clean_path, dirty_path):
    repaired_df = pd.read_csv(repaired_path, na_values=["nan", "NaN", "N/A", "None", "null"]).replace(np.nan, '').astype(str)
    clean_df = pd.read_csv(clean_path, na_values=["nan", "NaN", "N/A", "None", "null"]).replace(np.nan, '').astype(str)
    dirty_df = pd.read_csv(dirty_path, na_values=["nan", "NaN", "N/A", "None", "null"]).replace(np.nan, '').astype(str)
    repaired_df.columns = clean_df.columns
    dirty_df.columns = clean_df.columns
    column_names = repaired_df.columns
    count = 0
    # print(f"F1 score: {F1_score(clean_df, repaired_df, dirty_df)}")
    for row in range(repaired_df.shape[0]):
        for col in range(repaired_df.shape[1]):
            if repaired_df.iloc[row, col] != clean_df.iloc[row, col]:
                print(f"clean: {clean_df.iloc[row, col]}, repaired: {repaired_df.iloc[row, col]}, dirty: {dirty_df.iloc[row, col]}, column: {column_names[col]}")
                count += 1
                if count > 20:
                    return 0

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="tax_200k")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    experiment = args.experiment
    repaired_path = args.results_dir / f"{experiment}_repaired_original.csv"
    clean_path = args.data_dir / experiment / "clean.csv"
    dirty_path = args.data_dir / experiment / "dirty.csv"
    repaired_df = pd.read_csv(repaired_path, na_values=["nan", "NaN", "N/A", "None", "null"]).replace(np.nan, '').astype(str)
    clean_df = pd.read_csv(clean_path, na_values=["nan", "NaN", "N/A", "None", "null"]).replace(np.nan, '').astype(str)
    dirty_df = pd.read_csv(dirty_path, na_values=["nan", "NaN", "N/A", "None", "null"]).replace(np.nan, '').astype(str)
    repaired_df.columns = clean_df.columns
    dirty_df.columns = clean_df.columns
    # show_diff(repaired_path, clean_path, dirty_path)
    ori = edit_distance(clean_df, dirty_df)
    rep = edit_distance(clean_df, repaired_df)
    print(f"original: {ori}, repaired: {rep}, improvement: {(ori - rep) / ori}")
