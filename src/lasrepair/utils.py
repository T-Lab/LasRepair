import numpy as np
import pandas as pd

from .paths import DEFAULT_DATASETS_DIR, DEFAULT_RESULTS_DIR

# 支持相对导入和绝对导入
try:
    from .max_modularity import spectral_modularity_maximization, embedding2graph
except ImportError:
    from max_modularity import spectral_modularity_maximization, embedding2graph


def EDR(df_clean, df_dirty, df_repaired):
    """
    Calculate the error drop rate using edit distance.
    """
    original_error_count = df_dirty.ne(df_clean).sum(axis=0).sum()
    repaired_error_count = df_repaired.ne(df_clean).sum(axis=0).sum()
    return (original_error_count - repaired_error_count) / original_error_count

"""
df1: clean, df2: repaired, df3: original
"""
def RDRR(df_clean, df_dirty, df_repaired):
    """
    Calculate the error drop rate using edit distance.
    
    This metric measures how much the repair process reduced the distance
    to the clean data compared to the original dirty data.
    
    Args:
        df_clean (pd.DataFrame): Clean/ground truth dataset
        df_dirty (pd.DataFrame): Original dirty dataset
        df_repaired (pd.DataFrame): Repaired dataset
        
    Returns:
        float: Error drop rate as (d2-d1)/d1 where:
               d1 = edit_distance(clean, dirty)
               d2 = edit_distance(clean, repaired)
               
               Positive values indicate improvement (error reduction)
               Negative values indicate degradation (error increase)
    """
    # Calculate edit distance between clean and dirty (baseline error)
    d1 = edit_distance(df_clean, df_dirty)
    
    # Calculate edit distance between clean and repaired (remaining error)
    d2 = edit_distance(df_clean, df_repaired)
    
    # Calculate error drop rate: (d2-d1)/d1
    # Note: This is the negative of improvement rate
    # Positive values mean errors increased (bad)
    # Negative values mean errors decreased (good)
    if d1 == 0:
        # If clean and dirty are identical, return 0 or handle special case
        return 0.0 if d2 == 0 else float('inf')
    
    error_drop_rate = (d1 - d2) / d1
    return error_drop_rate

def casual_matrix_to_dict(matrix, column_names=None, threshold=0.5):
    """
    Convert upper triangular causal matrix to a dictionary.
    
    Args:
        matrix: Upper triangular causal matrix (numpy array)
        column_names: List of column names. If None, use indices.
        threshold: Minimum weight threshold (default 0.5)
    
    Returns:
        dict: {target_col: [related_cols]} considering 1-hop and 2-hop relationships
    """
    n = matrix.shape[0]
    
    # Use indices as column names if not provided
    if column_names is None:
        column_names = list(range(n))
    
    # Build symmetric matrix from upper triangular
    sym_matrix = matrix + matrix.T
    
    casual_dict = {}
    
    for i in range(n):
        related_cols = set()
        
        # 1-hop: direct connections
        for j in range(n):
            if i != j and sym_matrix[i, j] >= threshold:
                related_cols.add(column_names[j])
        
        # 2-hop: connections through intermediate nodes
        for k in range(n):  # intermediate node
            if i != k and sym_matrix[i, k] > 0:
                for j in range(n):
                    if i != j and j != k:
                        # Calculate 2-hop weight: i->k->j
                        hop2_weight = sym_matrix[i, k] * sym_matrix[k, j]
                        if hop2_weight >= threshold:
                            related_cols.add(column_names[j])
        
        casual_dict[column_names[i]] = sorted(list(related_cols))
    
    return casual_dict
    



def F1_score(df1, df2, df3):
    new_errors = df1.ne(df2)
    old_errors = df1.ne(df3)
    new_errors_count = int(new_errors.values.sum())
    old_errors_count = int(old_errors.values.sum())
    recall = (old_errors_count-new_errors_count) / old_errors_count
    f1 = 2 * recall / (1 + recall)
    return f1


def edit_distance(df1, df2):
    """
    Calculate the average normalized Levenshtein distance between two dataframes.
    
    Compares corresponding cells in two dataframes and calculates the normalized
    Levenshtein distance for each pair, then returns the average.
    
    Args:
        df1 (pd.DataFrame): First dataframe
        df2 (pd.DataFrame): Second dataframe
        
    Returns:
        float: Average normalized Levenshtein distance (0-1, where 0 means identical)
    """
    # Ensure dataframes have the same shape
    if df1.shape != df2.shape:
        raise ValueError(f"DataFrames must have the same shape. Got {df1.shape} and {df2.shape}")
    
    total_distance = 0
    total_comparisons = 0
    
    # Iterate through all cells in the dataframes
    for col in df1.columns:
        for idx in df1.index:
            s1 = str(df1.loc[idx, col])
            s2 = str(df2.loc[idx, col])
            
            # Calculate Levenshtein distance
            lev_dist = levenshtein_distance(s1, s2)
            
            # Normalize by the maximum length of the two strings
            max_len = max(len(s1), len(s2))
            if max_len > 0:
                normalized_dist = lev_dist / max_len
            else:
                # Both strings are empty, distance is 0
                normalized_dist = 0
            
            total_distance += normalized_dist
            total_comparisons += 1
    
    # Return average normalized distance
    if total_comparisons > 0:
        return total_distance / total_comparisons
    else:
        return 0


def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    
    The Levenshtein distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to change one string
    into another.
    
    Args:
        s1 (str): First string
        s2 (str): Second string
        
    Returns:
        int: The Levenshtein distance between s1 and s2
    """
    # Convert to strings in case non-string inputs are passed
    s1, s2 = str(s1), str(s2)
    
    # If one string is empty, return the length of the other
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    
    # Create a matrix to store distances
    rows = len(s1) + 1
    cols = len(s2) + 1
    dist = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Initialize first row and column
    for i in range(1, rows):
        dist[i][0] = i
    for j in range(1, cols):
        dist[0][j] = j
    
    # Fill the matrix
    for i in range(1, rows):
        for j in range(1, cols):
            if s1[i-1] == s2[j-1]:
                cost = 0
            else:
                cost = 1
            
            dist[i][j] = min(
                dist[i-1][j] + 1,      # deletion
                dist[i][j-1] + 1,      # insertion
                dist[i-1][j-1] + cost  # substitution
            )
    
    return dist[rows-1][cols-1]


def dataset_corrupter(dataset_path, error_rate, output_path):
    """
    Read CSV, introduce errors, and save corrupted version.
    
    Args:
        dataset_path: Path to input CSV file
        error_rate: Rate of corruption (0-1)
        output_path: Path to save corrupted CSV
    """
    # Step 1: Read the CSV file
    print(f"Reading CSV from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    
    return df

def test(a, b):
    return a + b

def group_by_modularity(df, sample_rows=30, dimensions=200, resolution=1):
    similarity_matrix = embedding2graph(df, num_rows=sample_rows, dimensions=dimensions)
    labels, Q = spectral_modularity_maximization(similarity_matrix, resolution=resolution)
    return labels

def all_wrong_corrector(clean_df, dirty_df, error_df, prop=0.2):
    """
    used to correct columns that are all wrong
    """
    col_sum = error_df.sum(axis=0)
    np.random.seed(114)
    for col in col_sum.index:
        # too many error, do correction
        if col_sum[col] > (1-prop) * error_df.shape[0]:
            print(f"correcting column {col} with prop {prop}")
            # not sample from the wrong, but sample from the all. The last rate may be lower than prop.
            correct_sample = np.random.choice(error_df.shape[0], int((1-prop) * error_df.shape[0]), replace=False)
            dirty_df.loc[correct_sample, col] = clean_df.loc[correct_sample, col]
        else:
            continue

    error_df = clean_df.ne(dirty_df)
    return clean_df, dirty_df, error_df


if __name__ == "__main__":
    # Load causal matrix
    # casual_matrix = np.load('path/to/rayyan_causal_matrix.npy')
    
    # # Load dataset to get column names
    # df = pd.read_csv(DEFAULT_DATASETS_DIR / 'rayyan' / 'clean.csv')
    # column_names = list(df.columns)
    
    # # Convert matrix to dict
    # casual_dict = casual_matrix_to_dict(casual_matrix, column_names=column_names, threshold=0.5)
    
    # print("Causal relationships (1-hop and 2-hop with threshold >= 0.5):")
    # for col, related in casual_dict.items():
    #     if related:  # Only print columns that have relationships
    #         print(f"{col}: {related}")
    # experiments = ["beers", "flight", "hospital", "tax_200k", "shuttle", "tax_20k", "walmart"]
    experiments = ["walmart"]
    for experiment in experiments:
        print(f"Experiment: {experiment}")
        clean_path = DEFAULT_DATASETS_DIR / experiment / "clean.csv"
        dirty_path = DEFAULT_DATASETS_DIR / experiment / "dirty.csv"
        repaired_path = DEFAULT_RESULTS_DIR / f"{experiment}_repaired_original.csv"
        clean_df = pd.read_csv(clean_path)
        dirty_df = pd.read_csv(dirty_path)
        dirty_df.columns = clean_df.columns
        repaired_df = pd.read_csv(repaired_path)
        repaired_df.columns = clean_df.columns
        print(f"EDRR: {EDRR(clean_df, dirty_df, repaired_df)}")
        print(f"RDRR: {RDRR(clean_df, dirty_df, repaired_df)}")
