import pandas as pd
import numpy as np
import random
import string


def dataset_corrupter(dataset_path, error_rate, output_path, available_columns_index):
    """
    Read CSV, introduce errors, and save corrupted version.
    
    Args:
        dataset_path: Path to input CSV file
        error_rate: Rate of corruption (0-1)
        output_path: Path to save corrupted CSV
        available_columns_index: List of available columns index
    """
    # Step 1: Read the CSV file
    print(f"Reading CSV from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Step 2: Randomly pick entries from available columns
    num_rows = len(df)
    total_available_cells = num_rows * len(available_columns_index)
    num_errors = int(total_available_cells * error_rate)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Available columns: {available_columns_index}")
    print(f"Total available cells: {total_available_cells}")
    print(f"Number of cells to corrupt: {num_errors}")
    
    # Generate random positions to corrupt
    corrupted_positions = []
    for _ in range(num_errors):
        # Pick random row
        row_idx = random.randint(0, num_rows - 1)
        # Pick random column from available indices
        col_idx = random.choice(available_columns_index)
        corrupted_positions.append((row_idx, col_idx))
    
    print(f"Selected {len(corrupted_positions)} positions to corrupt")
    
    # Step 3: Apply corruption to selected positions
    df_corrupted = df.copy()
    corruption_types = ['missing', 'typo', 'format']
    corruption_applied = []
    
    for row_idx, col_idx in corrupted_positions:
        original_value = df_corrupted.iloc[row_idx, col_idx]
        corruption_type = random.choice(corruption_types)
        
        if corruption_type == 'missing':
            # 1. Missing: delete the value
            df_corrupted.iloc[row_idx, col_idx] = np.nan
            corruption_applied.append(f"Row {row_idx}, Col {col_idx}: '{original_value}' -> Missing")
            
        elif corruption_type == 'typo' and isinstance(original_value, str) and not pd.isna(original_value):
            # 2. Typo: change some letters if it's a string and not NaN
            corrupted_text = introduce_typo(original_value)
            df_corrupted.iloc[row_idx, col_idx] = corrupted_text
            corruption_applied.append(f"Row {row_idx}, Col {col_idx}: '{original_value}' -> '{corrupted_text}'")
            
        elif corruption_type == 'format' and isinstance(original_value, (int, float)) and not pd.isna(original_value):
            # 3. Format: change between integer and float if it's a number and not NaN
            if isinstance(original_value, int):
                # Convert int to float with decimal
                corrupted_value = float(original_value) + random.uniform(0.1, 0.9)
            else:
                # Convert float to int
                corrupted_value = int(original_value)
            df_corrupted.iloc[row_idx, col_idx] = corrupted_value
            corruption_applied.append(f"Row {row_idx}, Col {col_idx}: '{original_value}' -> '{corrupted_value}'")
        
        else:
            # If the corruption type doesn't match the data type, apply missing instead
            df_corrupted.iloc[row_idx, col_idx] = np.nan
            corruption_applied.append(f"Row {row_idx}, Col {col_idx}: '{original_value}' -> Missing (fallback)")
    
    print(f"\nApplied {len(corruption_applied)} corruptions")
    print("First 5 corruptions:")
    for corruption in corruption_applied[:5]:
        print(f"  {corruption}")
    
    # Step 4: Save the corrupted dataset
    print(f"\nSaving corrupted dataset to: {output_path}")
    df_corrupted.to_csv(output_path, index=False)
    print("✓ Corrupted dataset saved successfully!")
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Corrupted {len(corruption_applied)} cells ({error_rate*100}% error rate)")
    print(f"Output saved to: {output_path}")
    
    return df_corrupted, corruption_applied

def introduce_typo(text):
    """
    Introduce a random typo in the text by changing one character.
    """
    if len(text) == 0:
        return text
    
    text_list = list(text)
    # Pick a random position to corrupt
    pos = random.randint(0, len(text_list) - 1)
    
    if text_list[pos].isalpha():
        # Replace with a random letter
        if text_list[pos].islower():
            text_list[pos] = random.choice(string.ascii_lowercase)
        else:
            text_list[pos] = random.choice(string.ascii_uppercase)
    elif text_list[pos].isdigit():
        # Replace with a random digit
        text_list[pos] = random.choice(string.digits)
    else:
        # For other characters, replace with a random letter
        text_list[pos] = random.choice(string.ascii_lowercase)
    
    return ''.join(text_list)