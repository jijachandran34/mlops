"""
Create poisoned versions of IRIS dataset at different levels
"""
import pandas as pd
import numpy as np
import os
import argparse

def poison_data(input_file, output_file, poison_level):
    """
    Poison data by adding random noise to features
    
    Args:
        input_file: Path to clean data
        output_file: Path to save poisoned data
        poison_level: Percentage of data to poison (0-100)
    """
    df = pd.read_csv(input_file)
    
    # Identify target column (could be 'target', 'species', 'class', etc.)
    possible_targets = ['target', 'species', 'class', 'label', 'y']
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    
    # Get only numeric feature columns (exclude target)
    if target_col:
        feature_cols = [col for col in df.columns if col != target_col]
    else:
        # If no known target, use all columns
        feature_cols = df.columns.tolist()
    
    # Filter to only numeric columns
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print("Warning: No numeric columns found to poison!")
        df.to_csv(output_file, index=False)
        return
    
    # Calculate number of samples to poison
    n_samples = len(df)
    n_poison = int(n_samples * poison_level / 100)
    
    if n_poison == 0:
        print("Warning: Poison level too low, no samples to poison")
        df.to_csv(output_file, index=False)
        return
    
    # Randomly select samples to poison
    poison_indices = np.random.choice(n_samples, n_poison, replace=False)
    
    # Poison selected samples by replacing with random values
    for idx in poison_indices:
        for col in numeric_cols:
            # Generate random values in the range of the feature
            min_val = df[col].min()
            max_val = df[col].max()
            df.loc[idx, col] = np.random.uniform(min_val, max_val)
    
    # Save poisoned data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"Poisoned {n_poison}/{n_samples} samples ({poison_level}%)")
    print(f"Target column: {target_col if target_col else 'None identified'}")
    print(f"Numeric features poisoned: {numeric_cols}")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/iris_raw.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--poison-level", type=float, required=True)
    args = parser.parse_args()
    
    poison_data(args.input, args.output, args.poison_level)