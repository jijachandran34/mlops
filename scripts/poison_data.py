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
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col != 'target']
    
    # Calculate number of samples to poison
    n_samples = len(df)
    n_poison = int(n_samples * poison_level / 100)
    
    # Randomly select samples to poison
    poison_indices = np.random.choice(n_samples, n_poison, replace=False)
    
    # Poison selected samples by replacing with random values
    for idx in poison_indices:
        for col in feature_cols:
            # Generate random values in the range of the feature
            min_val = df[col].min()
            max_val = df[col].max()
            df.loc[idx, col] = np.random.uniform(min_val, max_val)
    
    # Save poisoned data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"Poisoned {n_poison}/{n_samples} samples ({poison_level}%)")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/iris_raw.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--poison-level", type=float, required=True)
    args = parser.parse_args()
    
    poison_data(args.input, args.output, args.poison_level)
