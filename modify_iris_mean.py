import pandas as pd
import os

input_file = "iris.csv"
output_file = "iris_2.csv"

# Ensure the file exists
if not os.path.exists(input_file):
    raise FileNotFoundError(f"File '{input_file}' not found in current directory.")

# Read the iris dataset
df = pd.read_csv(input_file)

# Check if 'petal_length' column exists
if "petal_length" not in df.columns:
    raise KeyError("Column 'petal_length' not found in iris.csv")

# Compute mean petal length
mean_petal_length = df["petal_length"].mean()

# Add new column with mean value
df["mean_petal_length"] = mean_petal_length

# Save to a new CSV file
df.to_csv(output_file, index=False)

print(f"✅ Added mean petal length ({mean_petal_length:.2f}) to all rows and saved as '{output_file}'")
