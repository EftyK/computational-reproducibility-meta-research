import pandas as pd
import numpy as np

# === Configuration ===
input_csv = "../Harmonised_Data.csv"     # Replace with your actual file name
output_csv = "stratified_sample.csv"
sample_fraction = 0.10          # 10% sample

# === Read the CSV ===
df = pd.read_csv(input_csv)

# === Validate mutually exclusive 'yes' among columns 4, 5, 6 ===
binary_cols = df.columns[3:6]
yes_counts = (df[binary_cols] == 'Yes').sum(axis=1)

# Raise error if any row has zero or more than one 'yes'
if not all(yes_counts == 1):
    invalid_rows = df[yes_counts != 1]
    raise ValueError(f"Invalid rows found where columns 4-6 are not mutually exclusive:\n{invalid_rows}")

# === Assign category based on which column has 'yes' ===
def determine_category(row):
    for col in binary_cols:
        if row[col] == 'Yes':
            return col
    return None  # Should never hit this due to prior validation

df['category'] = df.apply(determine_category, axis=1)

# === Stratified sampling: 10% from each category ===
stratified_sample = df.groupby('category', group_keys=False).apply(
    lambda x: x.sample(frac=sample_fraction, random_state=42)
)

# === Save the sample to a new CSV ===
stratified_sample.to_csv(output_csv, index=False)
print(f"Stratified sample of 10% per category saved to {output_csv}")