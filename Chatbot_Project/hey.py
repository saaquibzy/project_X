import pandas as pd

# Load dataset
df = pd.read_json("Sarcasm_Headlines_Dataset_v2/Sarcasm_Headlines_Dataset_v2.json", lines=True)


# Print the first 5 rows and column names
print("\nDataset Preview:\n", df.head())
print("\nColumn Names:", df.columns)

# Ensure required columns exist
expected_columns = {"headline", "is_sarcastic"}
if not expected_columns.issubset(df.columns):
    raise ValueError(f"❌ Missing required columns! Found: {df.columns}")

# Rename 'is_sarcastic' to 'label' for model compatibility
df = df.rename(columns={"is_sarcastic": "label"})
df = df[["headline", "label"]]  # Keep only relevant columns

# Print final dataset
print("\nFinal Dataset Preview:\n", df.head())
