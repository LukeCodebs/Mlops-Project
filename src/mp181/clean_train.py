from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/train.csv")
OUTPUT_PATH = Path("data/cleaned_train.csv")

df = pd.read_csv(RAW_PATH)

print(f"Original dataset: {df.shape}")

# 1. Remove exact duplicate rows
df = df.drop_duplicates().copy()

# 2. Correct Shed 1 loom age (months -> years)
shed1 = df["shed"].str.strip().eq("Shed 1")
df.loc[shed1, "loom_age_yr"] = (
    df.loc[shed1, "loom_age_yr"] / 12
)

# 3. Standardize fabric spelling
fabric_map = {
    "cotton": "cotton",
    "cot.": "cotton",
    "silk": "silk",
    "sil.": "silk",
    "polyester": "polyester",
    "pol.": "polyester",
}

df["fabric"] = (
    df["fabric"]
    .str.strip()
    .str.lower()
    .map(fabric_map)
)

# 4. Remove known leakage column from training dataset
df = df.drop(columns=["metres_checked_at_folding"])

# 5. Validate cleaned dataset
assert df["fabric"].notna().all(), "Unknown fabric category found"
assert df["loom_age_yr"].between(0, 30).all(), "Invalid loom age"
assert df["metres_per_shift"].notna().all(), "Missing target"
assert not df.duplicated().any(), "Duplicates remain"

# 6. Save cleaned data
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Cleaned dataset: {df.shape}")
print(f"Saved to: {OUTPUT_PATH}")
print("Fabric categories:", df["fabric"].unique().tolist())
print("Cleaning completed successfully!")