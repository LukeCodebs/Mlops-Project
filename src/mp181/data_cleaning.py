"""
MP181 - Data Analysis & Cleaning
Group G037 - Power Loom Output Prediction

Run this script to find all 7 data problems in train.csv.
Usage: python -m src.mp181.data_cleaning
"""

import pandas as pd
import numpy as np

# ── Load data ────────────────────────────────────────────────────────────────
train   = pd.read_csv("data/raw/train.csv")
holdout = pd.read_csv("data/raw/holdout_features.csv")
later   = pd.read_csv("data/raw/later.csv")

print("=" * 60)
print(f"Train shape : {train.shape}")
print(f"Later shape : {later.shape}")
print(f"Holdout shape: {holdout.shape}")

# ── 1. LEAKING COLUMN ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("1. LEAKING COLUMN")
missing_from_holdout = [c for c in train.columns if c not in holdout.columns]
print(f"   Columns in train but NOT in holdout: {missing_from_holdout}")
# Check correlation with target
numeric = train.select_dtypes(include="number")
corr = numeric.corr()["metres_per_shift"].drop("metres_per_shift").abs().sort_values(ascending=False)
print("   Correlation with target:")
print(corr.to_string())

# ── 2. UNIT MIXUP ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. UNIT MIXUP  (loom_age_yr per shed)")
print(train.groupby("shed")["loom_age_yr"].describe().round(2).to_string())

# ── 3. SENTINEL VALUES ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. SENTINEL VALUES")
num_cols = ["loom_age_yr", "thread_count", "power_cuts_min",
            "weaver_exp_yr", "humidity_pct", "shift_hours"]
for col in num_cols:
    vc = train[col].value_counts()
    suspicious = vc[(vc.index < 0) | (vc.index >= 900)]
    if not suspicious.empty:
        print(f"   ⚠  {col}: {suspicious.to_dict()}")
    # Also flag -999, -1, 999, 9999
    for val in [-1, -999, 999, 9999]:
        count = (train[col] == val).sum()
        if count > 0:
            print(f"   ⚠  {col} has sentinel {val} appearing {count} times")
print("   (no output above = no sentinel values found)")

# ── 4. DUPLICATE ROWS ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. DUPLICATE ROWS")
dups = train.duplicated().sum()
print(f"   Extra duplicate rows: {dups}")
if dups > 0:
    print(train[train.duplicated(keep=False)].head(4).to_string())

# ── 5. STUCK SENSOR ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. STUCK SENSOR")
train_s = train.sort_values("date").reset_index(drop=True)
for col in num_cols:
    grp = (train_s[col] != train_s[col].shift()).cumsum()
    runs = train_s.groupby(grp).agg(
        value=(col, "first"),
        count=(col, "count"),
        start=("date", "first"),
        end=("date", "last")
    )
    long = runs[runs["count"] >= 10]
    if not long.empty:
        print(f"   ⚠  STUCK: {col}")
        print(long.to_string())

# ── 6. TARGET TYPOS ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. TARGET TYPOS  (metres_per_shift)")
print(f"   describe:\n{train['metres_per_shift'].describe().round(2).to_string()}")
print(f"\n   Top 15 largest values:")
print(train["metres_per_shift"].nlargest(15).values)

# ── 7. SPELLING VARIANTS ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. SPELLING VARIANTS")
for col in train.select_dtypes(include="object").columns:
    if col in ["record_id", "date"]:
        continue
    print(f"\n   {col} value_counts:")
    print(train[col].value_counts().to_string())
    if col in later.columns:
        print(f"   {col} in later (clean reference):")
        print(later[col].value_counts().to_string())

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
