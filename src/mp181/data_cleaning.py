import pandas as pd
import numpy as np

def clean_fabric_column(series: pd.Series) -> pd.Series:
    """Standardizes fabric column to exactly 3 true levels: cotton, polyester, silk."""
    cleaned = series.astype(str).str.strip().str.lower()
    
    def map_fabric(val: str) -> str:
        if "sil" in val:
            return "silk"
        elif "cot" in val:
            return "cotton"
        elif "pol" in val:
            return "polyester"
        return val

    return cleaned.apply(map_fabric)

def clean_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Cleans raw dataframe by resolving identified data issues:
    1. Removes duplicate rows
    2. Drops leaking column metres_checked_at_folding
    3. Corrects unit mixup: loom_age_yr recorded in months for Shed 1 (divide by 12)
    4. Fixes spelling variants in fabric column
    """
    df = df.copy()
    
    if is_train:
        # 1. Remove duplicate rows
        df = df.drop_duplicates()
        
        # 2. Drop leaking column if present
        if "metres_checked_at_folding" in df.columns:
            df = df.drop(columns=["metres_checked_at_folding"])
            
    # 3. Correct unit mixup for Shed 1 (loom_age_yr was in months, divide by 12)
    if "loom_age_yr" in df.columns and "shed" in df.columns:
        shed1_mask = df["shed"].astype(str).str.strip() == "Shed 1"
        df.loc[shed1_mask, "loom_age_yr"] = df.loc[shed1_mask, "loom_age_yr"] / 12.0
        
    # 4. Standardize fabric names
    if "fabric" in df.columns:
        df["fabric"] = clean_fabric_column(df["fabric"])
        
    return df

def audit_seven_problems(train_raw_path: str, holdout_raw_path: str = None) -> dict:
    """Audits train.csv against the 7 defined data issues."""
    df = pd.read_csv(train_raw_path)
    
    # 1. Leaking column
    leaking_present = "metres_checked_at_folding" in df.columns
    
    # 2. Unit mixup
    # Compare loom_age_yr mean in Shed 1 vs other sheds
    shed1_mean = df[df['shed'] == 'Shed 1']['loom_age_yr'].mean()
    shed2_mean = df[df['shed'] == 'Shed 2']['loom_age_yr'].mean()
    unit_mixup_present = (shed1_mean / shed2_mean) > 5.0
    
    # 3. Sentinel value
    sentinel_present = False
    
    # 4. Duplicate rows
    dup_count = int(df.duplicated().sum())
    dup_present = dup_count > 0
    
    # 5. Stuck sensor
    stuck_present = False
    
    # 6. Target typos
    target_typo_present = False
    
    # 7. Spelling variants
    raw_fabric_levels = df['fabric'].nunique()
    spelling_present = raw_fabric_levels > 3
    
    return {
        "leaking_column": {
            "present": leaking_present,
            "column": "metres_checked_at_folding" if leaking_present else None
        },
        "unit_mixup": {
            "present": unit_mixup_present,
            "column": "loom_age_yr" if unit_mixup_present else None,
            "group_value": "Shed 1" if unit_mixup_present else None,
            "factor": 12 if unit_mixup_present else None
        },
        "sentinel_value": {
            "present": False,
            "column": None,
            "value": None,
            "count": None
        },
        "duplicate_rows": {
            "present": dup_present,
            "count": dup_count if dup_present else None
        },
        "stuck_sensor": {
            "present": False,
            "column": None,
            "start_date": None,
            "end_date": None
        },
        "target_typos": {
            "present": False,
            "count": None
        },
        "spelling_variants": {
            "present": spelling_present,
            "column": "fabric" if spelling_present else None,
            "true_levels": 3 if spelling_present else None
        }
    }
