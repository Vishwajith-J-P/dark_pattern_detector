import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# The 10 dark patterns we're targeting
PATTERN_LABELS = [
    "urgency",
    "scarcity",
    "confirmshaming",
    "hidden_costs",
    "forced_continuity",
    "misdirection",
    "price_comparison_prevention",
    "disguised_ads",
    "trick_questions",
    "social_proof_manipulation"
]

# Maps Princeton's category names → our labels
LABEL_MAPPING = {
    "urgency":       "urgency",
    "scarcity":      "scarcity",
    "social proof":  "social_proof_manipulation",
    "misdirection":  "misdirection",
    "obstruction":   "misdirection",        # closest match
    "sneaking":      "hidden_costs",         # hidden costs / sneaking charges
    "forced action": "forced_continuity",    # closest match
}

SUPPLEMENTARY_PATHS = [
    "training/data/supplementary.csv",
    "training/data/supplementary2.csv"
]

def load_supplementary(paths=SUPPLEMENTARY_PATHS):
    dfs = []
    for path in paths:
        try:
            df = pd.read_csv(path)
            df = df.dropna(subset=['text'])
            df['labels'] = df['label'].apply(lambda x: [x])
            df = df[df['label'].isin(PATTERN_LABELS)]
            print(f"Loaded {len(df)} samples from {path}")
            dfs.append(df)
        except FileNotFoundError:
            print(f"Not found: {path}, skipping.")
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        print(f"Total supplementary samples: {len(combined)}")
        return combined
    return pd.DataFrame()

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    # Rename columns to internal names
    df = df.rename(columns={
        'Pattern String':   'text',
        'Pattern Category': 'dark_pattern_type'
    })

    # Drop rows with missing text
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != '']

    # Normalize labels
    df['label'] = df['dark_pattern_type'].apply(normalize_label)

    # Drop rows that didn't map to any of our 10
    df = df[df['label'].notna()]

    # Wrap in list for multi-label binarizer
    df['labels'] = df['label'].apply(lambda x: [x])

    print(f"Dataset loaded: {len(df)} samples")
    print(df['label'].value_counts())

    supp = load_supplementary()
    if not supp.empty:
        df = pd.concat([df, supp], ignore_index=True)
        print(f"Total after merge: {len(df)} samples")

    return df

def normalize_label(raw_label):
    if not isinstance(raw_label, str):
        return None
    return LABEL_MAPPING.get(raw_label.lower().strip(), None)

def binarize_labels(df):
    mlb = MultiLabelBinarizer(classes=PATTERN_LABELS)
    label_matrix = mlb.fit_transform(df['labels'])
    return label_matrix, mlb

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def get_splits(df, label_matrix, test_size=0.2, val_size=0.1):
    texts = df['text'].tolist()

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        # First split: train+val vs test
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        for train_val_idx, test_idx in msss.split(texts, label_matrix):
            X_train_val = [texts[i] for i in train_val_idx]
            y_train_val = label_matrix[train_val_idx]
            X_test       = [texts[i] for i in test_idx]
            y_test        = label_matrix[test_idx]

        # Second split: train vs val
        msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
        for train_idx, val_idx in msss2.split(X_train_val, y_train_val):
            X_train = [X_train_val[i] for i in train_idx]
            y_train  = y_train_val[train_idx]
            X_val    = [X_train_val[i] for i in val_idx]
            y_val    = y_train_val[val_idx]

    except ImportError:
        # Fallback to regular split if library not installed
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            texts, label_matrix, test_size=test_size, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42
        )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test