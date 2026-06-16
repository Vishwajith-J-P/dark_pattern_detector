import json
import torch
import numpy as np
import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

from training.dataset import (
    load_dataset,
    binarize_labels,
    get_splits,
    PATTERN_LABELS
)

from sklearn.metrics import (
    classification_report,
    accuracy_score
)

# =====================================================
# Load Dataset
# =====================================================

df = load_dataset(
    "training/data/dark_patterns.csv"
)

label_matrix, mlb = binarize_labels(df)

(
    _,
    _,
    X_test,
    _,
    _,
    y_test
) = get_splits(
    df,
    label_matrix
)

# =====================================================
# Load Model
# =====================================================

MODEL_PATH = "model/distilbert_finetuned"

tokenizer = DistilBertTokenizerFast.from_pretrained(
    MODEL_PATH
)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

# =====================================================
# Tokenize Test Data
# =====================================================

encodings = tokenizer(
    X_test,
    truncation=True,
    padding=True,
    max_length=64,
    return_tensors="pt"
)

# =====================================================
# Predict
# =====================================================

with torch.no_grad():

    logits = model(
        **encodings
    ).logits

predictions = (
    torch.sigmoid(logits) >= 0.4
).int().numpy()

# =====================================================
# Classification Report
# =====================================================

report = classification_report(
    y_test,
    predictions,
    target_names=PATTERN_LABELS,
    zero_division=0,
    output_dict=True
)

# =====================================================
# Overall Metrics
# =====================================================

weighted = report["weighted avg"]

subset_accuracy = accuracy_score(
    y_test,
    predictions
)

overall_metrics = {

    "accuracy": round(
        subset_accuracy * 100,
        2
    ),

    "precision": round(
        weighted["precision"] * 100,
        2
    ),

    "recall": round(
        weighted["recall"] * 100,
        2
    ),

    "f1": round(
        weighted["f1-score"] * 100,
        2
    )
}

# =====================================================
# Per-Class Metrics
# =====================================================

class_metrics = {}

for label in PATTERN_LABELS:

    class_metrics[label] = {

        "precision": round(
            report[label]["precision"] * 100,
            2
        ),

        "recall": round(
            report[label]["recall"] * 100,
            2
        ),

        "f1": round(
            report[label]["f1-score"] * 100,
            2
        ),

        "support": int(
            report[label]["support"]
        )
    }

# =====================================================
# Save Metrics
# =====================================================

os.makedirs(
    "model_metrics",
    exist_ok=True
)

with open(
    "model_metrics/model_metrics.json",
    "w"
) as f:

    json.dump(
        overall_metrics,
        f,
        indent=4
    )

with open(
    "model_metrics/class_metrics.json",
    "w"
) as f:

    json.dump(
        class_metrics,
        f,
        indent=4
    )

# =====================================================
# Terminal Output
# =====================================================

print("\n========== OVERALL METRICS ==========\n")

print(
    json.dumps(
        overall_metrics,
        indent=4
    )
)

print("\n========== PER CLASS METRICS ==========\n")

for label in PATTERN_LABELS:

    print(
        f"{label:35} "
        f"F1={class_metrics[label]['f1']:6.2f}% "
        f"P={class_metrics[label]['precision']:6.2f}% "
        f"R={class_metrics[label]['recall']:6.2f}%"
    )

print("\nMetrics saved successfully.")

