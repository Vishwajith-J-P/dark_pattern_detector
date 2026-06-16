import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import (
    f1_score, hamming_loss, classification_report,
    multilabel_confusion_matrix, roc_auc_score, roc_curve
)
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset import load_dataset, binarize_labels, get_splits, PATTERN_LABELS
from training.train import DarkPatternDataset, BATCH_SIZE, MAX_LEN, THRESHOLD

CSV_PATH   = "training/data/dark_patterns.csv"
MODEL_PATH = "model/distilbert_finetuned"
RESULTS    = "training/results"
os.makedirs(RESULTS, exist_ok=True)

def evaluate():
    print("Loading test data...")
    df = load_dataset(CSV_PATH)
    label_matrix, mlb = binarize_labels(df)
    _, _, X_test, _, _, y_test = get_splits(df, label_matrix)

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model     = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    test_dataset = DarkPatternDataset(X_test, y_test, tokenizer)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            probs = torch.sigmoid(outputs.logits).numpy()
            preds = (probs >= THRESHOLD).astype(int)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(batch['labels'].int().numpy())

    all_probs  = np.vstack(all_probs)
    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # ── Text report ──
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                 target_names=PATTERN_LABELS, zero_division=0))
    print(f"Hamming Loss: {hamming_loss(all_labels, all_preds):.4f}")

    # ── Confusion matrices ──
    plot_confusion_matrices(all_labels, all_preds)

    # ── ROC curves ──
    plot_roc_curves(all_labels, all_probs)

def plot_confusion_matrices(y_true, y_pred):
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, (cm, label) in enumerate(zip(mcm, PATTERN_LABELS)):
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i],
                    cmap='Reds', cbar=False,
                    xticklabels=['Pred 0', 'Pred 1'],
                    yticklabels=['True 0', 'True 1'])
        axes[i].set_title(label.replace('_', ' ').title(), fontsize=9)

    plt.suptitle('Confusion Matrices per Dark Pattern Class', fontsize=14)
    plt.tight_layout()
    path = os.path.join(RESULTS, 'confusion_matrices.png')
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")

def plot_roc_curves(y_true, y_probs):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, label in enumerate(PATTERN_LABELS):
        if len(np.unique(y_true[:, i])) < 2:
            axes[i].set_title(f"{label}\n(no positive samples)")
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
        auc = roc_auc_score(y_true[:, i], y_probs[:, i])
        axes[i].plot(fpr, tpr, color='crimson', lw=2,
                     label=f'AUC = {auc:.2f}')
        axes[i].plot([0,1],[0,1],'k--', lw=1)
        axes[i].set_title(label.replace('_', ' ').title(), fontsize=9)
        axes[i].legend(fontsize=8)
        axes[i].set_xlabel('FPR'); axes[i].set_ylabel('TPR')

    plt.suptitle('ROC Curves per Dark Pattern Class', fontsize=14)
    plt.tight_layout()
    path = os.path.join(RESULTS, 'roc_curves.png')
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")

if __name__ == "__main__":
    evaluate()