import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.metrics import f1_score, hamming_loss
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset import (
    load_dataset, binarize_labels, get_splits, PATTERN_LABELS
)

# ── Config ───────────────────────────────────────────────────────────────────
CSV_PATH   = "training/data/dark_patterns.csv"
MODEL_SAVE = "model/distilbert_finetuned"
NUM_LABELS = len(PATTERN_LABELS)
MAX_LEN    = 64
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 3e-5
THRESHOLD  = 0.4
# ─────────────────────────────────────────────────────────────────────────────

class DarkPatternDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels':         self.labels[idx]
        }

def train():
    # ── Load data ──
    print("Loading dataset...")
    df = load_dataset(CSV_PATH)
    label_matrix, mlb = binarize_labels(df)
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(df, label_matrix)

    # ── Load tokenizer and model ──
    print("Loading tokenizer and model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    )

    # ── Build dataloaders ──
    train_dataset = DarkPatternDataset(X_train, y_train, tokenizer)
    val_dataset   = DarkPatternDataset(X_val,   y_val,   tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    # ── Optimizer and scheduler ──
    optimizer   = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_scheduler("linear", optimizer=optimizer,
                                num_warmup_steps=0,
                                num_training_steps=total_steps)

    # ── Training loop ──
    best_f1          = 0
    patience         = 3
    patience_counter = 0

    print(f"\nTraining on CPU for up to {EPOCHS} epochs...")
    for epoch in range(EPOCHS):

        # Training
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                probs = torch.sigmoid(outputs.logits)
                preds = (probs >= THRESHOLD).int().numpy()
                all_preds.append(preds)
                all_labels.append(batch['labels'].int().numpy())

        all_preds  = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_hl = hamming_loss(all_labels, all_preds)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | Hamming Loss: {val_hl:.4f}")

        # Early stopping
        if val_f1 > best_f1:
            best_f1          = val_f1
            patience_counter = 0
            model.save_pretrained(MODEL_SAVE)
            tokenizer.save_pretrained(MODEL_SAVE)
            print(f"  ✓ New best F1: {best_f1:.4f} — model saved")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print(f"\nBest Val F1: {best_f1:.4f}")
    print("Training complete.")

if __name__ == "__main__":
    train()