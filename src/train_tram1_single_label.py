"""
TRAM 1 Single-Label MITRE ATT&CK Technique Classification Pipeline.

Fine-tunes a domain-adapted BERT encoder (`nlpaueb/sec-bert-base` or `jackaduma/SecBERT`)
on the TRAM 1 sentence dataset (5,089 raw -> 4,837 deduplicated sentences, 50 ATT&CK
techniques, 171x imbalance ratio) using inverse-frequency Class-Weighted Cross-Entropy.

Reproduces Paper Table 3 & Table 5 (TRAM 1):
- Test Micro-F1 / Accuracy : 0.6818
- Test Macro-F1            : 0.6160
- Top Error Pair           : T1027 -> T1140 (23x) & T1027 -> T1573.001 (18x) (26.6% of errors)
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TRAM1Dataset(Dataset):
    """PyTorch Dataset for single-label TRAM 1 CTI sentences."""

    def __init__(self, texts, labels, tokenizer, max_length: int = 128) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class WeightedTrainer(Trainer):
    """Hugging Face Trainer with inverse-frequency class-weighted Cross-Entropy (Eq. 1)."""

    def __init__(self, class_weights_tensor: torch.Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights_tensor = class_weights_tensor

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "micro_f1": float(f1_score(labels, preds, average="micro")),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "accuracy": float(accuracy_score(labels, preds)),
    }


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train & evaluate SecBERT on TRAM 1 (Single-Label 50 ATT&CK classes)."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=root_dir / "dataset" / "single_label.json",
        help="Path to single_label.json.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="nlpaueb/sec-bert-base",
        help="Hugging Face model checkpoint (e.g. nlpaueb/sec-bert-base or jackaduma/SecBERT).",
    )
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=root_dir / "tram1_export",
        help="Directory to save model artifacts, metrics, and plots.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[TRAM 1] Device: {device} | Backbone: {args.model_name}")

    # 1. Load & Deduplicate Data
    with args.data_path.open("r", encoding="utf-8") as fh:
        raw_data = json.load(fh)
    df = pd.DataFrame(raw_data)
    df_clean = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    print(
        f"[TRAM 1] Raw: {len(df)} -> Deduplicated: {len(df_clean)} "
        f"(removed {len(df) - len(df_clean)} exact duplicates)"
    )

    # 2. Label Encoding & Stratified 80/10/10 Split
    le = LabelEncoder()
    df_clean["label_id"] = le.fit_transform(df_clean["label"])
    num_labels = len(le.classes_)

    X = df_clean["text"].values
    y = df_clean["label_id"].values
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=args.seed
    )
    try:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=args.seed
        )
    except ValueError:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=args.seed
        )

    print(
        f"[TRAM 1] Split sizes — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
    )

    # 3. Balanced Class Weights (Eq. 1: w_c = N / (K * n_c))
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # 4. Tokenization & Datasets
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = TRAM1Dataset(X_train, y_train, tokenizer, args.max_length)
    val_dataset = TRAM1Dataset(X_val, y_val, tokenizer, args.max_length)
    test_dataset = TRAM1Dataset(X_test, y_test, tokenizer, args.max_length)

    # 5. Model & Weighted Trainer
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels, ignore_mismatched_sizes=True
    ).to(device)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights_tensor=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("[TRAM 1] Starting fine-tuning...")
    trainer.train()

    # 6. Test Set Evaluation & Error Analysis
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    micro_f1 = float(f1_score(y_true, y_pred, average="micro"))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    acc = float(accuracy_score(y_true, y_pred))

    print(
        f"\n=== [TRAM 1] Test Set Results ===\n"
        f"  Accuracy / Micro-F1 : {micro_f1:.4f}\n"
        f"  Macro-F1            : {macro_f1:.4f}"
    )

    report_str = classification_report(
        y_true, y_pred, target_names=le.classes_, digits=4, zero_division=0
    )
    (args.output_dir / "tram1_classification_report.txt").write_text(
        f"Test Accuracy : {acc:.4f}\nMicro F1      : {micro_f1:.4f}\nMacro F1      : {macro_f1:.4f}\n\n{report_str}",
        encoding="utf-8",
    )

    # Top-15 Misclassification Pairs
    errors = [
        (le.classes_[y_true[i]], le.classes_[y_pred[i]])
        for i in range(len(y_true))
        if y_true[i] != y_pred[i]
    ]
    top15_errors = Counter(errors).most_common(15)
    error_labels = [f"{t} -> {p}" for (t, p), _ in top15_errors]
    error_values = [c for _, c in top15_errors]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(error_labels[::-1], error_values[::-1], color="tomato")
    ax.set_title("Top 15 Misclassifications (True -> Predicted)")
    ax.set_xlabel("Count")
    for i, v in enumerate(error_values[::-1]):
        ax.text(v + 0.1, i, str(v), va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(
        args.output_dir / "tram1_cell7_plot2_top_errors.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        linewidths=0.3,
        ax=ax,
    )
    ax.set_title("Confusion Matrix — SecBERT on TRAM1 Test Set", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(
        args.output_dir / "tram1_cell7_plot1_confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    metadata = {
        "model_name": args.model_name,
        "task": "single-label classification (TRAM 1)",
        "num_labels": num_labels,
        "max_length": args.max_length,
        "test_accuracy": round(acc, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "epochs": args.epochs,
        "top_error_pairs": [
            {"true": t, "pred": p, "count": c} for (t, p), c in top15_errors
        ],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[TRAM 1] Saved reports and metadata to {args.output_dir}")


if __name__ == "__main__":
    main()
