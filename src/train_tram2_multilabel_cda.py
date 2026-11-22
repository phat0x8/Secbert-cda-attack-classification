"""
TRAM 2 Multi-Label MITRE ATT&CK Classification Pipeline with Contextual Data
Augmentation (CDA), 1:1 Background Noise Downsampling, and Multi-Label Focal Loss.

Evaluates on TRAM 2 (`dataset/multi_label.json`: 19,178 raw sentences; 4,070 labelled
CTI sentences + 15,108 unlabelled background noise sentences = 78.8% noise; 50 classes):
1. 1:1 Background-Noise Downsampling -> 8,140 working corpus.
2. Iterative Stratification (`scikit-multilearn`) -> Train 6,513 / Val 814 / Test 814.
3. Baseline Model: Positive-Weighted BCE (`pos_weight` clipped at 50, Eq. 2), tau=0.50
   -> Micro-F1 = 0.3344, Macro-F1 = 0.2767, EMR = 0.3821.
4. Optimized Model: Contextual Data Augmentation (`nlpaug` MLM substitution for 24
   minority techniques < 50 samples -> +469 synthetic samples) + Multi-Label Focal Loss
   (alpha=0.25, gamma=2.0) + Threshold Tuning (tau=0.35)
   -> Micro-F1 = 0.5847 (+74.8%), Macro-F1 = 0.4826 (+74.5%), EMR = 0.6130.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from secbert_classifier import (
    ContextualDataAugmenter,
    FocalLoss,
    compute_clipped_pos_weights,
    evaluate_multilabel,
)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CTIMultiLabelDataset(Dataset):
    """PyTorch Dataset for multi-label CTI sentence classification."""

    def __init__(self, texts, labels, tokenizer, max_length: int = 256) -> None:
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
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train & evaluate SecBERT + CDA + Focal Loss on TRAM 2 (Multi-Label 50 ATT&CK classes)."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=root_dir / "dataset" / "multi_label.json",
        help="Path to TRAM 2 multi_label.json.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="nlpaueb/sec-bert-base",
        help="Hugging Face checkpoint (default: nlpaueb/sec-bert-base).",
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--minority_threshold", type=int, default=50)
    parser.add_argument("--eval_threshold", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=root_dir / "tram2_export",
        help="Directory to save model checkpoints, reports, and figures.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[TRAM 2] Device: {device} | Backbone: {args.model_name}")

    # 1. Load Raw Data & Encode Multi-Hot Labels
    with args.data_path.open("r", encoding="utf-8") as fh:
        raw_records = json.load(fh)
    df_raw = pd.DataFrame(raw_records)
    df_raw["num_labels"] = df_raw["labels"].apply(len)

    mlb = MultiLabelBinarizer()
    mlb.fit(df_raw["labels"])
    num_labels = len(mlb.classes_)

    # 2. 1:1 Background Noise Downsampling
    df_has_labels = df_raw[df_raw["num_labels"] > 0]
    df_no_labels = df_raw[df_raw["num_labels"] == 0]
    df_no_labels_sampled = df_no_labels.sample(
        n=len(df_has_labels), random_state=args.seed
    )
    df = (
        pd.concat([df_has_labels, df_no_labels_sampled])
        .sample(frac=1, random_state=args.seed)
        .reset_index(drop=True)
    )
    labels_encoded = mlb.transform(df["labels"])
    print(
        f"[TRAM 2] Raw: {len(df_raw)} ({len(df_no_labels)/len(df_raw)*100:.1f}% noise) "
        f"-> Downsampled 1:1: {len(df)} samples across {num_labels} classes"
    )

    # 3. Iterative Stratified Split (80 / 10 / 10)
    X = df["sentence"].values.reshape(-1, 1)
    y = labels_encoded
    X_train, y_train, X_temp, y_temp = iterative_train_test_split(X, y, test_size=0.2)
    X_val, y_val, X_test, y_test = iterative_train_test_split(
        X_temp, y_temp, test_size=0.5
    )
    X_train, X_val, X_test = X_train.ravel(), X_val.ravel(), X_test.ravel()
    print(
        f"[TRAM 2] Iterative Split — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
    )

    # 4. Contextual Data Augmentation (CDA) on Training Split Only
    augmenter = ContextualDataAugmenter(
        model_name=args.model_name,
        aug_p=0.15,
        top_k=20,
        minority_threshold=args.minority_threshold,
        device=str(device),
    )
    print(
        f"[TRAM 2] Running Contextual Data Augmentation for minority classes (< {args.minority_threshold} samples)..."
    )
    X_train_aug, y_train_aug, num_synth = augmenter.augment_multilabel_minority_classes(
        X_train, y_train, seed=args.seed
    )
    print(
        f"[TRAM 2] Synthesized {num_synth} CDA samples -> Total Augmented Train: {len(X_train_aug)}"
    )

    # 5. Tokenization & DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader_baseline = DataLoader(
        CTIMultiLabelDataset(X_train, y_train, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    train_loader_opt = DataLoader(
        CTIMultiLabelDataset(X_train_aug, y_train_aug, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        CTIMultiLabelDataset(X_val, y_val, tokenizer, args.max_length),
        batch_size=64,
        shuffle=False,
    )
    test_loader = DataLoader(
        CTIMultiLabelDataset(X_test, y_test, tokenizer, args.max_length),
        batch_size=64,
        shuffle=False,
    )

    # 6. Baseline Training (Weighted BCE, Eq. 2, tau=0.50)
    print("\n[TRAM 2] Training Baseline Model (Weighted BCE, no CDA, tau=0.50)...")
    baseline_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    ).to(device)
    baseline_optimizer = AdamW(baseline_model.parameters(), lr=args.lr, eps=1e-8)
    pos_weight = compute_clipped_pos_weights(y_train, max_weight=50.0, device=device)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(args.epochs):
        baseline_model.train()
        total_loss = 0.0
        for batch in tqdm(
            train_loader_baseline, desc=f"[Baseline] Epoch {epoch+1}/{args.epochs}"
        ):
            baseline_optimizer.zero_grad()
            outputs = baseline_model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = bce_loss_fn(outputs.logits, batch["labels"].to(device))
            total_loss += loss.item()
            loss.backward()
            baseline_optimizer.step()

    base_micro, base_macro, base_emr, _, _ = evaluate_multilabel(
        baseline_model, test_loader, device, threshold=0.50
    )
    print(
        f"[Baseline Test] Micro-F1={base_micro:.4f} | Macro-F1={base_macro:.4f} | EMR={base_emr:.4f}"
    )
    del baseline_model, baseline_optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 7. Optimized Training (1:1 Downsample + CDA + Focal Loss, tau=0.35)
    print(
        f"\n[TRAM 2] Training Optimized Model (CDA + Focal Loss alpha=0.25, gamma=2.0, tau={args.eval_threshold})..."
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    best_val_macro = 0.0
    best_ckpt_path = args.output_dir / "secbert_tram2_best.pt"

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(
            train_loader_opt, desc=f"[Optimized] Epoch {epoch+1}/{args.epochs}"
        ):
            optimizer.zero_grad()
            outputs = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = focal_loss_fn(outputs.logits, batch["labels"].to(device))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        val_micro, val_macro, val_emr, _, _ = evaluate_multilabel(
            model, val_loader, device, threshold=args.eval_threshold
        )
        print(
            f"Epoch {epoch+1}: Val Micro-F1={val_micro:.4f} | Val Macro-F1={val_macro:.4f} | Val EMR={val_emr:.4f}"
        )
        if val_macro > best_val_macro:
            best_val_macro = val_macro
            torch.save(model.state_dict(), best_ckpt_path)

    if best_ckpt_path.exists():
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    test_micro, test_macro, test_emr, y_pred, y_true = evaluate_multilabel(
        model, test_loader, device, threshold=args.eval_threshold
    )
    print(
        f"\n=== [TRAM 2 Optimized] Test Set Results (tau={args.eval_threshold}) ===\n"
        f"  Micro-F1    : {test_micro:.4f} (Baseline: {base_micro:.4f})\n"
        f"  Macro-F1    : {test_macro:.4f} (Baseline: {base_macro:.4f})\n"
        f"  Exact Match : {test_emr:.4f} (Baseline: {base_emr:.4f})"
    )

    report_str = classification_report(
        y_true, y_pred, target_names=mlb.classes_, digits=4, zero_division=0
    )
    (args.output_dir / "tram2_classification_report.txt").write_text(
        f"Test F1-Micro    : {test_micro:.4f}\nTest F1-Macro    : {test_macro:.4f}\nTest Exact Match : {test_emr:.4f}\n\n{report_str}",
        encoding="utf-8",
    )

    # FP / FN Error Analysis Plot (English Labels)
    fp_counts = ((y_pred == 1) & (y_true == 0)).sum(axis=0)
    fn_counts = ((y_pred == 0) & (y_true == 1)).sum(axis=0)
    total_errors = fp_counts + fn_counts
    top15_idx = np.argsort(total_errors)[::-1][:15]

    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = np.arange(len(top15_idx))
    fn_vals = fn_counts[top15_idx]
    fp_vals = fp_counts[top15_idx]
    ax.barh(
        y_pos, fn_vals, color="tomato", edgecolor="black", label="Missed (False Negatives)"
    )
    ax.barh(
        y_pos,
        fp_vals,
        left=fn_vals,
        color="orange",
        edgecolor="black",
        label="False Alarms (False Positives)",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([mlb.classes_[i] for i in top15_idx])
    ax.invert_yaxis()
    ax.set_xlabel("Number of Errors")
    ax.set_title(
        "Top 15 Most Misclassified Techniques — TRAM 2\n(False Negatives vs. False Positives)"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        args.output_dir / "tram2_cell7_plot4_error_analysis.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    metadata = {
        "model_name": args.model_name,
        "task": "multi-label classification (TRAM 2)",
        "num_labels": num_labels,
        "max_length": args.max_length,
        "eval_threshold": args.eval_threshold,
        "train_samples_augmented": len(X_train_aug),
        "cda_synthetic_samples": num_synth,
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "baseline_test_f1_micro": round(base_micro, 4),
        "baseline_test_f1_macro": round(base_macro, 4),
        "baseline_test_exact_match": round(base_emr, 4),
        "optimized_test_f1_micro": round(test_micro, 4),
        "optimized_test_f1_macro": round(test_macro, 4),
        "optimized_test_exact_match": round(test_emr, 4),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[TRAM 2] Saved artifacts and reports to {args.output_dir}")


if __name__ == "__main__":
    main()
