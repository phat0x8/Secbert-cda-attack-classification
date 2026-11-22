"""
rcATT Document-Level Multi-Label MITRE ATT&CK Classification & Truncation Analysis.

Evaluates `nlpaueb/sec-bert-base` (256-token fixed context window) on the document-level
rcATT benchmark (`dataset/training_data_original.zip`: 1,490 full CTI reports across
227 tactics and techniques; mean 2,829 words / 6,538 WordPiece tokens per report).

Quantifies the architectural context-window bottleneck reported in Paper Section 4.4:
- Average WordPiece token length : 6,538 tokens (median 3,980)
- Retained content at 256 tokens : 3.9% per report (99.5% of documents truncated)
- Resulting Test Performance     : Micro-F1 = 0.0905, Macro-F1 = 0.0368, EMR = 0.0000
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Dict, List
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from secbert_classifier import FocalLoss, evaluate_multilabel


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RcATTDataset(Dataset):
    """PyTorch Dataset for document-level rcATT reports."""

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


def load_rcatt_dataframe(data_path: Path) -> pd.DataFrame:
    """Load rcATT training_data_original.csv directly or from a ZIP archive."""
    if data_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(data_path, "r") as zf:
            csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
            with zf.open(csv_files[0]) as fh:
                return pd.read_csv(io.BytesIO(fh.read()))
    return pd.read_csv(data_path)


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate SecBERT on document-level rcATT (227 classes) and measure 256-token truncation."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=root_dir / "dataset" / "training_data_original.zip",
        help="Path to training_data_original.zip or training_data_original.csv.",
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
    parser.add_argument("--eval_threshold", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=root_dir / "rcATT_export",
        help="Directory to save evaluation reports and truncation metrics.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rcATT] Device: {device} | Backbone: {args.model_name}")

    # 1. Load Data & Extract 227 ATT&CK Columns
    df_raw = load_rcatt_dataframe(args.data_path)
    text_column = "Text"
    technique_columns = [
        c for c in df_raw.columns if c.startswith("T") and c != text_column
    ]
    df_raw["sentence"] = df_raw[text_column].fillna("").astype(str)

    def get_labels_for_row(row) -> List[str]:
        active = []
        for tech in technique_columns:
            val = row[tech]
            if val == 1 or val is True or (isinstance(val, str) and val.lower() == "true"):
                active.append(tech)
        return active

    df_raw["labels"] = df_raw.apply(get_labels_for_row, axis=1)
    mlb = MultiLabelBinarizer()
    labels_encoded = mlb.fit_transform(df_raw["labels"])
    num_labels = len(mlb.classes_)

    # 2. Measure WordPiece Token Truncation at max_length=256
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    token_counts = [
        len(tokenizer.encode(txt, truncation=False)) for txt in df_raw["sentence"]
    ]
    mean_tokens = float(np.mean(token_counts))
    retention_pct = args.max_length / mean_tokens * 100.0
    truncated_pct = float(np.mean([c > args.max_length for c in token_counts]) * 100.0)
    print(
        f"[rcATT] Reports: {len(df_raw)} | Labels: {num_labels}\n"
        f"        Mean WordPiece tokens: {mean_tokens:.0f} | Retained at {args.max_length} tokens: "
        f"{retention_pct:.2f}% | Truncated docs: {truncated_pct:.2f}%"
    )

    # 3. Iterative Stratified Split (80 / 10 / 10)
    X = df_raw["sentence"].values.reshape(-1, 1)
    y = labels_encoded
    X_train, y_train, X_temp, y_temp = iterative_train_test_split(X, y, test_size=0.2)
    X_val, y_val, X_test, y_test = iterative_train_test_split(
        X_temp, y_temp, test_size=0.5
    )
    X_train, X_val, X_test = X_train.ravel(), X_val.ravel(), X_test.ravel()

    train_loader = DataLoader(
        RcATTDataset(X_train, y_train, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        RcATTDataset(X_val, y_val, tokenizer, args.max_length),
        batch_size=64,
        shuffle=False,
    )
    test_loader = DataLoader(
        RcATTDataset(X_test, y_test, tokenizer, args.max_length),
        batch_size=64,
        shuffle=False,
    )

    # 4. Fine-Tune with Multi-Label Focal Loss (alpha=0.25, gamma=2.0)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    best_val_macro = 0.0
    best_ckpt = args.output_dir / "secbert_rcATT_best.pt"

    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(
            train_loader, desc=f"[rcATT] Epoch {epoch+1}/{args.epochs}"
        ):
            optimizer.zero_grad()
            outputs = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = focal_loss_fn(outputs.logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()

        val_micro, val_macro, val_emr, _, _ = evaluate_multilabel(
            model, val_loader, device, threshold=args.eval_threshold
        )
        print(
            f"Epoch {epoch+1}: Val Micro-F1={val_micro:.4f} | Val Macro-F1={val_macro:.4f}"
        )
        if val_macro > best_val_macro:
            best_val_macro = val_macro
            torch.save(model.state_dict(), best_ckpt)

    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    test_micro, test_macro, test_emr, y_pred, y_true = evaluate_multilabel(
        model, test_loader, device, threshold=args.eval_threshold
    )
    print(
        f"\n=== [rcATT] Test Set Results (tau={args.eval_threshold}) ===\n"
        f"  Micro-F1    : {test_micro:.4f}\n"
        f"  Macro-F1    : {test_macro:.4f}\n"
        f"  Exact Match : {test_emr:.4f}"
    )

    report_str = classification_report(
        y_true, y_pred, target_names=mlb.classes_, digits=4, zero_division=0
    )
    (args.output_dir / "rcATT_classification_report.txt").write_text(
        f"Test F1-Micro    : {test_micro:.4f}\nTest F1-Macro    : {test_macro:.4f}\nTest Exact Match : {test_emr:.4f}\n\n{report_str}",
        encoding="utf-8",
    )

    metadata = {
        "model_name": args.model_name,
        "task": "document-level multi-label classification (rcATT)",
        "num_labels": num_labels,
        "max_length": args.max_length,
        "mean_wordpiece_tokens": round(mean_tokens, 1),
        "avg_token_retention_pct": round(retention_pct, 2),
        "truncated_documents_pct": round(truncated_pct, 2),
        "optimized_test_f1_micro": round(test_micro, 4),
        "optimized_test_f1_macro": round(test_macro, 4),
        "optimized_test_exact_match": round(test_emr, 4),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"[rcATT] Saved evaluation artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
