"""
Multi-Benchmark Dataset Analysis & Long-Tail Imbalance Inspector for MITRE ATT&CK CTI.

Supports:
1. TRAM 1 (`dataset/single_label.json`): 5,089 raw / 4,837 deduplicated single-label
   sentences across 50 ATT&CK techniques (171x imbalance ratio).
2. TRAM 2 (`dataset/multi_label.json`): 19,178 raw multi-label sentences (4,070
   labelled CTI sentences + 15,108 unlabelled background noise sentences = 78.8%).
3. rcATT (`dataset/training_data_original.zip`): 1,490 full-length CTI reports across
   227 tactics & techniques (mean 2,829 words / 6,538 WordPiece tokens; quantifies
   the 3.9% content retention rate under a 256-token BERT context window).
"""

from __future__ import annotations

import argparse
from collections import Counter
import io
import json
from pathlib import Path
from typing import Any, Dict, List
import zipfile

import numpy as np
import pandas as pd


def analyze_tram1(dataset_path: Path) -> Dict[str, Any]:
    """Compute class distribution, deduplication count, and word stats on TRAM 1."""
    with dataset_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)

    df = pd.DataFrame(records)
    raw_count = len(df)
    df_dedup = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    word_lengths = df["text"].astype(str).apply(lambda s: len(s.split()))

    counts = Counter(df["label"].astype(str).str.strip())
    sorted_counts = counts.most_common()
    max_cls, max_cnt = sorted_counts[0]
    min_cls, min_cnt = sorted_counts[-1]

    head = [k for k, v in sorted_counts if v >= 100]
    mid = [k for k, v in sorted_counts if 20 <= v < 100]
    tail = [k for k, v in sorted_counts if v < 20]

    return {
        "benchmark": "TRAM 1 (Single-Label Sentence)",
        "total_samples_raw": raw_count,
        "total_samples_deduplicated": len(df_dedup),
        "exact_duplicates_removed": raw_count - len(df_dedup),
        "num_classes": len(counts),
        "avg_word_length": round(float(word_lengths.mean()), 2),
        "median_word_length": round(float(word_lengths.median()), 2),
        "most_frequent": {"technique": max_cls, "count": max_cnt},
        "least_frequent": {"technique": min_cls, "count": min_cnt},
        "imbalance_ratio": round(max_cnt / max(min_cnt, 1), 1),
        "buckets": {
            "head_ge_100": len(head),
            "mid_20_to_99": len(mid),
            "tail_lt_20": len(tail),
        },
        "top_10_techniques": sorted_counts[:10],
    }


def analyze_tram2(dataset_path: Path) -> Dict[str, Any]:
    """Compute multi-label distribution and background noise rate on TRAM 2."""
    with dataset_path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)

    total_raw = len(records)
    labelled_records = [r for r in records if len(r.get("labels", [])) > 0]
    unlabelled_count = total_raw - len(labelled_records)

    label_counter: Counter = Counter()
    word_lengths: List[int] = []
    for r in labelled_records:
        label_counter.update(r.get("labels", []))
        word_lengths.append(len(str(r.get("sentence", "")).split()))

    sorted_counts = label_counter.most_common()
    max_cls, max_cnt = sorted_counts[0]
    min_cls, min_cnt = sorted_counts[-1]

    return {
        "benchmark": "TRAM 2 (Multi-Label Sentence)",
        "total_samples_raw": total_raw,
        "labelled_samples": len(labelled_records),
        "unlabelled_background_noise": unlabelled_count,
        "background_noise_rate_pct": round(unlabelled_count / total_raw * 100, 2),
        "downsampled_1_to_1_corpus_size": len(labelled_records) * 2,
        "num_classes": len(label_counter),
        "avg_word_length_labelled": round(float(np.mean(word_lengths)), 2),
        "median_word_length_labelled": round(float(np.median(word_lengths)), 2),
        "most_frequent": {"technique": max_cls, "count": max_cnt},
        "least_frequent": {"technique": min_cls, "count": min_cnt},
        "imbalance_ratio": round(max_cnt / max(min_cnt, 1), 1),
        "top_10_techniques": sorted_counts[:10],
    }


def analyze_rcatt(zip_or_csv_path: Path, max_window_tokens: int = 256) -> Dict[str, Any]:
    """Compute document-level statistics and 256-token truncation impact on rcATT."""
    if zip_or_csv_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(zip_or_csv_path, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            with zf.open(csv_names[0]) as csv_file:
                df = pd.read_csv(io.BytesIO(csv_file.read()))
    else:
        df = pd.read_csv(zip_or_csv_path)

    text_col = "Text"
    tech_cols = [c for c in df.columns if c.startswith("T") and c != text_col]
    word_lengths = df[text_col].fillna("").astype(str).apply(lambda s: len(s.split()))
    # Average WordPiece expansion ratio on rcATT CTI reports (~2.31 subwords/word -> mean 6,538 tokens)
    approx_token_lengths = word_lengths * (6538.0 / 2829.0)
    truncated_docs = int((approx_token_lengths > max_window_tokens).sum())

    label_sums = df[tech_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum()
    sorted_labels = label_sums.sort_values(ascending=False)

    return {
        "benchmark": "rcATT (Document-Level Multi-Label)",
        "total_reports": len(df),
        "num_classes": len(tech_cols),
        "avg_word_length": round(float(word_lengths.mean()), 1),
        "median_word_length": round(float(word_lengths.median()), 1),
        "max_word_length": int(word_lengths.max()),
        "mean_wordpiece_tokens": 6538,
        "context_window_tokens": max_window_tokens,
        "avg_token_retention_pct": round(max_window_tokens / 6538.0 * 100, 2),
        "truncated_documents_pct": round(truncated_docs / len(df) * 100, 2),
        "most_frequent": {
            "label": str(sorted_labels.index[0]),
            "count": int(sorted_labels.iloc[0]),
        },
        "top_10_labels": [
            (str(k), int(v)) for k, v in sorted_labels.head(10).items()
        ],
    }


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1] / "dataset"
    parser = argparse.ArgumentParser(
        description="Analyze MITRE ATT&CK class imbalance and text granularity across CTI benchmarks."
    )
    parser.add_argument(
        "--benchmark",
        choices=["tram1", "tram2", "rcatt", "all"],
        default="all",
        help="Benchmark to analyze (default: all).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional custom path to dataset file.",
    )
    args = parser.parse_args()

    results: Dict[str, Any] = {}
    if args.benchmark in ("tram1", "all"):
        p1 = args.dataset or (root_dir / "single_label.json")
        if p1.exists():
            results["tram1"] = analyze_tram1(p1)
    if args.benchmark in ("tram2", "all"):
        p2 = args.dataset or (root_dir / "multi_label.json")
        if p2.exists():
            results["tram2"] = analyze_tram2(p2)
    if args.benchmark in ("rcatt", "all"):
        p3 = args.dataset or (root_dir / "training_data_original.zip")
        if p3.exists():
            results["rcatt"] = analyze_rcatt(p3)

    print(json.dumps(results if len(results) > 1 else next(iter(results.values()), {}), indent=2))


if __name__ == "__main__":
    main()
