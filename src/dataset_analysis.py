"""
Dataset Analysis & Long-Tail Imbalance Inspector for MITRE ATT&CK CTI Reports.

Analyzes technique distribution, head/mid/tail frequency buckets, and sentence
length statistics on the TRAM1 single-label CTI dataset (dataset/single_label.json).
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List


def load_cti_records(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load CTI sentence-level ATT&CK records from a JSON file."""
    with dataset_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {dataset_path}")
    return data


def compute_imbalance_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute class distribution, imbalance ratio, and head/tail buckets."""
    labels = [ str(r.get("label", "")).strip() for r in records if r.get("label") ]
    counts = Counter(labels)
    if not counts:
        return {"total_samples": 0, "num_classes": 0}

    sorted_counts = counts.most_common()
    max_cls, max_cnt = sorted_counts[0]
    min_cls, min_cnt = sorted_counts[-1]

    head = [k for k, v in sorted_counts if v >= 100]
    mid = [k for k, v in sorted_counts if 20 <= v < 100]
    tail = [k for k, v in sorted_counts if v < 20]

    return {
        "total_samples": len(labels),
        "num_classes": len(counts),
        "most_frequent": {"technique": max_cls, "count": max_cnt},
        "least_frequent": {"technique": min_cls, "count": min_cnt},
        "imbalance_ratio": round(max_cnt / max(min_cnt, 1), 2),
        "buckets": {
            "head_ge_100": len(head),
            "mid_20_to_99": len(mid),
            "tail_lt_20": len(tail),
        },
        "top_10_techniques": sorted_counts[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze MITRE ATT&CK class imbalance in CTI datasets."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset" / "single_label.json",
        help="Path to single_label.json dataset file.",
    )
    args = parser.parse_args()

    records = load_cti_records(args.dataset)
    summary = compute_imbalance_metrics(records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
