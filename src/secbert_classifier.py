"""
SecBERT Classifier & Contextual Data Augmentation (CDA) Module for MITRE ATT&CK CTI.

Provides a PyTorch + Hugging Face Transformers architecture wrapping
`jackaduma/SecBERT` for single-label (TRAM1) and multi-label (TRAM2)
MITRE ATT&CK technique classification under severe class imbalance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class SecBERTConfig:
    """Configuration for SecBERT ATT&CK classification."""

    pretrained_model_name: str = "jackaduma/SecBERT"
    num_labels: int = 50
    max_seq_length: int = 256
    dropout_prob: float = 0.2
    multi_label: bool = False


class SecBERTAttackClassifier(nn.Module):
    """Domain-adapted SecBERT encoder with classification head for MITRE ATT&CK."""

    def __init__(self, config: SecBERTConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.pretrained_model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(config.dropout_prob)
        self.classifier = nn.Linear(hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_embedding))
        return logits


class ContextualDataAugmenter:
    """Masked Language Modeling (MLM) Contextual Data Augmentation for tail ATT&CK classes."""

    def __init__(
        self,
        model_name: str = "jackaduma/SecBERT",
        mask_ratio: float = 0.15,
    ) -> None:
        self.model_name = model_name
        self.mask_ratio = mask_ratio
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def compute_augmentation_quota(
        self, class_counts: Dict[str, int], target_min_samples: int = 50
    ) -> Dict[str, int]:
        """Return number of synthetic samples needed for each minority technique."""
        quotas: Dict[str, int] = {}
        for label, count in class_counts.items():
            if count < target_min_samples:
                quotas[label] = target_min_samples - count
        return quotas
