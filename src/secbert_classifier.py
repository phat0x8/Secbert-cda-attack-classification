"""
SecBERT Classifier, Focal Loss & Contextual Data Augmentation (CDA) Module
for MITRE ATT&CK Cyber Threat Intelligence (CTI) Classification.

Implements:
- `SecBERTConfig` & `SecBERTAttackClassifier`: Unified transformer classification
  wrapper supporting `nlpaueb/sec-bert-base` (benchmark protocol default),
  `jackaduma/SecBERT`, and `ehsanaghaei/SecureBERT`.
- `FocalLoss`: Multi-label Focal Loss (Lin et al., 2017; alpha=0.25, gamma=2.0)
  for mitigating extreme class imbalance and background noise in TRAM 2 & rcATT.
- `compute_balanced_class_weights` (Eq. 1) & `compute_clipped_pos_weights` (Eq. 2).
- `ContextualDataAugmenter`: Masked Language Modeling (MLM) contextual word
  substitution via `nlpaug` (`ContextualWordEmbsAug`, p=0.15, top_k=20) for
  minority ATT&CK techniques (< 50 training samples).
- `evaluate_multilabel`: Micro-F1, Macro-F1, and Exact Match Ratio (EMR) evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModel, AutoTokenizer


@dataclass
class SecBERTConfig:
    """Configuration for domain-adapted BERT ATT&CK classification."""

    pretrained_model_name: str = "nlpaueb/sec-bert-base"
    num_labels: int = 50
    max_seq_length: int = 256
    dropout_prob: float = 0.2
    multi_label: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    eval_threshold: float = 0.35


class FocalLoss(nn.Module):
    """
    Multi-Label Focal Loss (Lin et al., 2017) for sparse, long-tail CTI targets.

    Down-weights easy negative predictions and focuses gradient updates on
    hard minority-class ATT&CK techniques:
        L_focal = alpha * (1 - p_t)^gamma * BCEWithLogits(logits, targets)
    """

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduce: bool = True
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction="none"
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * bce_loss
        return torch.mean(focal_loss) if self.reduce else focal_loss


def compute_balanced_class_weights(
    y_train: np.ndarray, device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Compute inverse-frequency balanced class weights for single-label TRAM 1 (Eq. 1):
        w_c = N / (K * n_c)
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    tensor = torch.tensor(weights, dtype=torch.float)
    return tensor.to(device) if device is not None else tensor


def compute_clipped_pos_weights(
    y_train_multihot: np.ndarray,
    max_weight: float = 50.0,
    eps: float = 1e-5,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute per-class positive weights for multi-label BCE baseline (Eq. 2) to
    prevent all-zero model collapse on sparse CTI targets:
        pw_c = min(n_c^- / (n_c^+ + eps), max_weight)
    """
    pos_counts = y_train_multihot.sum(axis=0)
    neg_counts = len(y_train_multihot) - pos_counts
    clipped = np.clip(neg_counts / (pos_counts + eps), a_min=None, a_max=max_weight)
    tensor = torch.tensor(clipped, dtype=torch.float)
    return tensor.to(device) if device is not None else tensor


class SecBERTAttackClassifier(nn.Module):
    """Domain-adapted BERT encoder with classification head for MITRE ATT&CK."""

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
    """
    Masked Language Modeling (MLM) Contextual Data Augmentation (Kobayashi, 2018;
    Ma, 2019 `nlpaug`) for underrepresented MITRE ATT&CK technique classes.
    """

    def __init__(
        self,
        model_name: str = "nlpaueb/sec-bert-base",
        aug_p: float = 0.15,
        top_k: int = 20,
        minority_threshold: int = 50,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.aug_p = aug_p
        self.top_k = top_k
        self.minority_threshold = minority_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._augmenter = None

    def compute_augmentation_quota(
        self, class_counts: Dict[str, int], target_min_samples: Optional[int] = None
    ) -> Dict[str, int]:
        """Return number of synthetic samples needed for each minority technique."""
        threshold = target_min_samples or self.minority_threshold
        quotas: Dict[str, int] = {}
        for label, count in class_counts.items():
            if 0 < count < threshold:
                quotas[label] = threshold - count
        return quotas

    def _init_nlpaug(self):
        if self._augmenter is not None:
            return self._augmenter
        import nlpaug.augmenter.word as naw

        if not hasattr(
            transformers.PreTrainedTokenizerBase, "_convert_token_to_id"
        ):
            transformers.PreTrainedTokenizerBase._convert_token_to_id = (
                transformers.PreTrainedTokenizerBase.convert_tokens_to_ids
            )
        try:
            self._augmenter = naw.ContextualWordEmbsAug(
                model_path=self.model_name,
                model_type="bert",
                action="substitute",
                device=self.device,
                top_k=self.top_k,
                aug_p=self.aug_p,
            )
        except Exception:
            self._augmenter = naw.ContextualWordEmbsAug(
                model_path="bert-base-uncased",
                action="substitute",
                device=self.device,
                top_k=self.top_k,
                aug_p=self.aug_p,
            )
        return self._augmenter

    def augment_multilabel_minority_classes(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Synthesize contextually substituted CTI sentences exclusively for training
        classes with fewer than `self.minority_threshold` positive examples.
        """
        rng = np.random.default_rng(seed)
        aug = self._init_nlpaug()

        train_class_counts = np.sum(y_train, axis=0)
        minority_indices = np.where(
            (train_class_counts > 0)
            & (train_class_counts < self.minority_threshold)
        )[0]

        augmented_sentences: List[str] = []
        augmented_labels: List[np.ndarray] = []

        for class_idx in minority_indices:
            sample_indices = np.where(y_train[:, class_idx] == 1)[0]
            needed = self.minority_threshold - len(sample_indices)
            if needed <= 0:
                continue
            chosen = rng.choice(sample_indices, size=needed, replace=True)
            for idx in chosen:
                original_text = str(X_train[idx])
                original_label = y_train[idx]
                try:
                    aug_text = aug.augment(original_text)
                    if isinstance(aug_text, list):
                        aug_text = aug_text[0]
                    augmented_sentences.append(str(aug_text))
                    augmented_labels.append(original_label)
                except Exception:
                    continue

        if augmented_sentences:
            X_aug = np.concatenate((X_train, np.array(augmented_sentences)))
            y_aug = np.vstack((y_train, np.array(augmented_labels)))
            return X_aug, y_aug, len(augmented_sentences)
        return X_train, y_train, 0


def evaluate_multilabel(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.35,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluate a multi-label sequence classification model and return:
    (f1_micro, f1_macro, exact_match_ratio, all_preds, all_labels).
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(int)

            all_preds.append(preds)
            all_labels.append(labels)

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)

    f1_micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    exact_match = float(accuracy_score(y_true, y_pred))
    return f1_micro, f1_macro, exact_match, y_pred, y_true
