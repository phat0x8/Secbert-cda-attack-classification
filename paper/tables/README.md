# Experimental Results Tables (Paper Tables 1–5)

## Table 1: Dataset Statistics Across the Three Evaluated CTI Benchmarks

| Property | TRAM 1 | TRAM 2 | rcATT |
| :--- | ---: | ---: | ---: |
| **Total samples (raw)** | 5,089 | 19,178 | 1,490 |
| **Samples with $\ge 1$ label** | 5,089 (4,837 deduped) | 4,070 | 1,490 |
| **Unique target labels ($K$)** | 50 | 50 | 227 (12 tactics + 215 techniques) |
| **Classification paradigm** | Single-label | Multi-label | Multi-label |
| **Avg. text length (words)** | 8.7 | 8.9 (raw) / 24.4 (labelled) | 2,829 (median 1,733; mean 6,538 tokens) |
| **Most frequent label** | `T1027` (685) | `T1027` (678) | `TA0005` (768) |
| **Class imbalance ratio** | 171× | 97× | >100× |
| **Background noise rate** | 0.0% | 78.8% | 0.0% |

---

## Table 2: Experimental Hyperparameter Configurations

| Hyperparameter | TRAM 1 | TRAM 2 (Optimized) | rcATT (Optimized) |
| :--- | :--- | :--- | :--- |
| **Backbone** | `nlpaueb/sec-bert-base` | `nlpaueb/sec-bert-base` | `nlpaueb/sec-bert-base` |
| **Max sequence length** | 128 | 256 | 256 (retains 3.9% of report tokens) |
| **Batch size / Epochs** | 32 / 5 | 32 / 5 | 32 / 5 |
| **Optimizer / Learning rate** | AdamW, $\eta = 2 \times 10^{-5}$ | AdamW, $\eta = 2 \times 10^{-5}$ | AdamW, $\eta = 2 \times 10^{-5}$ |
| **Loss function** | Class-Weighted CE (Eq. 1) | Multi-Label Focal Loss ($\alpha=0.25, \gamma=2.0$) | Multi-Label Focal Loss ($\alpha=0.25, \gamma=2.0$) |
| **Inference threshold ($\tau$)** | $\arg\max$ | 0.35 | 0.35 |
| **Data rebalancing / CDA** | Deduplication (-4.9%) | 1:1 Noise Downsampling + CDA ($p=0.15, k=20$, +469 samples for 24 rare classes) | Iterative Stratification |

---

## Table 3: TRAM 1 Per-Epoch Validation Metrics (Single-Label, 50 Classes)

| Epoch | Train Loss | Val Loss | Val Micro-F1 | Val Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 3.849 | 3.595 | 0.1715 | 0.1208 |
| **2** | 3.305 | 2.822 | 0.4773 | 0.4190 |
| **3** | 2.460 | 2.205 | 0.6095 | 0.5510 |
| **4** | 2.052 | 1.875 | 0.6674 | 0.6186 |
| **5** | **1.782** | **1.771** | **0.6901** | **0.6448** |

*Final Independent Test Split (484 samples):* **Micro-F1 / Accuracy = `0.6818`**, **Macro-F1 = `0.6160`**.

---

## Table 4: TRAM 2 Optimized Per-Epoch Validation Metrics (Multi-Label, 50 Classes, $\tau = 0.35$)

| Epoch | Train Loss | Val Micro-F1 | Val Macro-F1 | Val Exact Match Ratio (EMR) |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 0.0080 | 0.0479 | 0.0096 | 0.4988 |
| **2** | 0.0048 | 0.2757 | 0.0477 | 0.5086 |
| **3** | 0.0037 | 0.5046 | 0.2828 | 0.5676 |
| **4** | 0.0027 | 0.5637 | 0.3648 | **0.6229** |
| **5** | **0.0020** | **0.6105** | **0.5628** | 0.6130 |

*Final Independent Test Split (814 samples):* **Micro-F1 = `0.5847`**, **Macro-F1 = `0.4826`** (**+74.5% relative gain** over weighted BCE baseline `0.2767`), **EMR = `0.6130`**.

---

## Table 5: Cross-Dataset Empirical Results and Comparison with Prior Literature

| Method / System | Benchmark | Classes ($K$) | Micro-F1 | Macro-F1 | Exact Match Ratio (EMR) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Legoy et al. (2020) — TF-IDF + Linear | rcATT (tactics only) | 12 | — | 0.6200 | — |
| Orbinato et al. (2022) — SecBERT | Custom CTI sentences | 50 | 0.6500 | — | — |
| Li et al. (2024) — BiLSTM + Attention | Custom CTI reports | — | 0.7200 | 0.6800 | — |
| AC_MAPPER (Albarrak et al., 2025) | TRAM Bootstrap (curated subset) | 50 | — | 0.8975 | — |
| **Ours (Weighted CE)** | **TRAM 1 (Single-label)** | **50** | **0.6818** | **0.6160** | **0.6818** |
| **Ours (Baseline — Weighted BCE, $\tau=0.50$)** | **TRAM 2 (Multi-label)** | **50** | 0.3344 | 0.2767 | 0.3821 |
| **Ours (Optimized — 1:1 Downsample + CDA + Focal Loss, $\tau=0.35$, Best)** | **TRAM 2 (Multi-label)** | **50** | **0.5847** | **0.4826 (+74.5%)** | **0.6130** |
| **Ours (Optimized — 4-Run Mean $\pm$ Std)** | **TRAM 2 (Multi-label)** | **50** | $0.5706 \pm 0.0025$ | $0.4666 \pm 0.0016$ | $0.6057 \pm 0.0018$ |
| **Ours (256-Token Bottleneck Analysis)** | **rcATT (Document-level)** | **227** | 0.0905 | 0.0368 | 0.0000 |
