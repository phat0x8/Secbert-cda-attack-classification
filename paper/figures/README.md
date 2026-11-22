# Experimental Figures Gallery

All figures generated across the three CTI benchmarks (**TRAM 1**, **TRAM 2**, and **rcATT**):

## 1. TRAM 1 (Single-Label Sentence Benchmark — 50 ATT&CK Techniques)
- `tram1/tram1_cell2_eda.png` — Top-20 ATT&CK technique distribution & sentence length histogram (mean 8.7 words, 171× imbalance).
- `tram1/tram1_cell3_plot1_dedup.png` — Sample count before/after exact-duplicate removal (5,089 $\to$ 4,837).
- `tram1/tram1_cell3_plot2_split.png` — Stratified 80/10/10 Train/Val/Test split (3,869 / 484 / 484).
- `tram1/tram1_cell3_plot3_classweights.png` — Inverse-frequency balanced class weights across 50 techniques.
- `tram1/tram1_cell6_plot1_metrics.png` — Test set overall metrics (Micro-F1 = `0.6818`, Macro-F1 = `0.6160`).
- `tram1/tram1_cell6_plot2_per_class_f1.png` — Per-technique F1 score distribution sorted descending.
- `tram1/tram1_cell6_plot3_training_curve.png` — 5-epoch training & validation loss vs. validation Macro-F1 progression.
- `tram1/tram1_cell7_plot1_confusion_matrix.png` — Full $50 \times 50$ confusion matrix on the TRAM 1 test split.
- `tram1/tram1_cell7_plot2_top_errors.png` — Top-15 misclassification pairs dominated by `T1027 -> T1140` (23×) and `T1027 -> T1573.001` (18×).

## 2. TRAM 2 (Multi-Label Sentence Benchmark — 50 ATT&CK Techniques, 78.8% Background Noise)
- `tram2/tram2_tram2_cell2_eda_label_distribution.png` — Top-40 technique frequency distribution among labelled CTI sentences.
- `tram2/tram2_cell3_plot1_downsampling.png` — Corpus size before/after 1:1 background noise downsampling (19,178 $\to$ 8,140).
- `tram2/tram2_cell3_plot2_split.png` — Iterative Stratification 80/10/10 split (6,513 / 814 / 814; augmented train = 6,982).
- `tram2/tram2_cell7_plot1_baseline_vs_optimized.png` — Test set comparison: Weighted BCE Baseline (`0.3344` Micro-F1 / `0.2767` Macro-F1) vs. Optimized Downsample 1:1 + CDA + Focal Loss (`0.5847` Micro-F1 / `0.4826` Macro-F1, **+74.5% relative gain**).
- `tram2/tram2_cell7_plot2_per_class_f1.png` — Per-class F1 score distribution on TRAM 2 test set.
- `tram2/tram2_cell7_plot3_training_curve.png` — Training loss and validation Macro-F1 trajectory over 5 epochs.
- `tram2/tram2_cell7_plot4_error_analysis.png` — Top-15 misclassified techniques broken down into False Negatives (missed detections) and False Positives (false alarms, led by `T1027` with 48 FP / 23 FN).
- `tram2/fig_grid_bvo.png` & `tram2/fig_grid_error.png` — Multi-run stability verification across 4 independent trials ($0.5706 \pm 0.0025$ Micro-F1).

## 3. rcATT (Document-Level Multi-Label Benchmark — 227 Tactics & Techniques)
- `rcATT/rcATT_cell2_eda_label_distribution.png` — Top-40 label distribution across 1,490 full CTI reports.
- `rcATT/rcATT_cell3_plot2_split.png` — Iterative Stratification Train/Val/Test split.
- `rcATT/rcATT_cell7_plot1_baseline_vs_optimized.png` — Baseline vs. Focal Loss comparison under the 256-token truncation constraint.
- `rcATT/rcATT_cell7_plot2_per_class_f1.png` — Per-class F1 distribution for classes with $F1 > 0$.
- `rcATT/rcATT_cell7_plot3_training_curve.png` — Training loss vs. oscillating validation Macro-F1 (< 0.05) due to 96.1% token truncation.
- `rcATT/rcATT_cell7_plot4_error_analysis.png` — Top-15 error-prone labels under truncated 256-token representations.
