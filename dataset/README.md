# Cyber Threat Intelligence (CTI) Benchmark Datasets

This directory houses the three public CTI benchmarks evaluated under our unified MITRE ATT&CK classification protocol:

| File | Benchmark | Granularity | Paradigm | Raw Samples | Labelled Samples | Target Classes ($K$) | Imbalance Ratio | Background Noise |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `single_label.json` | **TRAM 1** ([MITRE CTID](https://github.com/center-for-threat-informed-defense/tram)) | Sentence (mean 8.7 words) | Single-Label | 5,089 (4,837 deduped) | 5,089 | 50 techniques | 171.2× (`T1027`: 685 vs `T1557.001`: 4) | 0.0% |
| `multi_label.json` | **TRAM 2** ([MITRE CTID](https://github.com/center-for-threat-informed-defense/tram)) | Sentence (mean 24.4 words labelled) | Multi-Label | 19,178 | 4,070 | 50 techniques | 96.9× (`T1027`: 678 vs `T1557.001`: 7) | **78.8%** (15,108 unlabelled) |
| `training_data_original.zip` | **rcATT** ([Legoy et al., 2020](https://github.com/vlegoy/rcATT)) | Full CTI Report (mean 2,829 words / 6,538 WordPiece tokens) | Multi-Label | 1,490 | 1,490 | 227 (12 tactics + 215 techniques) | >100× (`TA0005`: 768) | 0.0% |

## Inspecting Dataset Statistics

Run the built-in multi-benchmark inspector from the repository root:

```bash
python src/dataset_analysis.py --benchmark all
```
