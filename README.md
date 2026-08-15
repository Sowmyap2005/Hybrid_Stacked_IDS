# Feature-Fused Intrusion Detection: A Stacking Framework With Ensemble Classifiers, Deep Learning, and Autoencoder Anomaly Signals

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIC--IDS2017-informational.svg)](docs/DATA.md)

This repository contains the reproducible implementation and supporting artifacts for a machine-learning-based intrusion detection study using the **CIC-IDS2017** network traffic dataset.

The primary system combines four complementary models—**Random Forest, XGBoost, a deep neural network (DNN), and a benign-traffic autoencoder**—through a logistic-regression stacking meta-classifier. The repository also contains separate experiments for PCA ablation, component ablation, reconstruction-loss selection, adaptive threshold analysis, SHAP-based interpretability, inference benchmarking, and repeated known/unseen attack-class evaluation.

> **Research/reproducibility note:** The repository is intended to accompany a scientific publication. Reported values should be reproduced from the supplied scripts and artifacts rather than inferred from screenshots or figures. The repository deliberately does **not** redistribute the full CIC-IDS2017 CSV files or large intermediate training matrices; see [Data availability](#data-availability).

---

## 1. Research objective

The study investigates whether a hybrid supervised/unsupervised architecture can improve network intrusion classification while providing an additional reconstruction-error signal for traffic belonging to attack classes not represented during autoencoder training.

The main multiclass pipeline is:

```text
CIC-IDS2017
    │
    ├── Cleaning / sampling / rare-class filtering
    ├── Train/test split
    ├── Standardization
    └── SMOTE on training data
            │
            ├───────────────┬────────────────┬─────────────────┐
            ▼               ▼                ▼                 ▼
       Random Forest     XGBoost           DNN          Autoencoder
       probabilities     probabilities     probabilities   reconstruction error
            │               │                │                 │
            └───────────────┴────────────────┴─────────────────┘
                                    │
                                    ▼
                         Logistic Regression
                          stacking meta-model
                                    │
                                    ▼
                           Final class prediction
```

The autoencoder is trained on **BENIGN traffic only** in the primary stacked architecture. Its reconstruction error is used as an additional meta-level feature. Separate experiments evaluate whether reconstruction error can distinguish held-out/unseen attack traffic from benign traffic.

---

## 2. Dataset

The experiments use the **CIC-IDS2017** dataset published by the Canadian Institute for Cybersecurity.

The original dataset contains labeled network-flow features covering benign traffic and multiple attack families. The experiments in this repository use the eight CSV files required by the supplied preprocessing workflow.

### Classes

The final multiclass experiment contains:

- `BENIGN`
- `Bot`
- `DDoS`
- `DoS GoldenEye`
- `DoS Hulk`
- `DoS Slowhttptest`
- `DoS slowloris`
- `FTP-Patator`
- `PortScan`
- `SSH-Patator`
- `Web Attack - Brute Force`
- `Web Attack - XSS`

### Data preprocessing used by the primary pipeline

1. Each source CSV is sampled at 20% using `random_state=42`.
2. Column names are stripped of leading/trailing whitespace.
3. Infinite values are replaced with missing values.
4. Rows containing missing values are removed.
5. `Flow ID`, `Source IP`, `Destination IP`, and `Timestamp` are removed.
6. Classes with fewer than 10 observations are removed.
7. An 80/20 stratified train/test split is created.
8. The training partition is reduced to 20% using stratified sampling to control memory usage.
9. `StandardScaler` is fitted on the training data and then applied to the test data.
10. SMOTE (`k_neighbors=3`, `random_state=42`) is applied **only to the training partition**.
11. PCA is not applied in the final pipeline. The supplied PCA ablation shows that retaining the original feature space performed better for the evaluated classifiers.

The exact preprocessing implementation is in [`src/01_preprocess.py`](src/01_preprocess.py).

---

## 3. Primary model

### Base learners

| Component       | Configuration                                                          |
| --------------- | ---------------------------------------------------------------------- |
| Random Forest   | 50 trees, `max_depth=12`                                               |
| XGBoost         | 50 estimators, `max_depth=6`, learning rate 0.1, histogram tree method |
| DNN             | Dense(128) → Dropout(0.3) → Dense(64) → Dropout(0.3) → Softmax         |
| Autoencoder     | 80% input → 16 → 8 → 16 → original feature dimension                   |
| Meta-classifier | Logistic Regression, `class_weight="balanced"`                         |

The DNN and autoencoder are trained for 10 epochs with batch size 512 in the supplied training workflow.

The stacking procedure first splits the processed training set into a 70/30 base/meta partition. The base learners are fitted on the base partition. Their predictions on the held-out meta partition, together with autoencoder reconstruction error, are used to train the logistic-regression meta-classifier.

See [`src/02_train_stack.py`](src/02_train_stack.py).

---

## 4. Reported evaluation artifacts

The repository includes the evaluation outputs generated from the supplied experiment.

### Final stacked model

The included `results/metrics/stacked_model_auc_metrics.csv` reports one-vs-rest ROC-AUC and average precision (PR-AUC) for each of the 12 classes.

From that supplied table:

- **Macro ROC-AUC:** 0.9981
- **Support-weighted ROC-AUC:** 0.9996
- **Macro PR-AUC:** 0.9173
- **Support-weighted PR-AUC:** 0.9993

These are descriptive summaries calculated from the per-class values stored in the repository. The complete per-class results, including support, should be used when reporting results in the manuscript.

> **Important:** Very high aggregate scores are strongly influenced by the dataset's class distribution and the experimental protocol. They should not be interpreted as evidence of equivalent performance on operational network traffic.

### Unseen-class experiment

A separate repeated known/unseen experiment evaluates the autoencoder reconstruction-error signal under five randomly generated splits, with six attack classes treated as known and five attack classes held out in each round.

The supplied experiment log reports:

- Mean ROC-AUC: **0.7861**
- Standard deviation: **0.0314**
- Mean PR-AUC: **0.7436**
- Standard deviation: **0.0947**
- ROC-AUC range: **0.7281–0.8188**

These values are **not** the same as the final supervised multiclass ROC-AUC values above. They measure a separate unseen-attack detection experiment.

See [`src/13_unseen_split_rotation.py`](src/13_unseen_split_rotation.py) and [`results/experiments/multi_split_rotation_log.txt`](results/experiments/multi_split_rotation_log.txt).

---

## 5. Repository structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── .gitignore
├── .gitattributes
├── requirements.txt
├── environment.yml
│
├── src/
│   ├── 01_preprocess.py
│   ├── 02_train_stack.py
│   ├── 03_evaluate_auc.py
│   ├── 04_normalized_confusion.py
│   ├── 05_plot_roc_curves.py
│   ├── 06_plot_component_ablation.py
│   ├── 07_plot_pca_ablation.py
│   ├── 08_plot_latency.py
│   ├── 09_benchmark_latency.py
│   ├── 10_shap_analysis.py
│   ├── 11_adaptive_threshold_analysis.py
│   ├── 12_compare_ae_loss.py
│   ├── 13_unseen_split_rotation.py
│   └── 14_pca_ablation_analysis.py
│
├── models/
│   ├── random_forest_model_nopca.pkl
│   ├── xgb_model_nopca.pkl
│   ├── dnn_model_nopca.keras
│   ├── autoencoder_model_nopca.keras
│   ├── meta_model_nopca.pkl
│   ├── scaler_nopca.pkl
│   ├── label_encoder_nopca.pkl
│   └── feature_names_nopca.pkl
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── experiments/
│
├── docs/
│   └── DATA.md
│
└── data/
    └── README.md
```

Large raw datasets and large intermediate arrays are intentionally excluded from version control.

---

## 6. Installation

Python **3.11** is recommended for the supplied environment.

### Option A — pip

```bash
git clone <REPOSITORY-URL>
cd <REPOSITORY-DIRECTORY>

python3.11 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate hybrid-ids
```

---

## 7. Obtaining the dataset

Download CIC-IDS2017 from the official Canadian Institute for Cybersecurity dataset page and place the required CSV files under:

```text
data/TrafficLabelling/
```

The repository expects the following files:

```text
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Monday-WorkingHours.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
```

See [`docs/DATA.md`](docs/DATA.md) for the data-availability statement and dataset handling policy.

If the dataset is stored elsewhere, set:

```bash
export CICIDS2017_DIR="/path/to/TrafficLabelling"
```

On Windows PowerShell:

```powershell
$env:CICIDS2017_DIR="C:\path\to\TrafficLabelling"
```

---

## 8. Reproducing the primary experiment

From the repository root:

```bash
python src/01_preprocess.py
python src/02_train_stack.py
python src/03_evaluate_auc.py
python src/04_normalized_confusion.py
python src/05_plot_roc_curves.py
```

The preprocessing stage produces the intermediate files required by the training stage. These generated files are intentionally ignored by Git because they can be hundreds of megabytes.

---

## 9. Additional analyses

### PCA ablation

```bash
python src/14_pca_ablation_analysis.py
python src/07_plot_pca_ablation.py
```

### Component ablation

```bash
python src/06_plot_component_ablation.py
```

### Autoencoder loss comparison

```bash
python src/12_compare_ae_loss.py
```

This compares MSE and MAE reconstruction losses under the same architecture and thresholding procedure.

### Adaptive threshold analysis

```bash
python src/11_adaptive_threshold_analysis.py
```

This evaluates the separation achievable by reconstruction-error thresholding in a controlled held-out-class experiment.

### Repeated known/unseen attack splits

```bash
python src/13_unseen_split_rotation.py
```

The script performs five random six-known/five-unseen attack-class rotations and reports mean and standard deviation of ROC-AUC and PR-AUC.

### SHAP interpretability

```bash
python src/10_shap_analysis.py
```

The supplied analysis uses XGBoost and a 2,000-row test subset to obtain global feature-importance estimates.

### Inference benchmark

```bash
python src/09_benchmark_latency.py
```

The benchmark measures the complete stacked inference pipeline at multiple batch sizes.

---

## 10. Reproducibility and random seeds

Where applicable, the supplied scripts use `random_state=42` for the main preprocessing and model-development workflow. The repeated unseen-class experiment intentionally uses different deterministic seeds (`100 + round_index`) to generate distinct class partitions.

Results can still vary across:

- operating systems,
- Python/package versions,
- CPU/GPU implementations,
- TensorFlow/XGBoost numerical behavior,
- BLAS libraries,
- hardware,
- and library-level changes.

For publication-quality reproduction, record the exact Python version, package versions, hardware, operating system, and commit hash used for the final experiment.

---

## 11. Data leakage considerations

The repository follows several safeguards intended to prevent direct train/test leakage:

- The main train/test split is performed before training transformations.
- `StandardScaler` is fitted only on the training partition.
- SMOTE is applied only to the training partition.
- The meta-classifier receives predictions generated from a held-out meta partition rather than the same samples used to fit the base learners.
- The autoencoder's primary reconstruction-error model is trained using benign samples from the base training partition.
- Unseen-class experiments explicitly remove the designated unseen attack classes from the autoencoder training data.

Nevertheless, the CIC-IDS2017 dataset is a controlled benchmark rather than a deployment-scale longitudinal network trace. Dataset-specific correlations and collection artifacts can affect measured performance.

---

## 12. Interpretability

The repository provides SHAP-based global feature importance for the XGBoost component.

The supplied analysis identifies, among the highest-ranked features:

- Destination Port
- Init_Win_bytes_backward
- Init_Win_bytes_forward
- Source Port
- min_seg_size_forward
- Bwd Packet Length Min
- Flow IAT Mean
- Fwd Packet Length Max

The complete table is available at:

`results/metrics/shap_feature_importance_nopca.csv`

Interpretation should remain model- and dataset-specific; feature importance does not by itself establish causal relationships.

---

## 13. Pretrained artifacts

Small trained model artifacts are included under `models/` for inspection and reuse. To reproduce the complete evaluation from these artifacts, the required processed test data must first be generated locally from the external dataset.

The repository does **not** treat these files as a substitute for full reproducibility. The preprocessing code, training code, dataset reference, and experiment logs are provided so that the models can be regenerated.

Before publication, it is recommended to record a checksum for the final model artifacts and tag the exact repository commit used for the manuscript.

---

## 14. Data availability

The full CIC-IDS2017 dataset is not redistributed in this repository.

This repository contains the code required to process the dataset, together with derived results and selected trained artifacts. Users must obtain the original dataset from its official source and comply with its applicable terms.

This separation is intentional to avoid redistributing a third-party dataset and to keep the Git repository within practical hosting limits.

See [`docs/DATA.md`](docs/DATA.md).

---

## 15. License

The original source code in this repository is released under the **MIT License**.

The license applies to the repository's original code and documentation. It does **not** grant additional rights to third-party datasets, trademarks, or other external materials.

See [`LICENSE`](LICENSE).

---

## 16. Citation

If you use this repository or its implementation in academic work, please cite the software using [`CITATION.cff`](CITATION.cff).

The CITATION.cff file contains the current repository and author metadata. Publication DOI information will be added when the associated manuscript is formally published.

---

## 17. Research status and limitations

This repository is a research artifact, not a production intrusion-prevention system.

Important limitations include:

1. CIC-IDS2017 is a benchmark dataset and may not represent contemporary operational network traffic.
2. The primary multiclass classifier evaluates attack classes represented in the labeled benchmark.
3. Unseen-attack detection is evaluated separately using reconstruction-error experiments.
4. Threshold selection has a substantial effect on anomaly detection results.
5. Class imbalance remains an important consideration even after SMOTE.
6. The reported aggregate metrics should be interpreted alongside per-class results.
7. The supplied latency benchmark is an experimental CPU inference measurement and is not a deployment SLA.
8. Model artifacts may be sensitive to package versions and hardware.

The repository is therefore best understood as a reproducible research implementation supporting the associated manuscript.

---

## 18. Contact

For questions concerning the scientific implementation, please use the contact information provided in the associated publication or repository metadata.

For reproducibility issues, please open a GitHub issue with:

- operating system,
- Python version,
- package versions,
- hardware,
- exact command executed,
- relevant error output,
- and repository commit hash.
