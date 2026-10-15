# AI-Driven Drug Discovery: hERG Classification and pIC50 Regression

## Overview

This project explores machine learning and deep learning approaches for AI-driven drug discovery through two molecular prediction tasks:

1. **hERG classification:** Predicting hERG-related activity or liability from molecular structures.
2. **pIC50 regression:** Predicting continuous pIC50 values to estimate the biological potency of compounds.

The project compares traditional machine learning, transformer-based molecular representations, graph neural networks, and ensemble learning. The objective is to evaluate how different molecular representations and model-combination strategies perform on classification and regression tasks.

## Objectives

- Develop predictive models for hERG classification and pIC50 regression.
- Compare molecular fingerprints, pretrained SMILES representations, and molecular graph representations.
- Evaluate models using training, validation, and held-out test sets.
- Investigate ensemble learning to combine predictions from different models.
- Use validation-based model selection to reduce information leakage into test evaluation.
- Build a reproducible workflow for molecular data preparation, model training, and evaluation.

## Models and Approaches

### Classification

| Model | Molecular representation / approach |
|---|---|
| XGBoost | Morgan molecular fingerprints |
| ChemBERTa | Transformer-based molecular representations from SMILES |
| Chemprop | Graph-based molecular learning |
| Equal-weight ensemble | Combines predictions using equal model weights |
| Validation-selected weighted ensemble | Combines predictions using validation-selected model weights |

The classification workflow evaluates individual models and ensemble strategies using metrics suitable for potentially imbalanced datasets.

The decision threshold is selected using validation data. The threshold-selection strategy aims to maximize the Matthews correlation coefficient (MCC), subject to a minimum sensitivity requirement of 90%, where applicable.

### Regression

| Model | Molecular representation / approach |
|---|---|
| XGBoost Regression | Gradient-boosted trees using molecular features |
| ChemBERTa Regression | Transformer-based molecular representations |
| Chemprop Regression | Graph-based molecular learning |
| Equal-weight ensemble | Combines individual model predictions with equal weights |
| Weighted ensemble | Combines individual model predictions using validation-selected weights |

For regression, the ensemble weights are selected using validation data and fixed before evaluation on the held-out test set.

## Repository Structure

```text
AIDD-Challenge-4-Project/
├── README.md
├── .gitignore
├── requirements.txt
│
├── classification/
│   ├── trees.py
│   ├── chemberta.py
│   ├── chemprop.py
│   └── ensemble.py
│
├── regression/
│   ├── xgboost_regression.py
│   ├── chemberta_regression.py
│   ├── chemprop_regression.py
│   └── regression_ensemble.py
│
├── scripts/
│   ├── scaffold_split.py
│   ├── create_regression_dataset.py
│   └── split_regression_dataset.py
│
├── data/
│   ├── raw/
│   │   ├── BindingDB_hERG_labeled_dataset.csv
│   │   └── herg_final_ML_dataset.csv
│   ├── processed/
│   │   └── herg_regression_dataset.csv
│   └── splits/
│
├── notebooks/
│   ├── classification/
│   └── regression/
│
├── results/
└── docs/
```

The tree represents the intended organization of the project. Empty directories may not appear on GitHub unless they contain tracked placeholder files.

## Data Preparation

The `data/` directory separates raw datasets from processed data and data splits.

### Raw data

The repository currently contains the following raw dataset files:

- `herg_final_ML_dataset.csv`
- `BindingDB_hERG_labeled_dataset.csv`

### Processed data

The regression dataset preparation workflow produces:

- `herg_regression_dataset.csv`

### Data splitting

The `scripts/` directory contains utilities for:

- Creating scaffold-based data splits.
- Preparing the regression dataset.
- Generating regression train, validation, and test splits.

Scaffold-based splitting helps evaluate generalization to molecular structures with different scaffolds rather than relying only on random partitioning.

The regression splitting workflow is intended to maintain consistency with the corresponding scaffold-based partitioning where applicable.

Dataset source attribution, target definitions, preprocessing steps, and redistribution permissions should be documented before the datasets are shared publicly.

## Methodology

### Classification evaluation

The classification models are evaluated using:

- **Accuracy:** Fraction of all predictions that are correct.
- **Precision:** Fraction of predicted positives that are true positives.
- **Sensitivity (recall):** Fraction of actual positives correctly identified.
- **Specificity:** Fraction of actual negatives correctly identified.
- **F1-score:** Harmonic mean of precision and sensitivity.
- **ROC-AUC:** Ability to distinguish positive and negative examples across thresholds.
- **PR-AUC:** Precision-recall performance across thresholds.
- **MCC:** A balanced measure of binary classification quality that incorporates all four confusion-matrix outcomes.

Threshold selection is performed using validation data, while the held-out test set is reserved for final evaluation.

### Regression evaluation

The regression models are evaluated using:

- **RMSE:** Root mean squared error; penalizes larger prediction errors more strongly.
- **MAE:** Mean absolute error; measures average absolute prediction error.
- **R²:** Coefficient of determination, measuring explained variation relative to a mean-prediction baseline.
- **Pearson correlation:** Measures linear association between predictions and observed values.
- **Spearman correlation:** Measures rank-order association between predictions and observed values.

Lower RMSE and MAE indicate better prediction accuracy. Higher R² and correlation values generally indicate stronger predictive performance.

### Ensemble strategies

The project evaluates equal-weight and validation-selected weighted ensembles.

For classification, the ensemble combines model outputs and uses a decision threshold selected using validation data.

For regression, the ensemble combines continuous predictions using weights selected on the validation set. These weights are held fixed during test evaluation.

The test set should not be used to tune model hyperparameters, ensemble weights, or decision thresholds.

## Results

The following results are based on the reported held-out test metrics. The regression test set contains 1,941 samples. The classification confusion matrices contain 1,944 samples.

### 1. hERG Classification

| Model / strategy | Threshold | Accuracy | Precision | Sensitivity | Specificity | F1 | ROC-AUC | PR-AUC | MCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation-selected weighted ensemble | 0.5208 | 0.7603 | 0.7882 | 0.8879 | 0.4846 | 0.8351 | 0.7738 | 0.8763 | 0.4116 |
| Equal-weight ensemble | 0.4334 | 0.7634 | 0.7766 | 0.9180 | 0.4293 | 0.8414 | 0.7933 | 0.8887 | 0.4101 |
| ChemBERTa | 0.2800 | 0.7428 | 0.7567 | 0.9195 | 0.3605 | 0.8302 | 0.7488 | 0.8552 | 0.3479 |
| Chemprop | 0.6423 | 0.7248 | 0.8297 | 0.7517 | 0.6667 | 0.7888 | 0.7792 | 0.8771 | 0.4007 |
| XGBoost — maximum MCC | 0.5463 | 0.7495 | 0.7880 | 0.8668 | 0.4959 | 0.8255 | 0.7682 | 0.8713 | 0.3907 |
| XGBoost — sensitivity-constrained strategy | 0.5264 | 0.7577 | 0.7849 | 0.8894 | 0.4732 | 0.8339 | 0.7682 | 0.8713 | 0.4036 |

*Metrics are reported as provided. Thresholds represent the reported operating points. The sensitivity-constrained XGBoost strategy's test sensitivity is below 90%; the constraint is intended for validation-based threshold selection.*

#### Classification observations

- The equal-weight ensemble achieved the highest reported accuracy (76.34%), F1-score (0.8414), ROC-AUC (0.7933), and PR-AUC (0.8887).
- ChemBERTa achieved high sensitivity (91.95%) but lower specificity (36.05%), reflecting a greater tendency to classify compounds as positive at the selected threshold.
- Chemprop achieved the highest precision (82.97%) and specificity (66.67%) among the reported strategies, but its sensitivity was lower (75.17%).
- The validation-selected weighted ensemble achieved an MCC of 0.4116, compared with 0.4101 for the equal-weight ensemble. Its performance advantage depends on the metric considered.
- These results illustrate the trade-off between identifying positive compounds and avoiding false-positive predictions.

### 2. pIC50 Regression

| Model | Test samples | RMSE ↓ | MAE ↓ | R² ↑ | Pearson r ↑ | Spearman r ↑ |
|---|---:|---:|---:|---:|---:|---:|
| ChemBERTa Regression | 1,941 | 0.710454 | 0.522612 | 0.350825 | 0.620339 | 0.563029 |
| XGBoost Regression | 1,941 | 0.682487 | 0.493057 | 0.400927 | 0.638974 | 0.575442 |
| Chemprop Regression | 1,941 | 0.689209 | 0.506381 | 0.389069 | 0.647074 | 0.605330 |
| Equal-weight ensemble | 1,941 | 0.636532 | 0.460889 | 0.478888 | 0.693712 | 0.642734 |
| Weighted ensemble | 1,941 | 0.635461 | 0.460873 | 0.480640 | 0.694477 | 0.644371 |

#### Regression observations

- The weighted ensemble achieved the best reported test performance across all five regression metrics.
- Its RMSE was 0.6355, compared with 0.6825 for XGBoost, 0.6892 for Chemprop, and 0.7105 for ChemBERTa.
- The weighted ensemble achieved an R² of 0.4806, indicating that it explained approximately 48.1% of the observed test-target variance under the reported evaluation setup.
- Chemprop had the strongest Pearson (0.6471) and Spearman (0.6053) correlations among the individual models.
- The weighted ensemble slightly outperformed the equal-weight ensemble, suggesting that validation-selected weights provided a modest improvement over uniform averaging.

The reported metrics should be interpreted in the context of the dataset, target distribution, split strategy, and any target transformations used during training.

## Installation

Clone the repository:

```bash
git clone https://github.com/rpghost7/AIDD-Challenge-4-Project.git
cd AIDD-Challenge-4-Project
```

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Depending on the selected model, dependencies may include:

- NumPy and Pandas
- scikit-learn
- RDKit
- XGBoost
- PyTorch
- Hugging Face Transformers
- Chemprop
- Matplotlib

Compatible versions should be specified in `requirements.txt`. PyTorch installation should account for the target CPU/GPU environment, and Chemprop should be installed using the instructions appropriate to the version used in the experiments.

## Running the Project

Run commands from the repository root.

### Data preparation

```bash
python scripts/scaffold_split.py
python scripts/create_regression_dataset.py
python scripts/split_regression_dataset.py
```

Execute these steps according to the input and output dependencies of the scripts. Check their configurations before rerunning data preparation to avoid overwriting existing datasets or splits.

### Classification

```bash
python classification/trees.py
python classification/chemberta.py
python classification/chemprop.py
python classification/ensemble.py
```

### Regression

```bash
python regression/xgboost_regression.py
python regression/chemberta_regression.py
python regression/chemprop_regression.py
python regression/regression_ensemble.py
```

These commands assume that the scripts use paths relative to the repository root and do not require additional command-line arguments. Check each script for its actual configuration, required inputs, and expected output files before execution.

## Reproducibility

To support reproducibility:

- Use the documented dataset versions and preprocessing steps.
- Preserve the train, validation, and test split assignments.
- Record model configurations, random seeds, package versions, and target transformations.
- Select thresholds and ensemble weights using validation data only.
- Evaluate final model performance on the held-out test set.
- Document the origin of pretrained model checkpoints and any required downloads.

## Limitations

- Predictive performance depends on dataset quality, class balance, target definitions, and the distribution of molecular scaffolds.
- Classification performance depends on the chosen decision threshold and the relative costs of false positives and false negatives.
- Regression metrics may be affected by assay variability, target transformations, and differences in experimental conditions.
- A higher correlation does not necessarily imply lower prediction error.
- Good benchmark performance does not by itself establish experimental or clinical validity.
- Results should be independently reproduced before drawing conclusions about generalization to new chemical space.

