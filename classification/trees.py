"""
hERG blocker classification using Morgan/ECFP4 fingerprints and XGBoost.

Expected input files:
    <data_dir>/train.csv
    <data_dir>/val.csv
    <data_dir>/test.csv

Required columns:
    smiles, hERG_blocker

Thresholds are selected using validation data only:
    1. Maximum MCC (unconstrained)
    2. Maximum MCC subject to sensitivity >= 0.90

The held-out test set is used only for final evaluation.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm
from xgboost import XGBClassifier


RANDOM_STATE = 123
TARGET_SENSITIVITY = 0.90
N_FINGERPRINT_BITS = 2048
MORGAN_RADIUS = 2
N_ESTIMATORS = 600
N_PARAMETER_COMBINATIONS = 15


def load_splits(data_dir: Path):
    """Load and validate the fixed train/validation/test CSV splits."""
    splits = {}
    for name in ("train", "val", "test"):
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Could not find split file: {path}")

        df = pd.read_csv(path)
        missing = {"smiles", "hERG_blocker"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        if df["smiles"].isna().any():
            raise ValueError(f"{path} contains missing SMILES.")
        if df["hERG_blocker"].isna().any():
            raise ValueError(f"{path} contains missing hERG labels.")

        labels = set(df["hERG_blocker"].unique())
        if not labels.issubset({0, 1}):
            raise ValueError(
                f"{path}: hERG_blocker must contain only 0/1 labels; got {labels}"
            )

        splits[name] = df
        print(f"{name.capitalize():>10}: {df.shape}; labels={df['hERG_blocker'].value_counts().to_dict()}")

    return splits


def smiles_to_fingerprints(smiles_values) -> np.ndarray:
    """Convert SMILES strings to 2048-bit Morgan fingerprints (radius 2)."""
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=N_FINGERPRINT_BITS,
    )
    fingerprints = []

    for smiles in tqdm(smiles_values, desc="Generating fingerprints"):
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            raise ValueError(f"Invalid SMILES encountered: {smiles}")

        fingerprint = generator.GetFingerprint(molecule)
        array = np.zeros(N_FINGERPRINT_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        fingerprints.append(array)

    return np.asarray(fingerprints, dtype=np.uint8)


def choose_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    target_sensitivity: float = TARGET_SENSITIVITY,
) -> tuple[dict, dict | None, pd.DataFrame]:
    """Choose maximum-MCC and sensitivity-constrained thresholds on validation data."""
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    rows = []

    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        rows.append({
            "threshold": float(threshold),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "MCC": float(matthews_corrcoef(y_true, predictions)),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        })

    threshold_df = pd.DataFrame(rows)

    # Strategy A: highest MCC regardless of sensitivity.
    best_mcc = threshold_df.sort_values(
        ["MCC", "specificity", "threshold"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()

    # Strategy B: highest MCC while satisfying the sensitivity target.
    eligible = threshold_df[
        threshold_df["sensitivity"] >= target_sensitivity
    ]
    best_high_sensitivity = None
    if not eligible.empty:
        best_high_sensitivity = eligible.sort_values(
            ["MCC", "specificity", "threshold"],
            ascending=[False, False, False],
        ).iloc[0].to_dict()

    return best_mcc, best_high_sensitivity, threshold_df


def evaluate_model(
    strategy: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    """Evaluate probabilities at a specified threshold."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "Model/Strategy": strategy,
        "Threshold": float(threshold),
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, probabilities)),
        "PR-AUC": float(average_precision_score(y_true, probabilities)),
        "MCC": float(matthews_corrcoef(y_true, predictions)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def build_parameter_candidates():
    """Reproduce the parameter grid and deterministic 15-combination sample."""
    parameter_grid = list(product(
        [3, 4, 5],       # max_depth
        [1, 3],          # min_child_weight
        [0.03, 0.05],    # learning_rate
        [0.8],           # subsample
        [0.8],           # colsample_bytree
        [1.0, 1.5],      # scale_pos_weight
    ))

    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(
        len(parameter_grid),
        size=min(N_PARAMETER_COMBINATIONS, len(parameter_grid)),
        replace=False,
    )
    return [parameter_grid[i] for i in indices]


def main(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(data_dir)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    print("\nGenerating ECFP4 fingerprints...")
    X_train = smiles_to_fingerprints(train_df["smiles"].tolist())
    X_val = smiles_to_fingerprints(val_df["smiles"].tolist())
    X_test = smiles_to_fingerprints(test_df["smiles"].tolist())

    y_train = train_df["hERG_blocker"].astype(int).to_numpy()
    y_val = val_df["hERG_blocker"].astype(int).to_numpy()
    y_test = test_df["hERG_blocker"].astype(int).to_numpy()

    print(f"\nFingerprint shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    tuning_rows = []
    trained_models = []

    for trial, params in enumerate(build_parameter_candidates(), start=1):
        (
            max_depth,
            min_child_weight,
            learning_rate,
            subsample,
            colsample_bytree,
            scale_pos_weight,
        ) = params

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=N_ESTIMATORS,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            reg_lambda=2.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )
        model.fit(X_train, y_train)

        val_probabilities = model.predict_proba(X_val)[:, 1]
        best_mcc, best_90, _ = choose_thresholds(y_val, val_probabilities)

        # Tune configurations using validation MCC at the >=90% sensitivity operating point.
        operating_point = best_90
        row = {
            "trial": trial,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "val_PR_AUC": float(average_precision_score(y_val, val_probabilities)),
            "val_ROC_AUC": float(roc_auc_score(y_val, val_probabilities)),
            "val_max_MCC": float(best_mcc["MCC"]),
            "val_max_MCC_threshold": float(best_mcc["threshold"]),
            "val_MCC_at_90_sensitivity": float(operating_point["MCC"]) if operating_point else np.nan,
            "val_90_sensitivity_threshold": float(operating_point["threshold"]) if operating_point else np.nan,
            "val_sensitivity_at_selected_threshold": float(operating_point["sensitivity"]) if operating_point else np.nan,
            "val_specificity_at_selected_threshold": float(operating_point["specificity"]) if operating_point else np.nan,
        }
        tuning_rows.append(row)
        trained_models.append((model, val_probabilities, row))

        score = row["val_MCC_at_90_sensitivity"]
        print(
            f"Trial {trial:02d}/{N_PARAMETER_COMBINATIONS} | "
            f"PR-AUC={row['val_PR_AUC']:.4f} | "
            f"MCC@>=90% sensitivity={score:.4f}"
            if np.isfinite(score)
            else f"Trial {trial:02d}/{N_PARAMETER_COMBINATIONS} | "
                 f"PR-AUC={row['val_PR_AUC']:.4f} | no threshold met 90% sensitivity"
        )

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(output_dir / "xgboost_tuning_results.csv", index=False)

    eligible_rows = tuning_df[
        tuning_df["val_sensitivity_at_selected_threshold"] >= TARGET_SENSITIVITY
    ]
    if eligible_rows.empty:
        raise RuntimeError(
            "No XGBoost configuration achieved >=90% sensitivity on validation data. "
            "No model was selected; inspect xgboost_tuning_results.csv."
        )

    # Primary ranking: constrained MCC, then PR-AUC, then specificity.
    selected_row = eligible_rows.sort_values(
        [
            "val_MCC_at_90_sensitivity",
            "val_PR_AUC",
            "val_specificity_at_selected_threshold",
        ],
        ascending=[False, False, False],
    ).iloc[0]
    selected_trial = int(selected_row["trial"])
    best_model, best_val_prob, _ = trained_models[selected_trial - 1]

    # Recompute both operating points for the selected model on validation data.
    best_mcc, best_90, threshold_df = choose_thresholds(y_val, best_val_prob)
    threshold_df.to_csv(output_dir / "xgboost_validation_thresholds.csv", index=False)

    print("\nSelected XGBoost configuration:")
    print(selected_row.to_string())
    print("\nValidation threshold — maximum MCC:")
    print(pd.Series(best_mcc).to_string())
    print("\nValidation threshold — maximum MCC with >=90% sensitivity:")
    if best_90 is None:
        raise RuntimeError("Selected model unexpectedly has no >=90% sensitivity threshold.")
    print(pd.Series(best_90).to_string())

    # Save selected model and threshold/configuration metadata.
    best_model.save_model(str(output_dir / "xgboost_tuned.json"))
    with open(output_dir / "xgboost_tuned_config.json", "w", encoding="utf-8") as file:
        json.dump({
            "fingerprint": {
                "type": "Morgan/ECFP4",
                "radius": MORGAN_RADIUS,
                "n_bits": N_FINGERPRINT_BITS,
            },
            "target_sensitivity": TARGET_SENSITIVITY,
            "selected_trial": selected_trial,
            "selected_hyperparameters": {
                key: (int(selected_row[key]) if key in {"max_depth", "min_child_weight"}
                      else float(selected_row[key]))
                for key in (
                    "max_depth", "min_child_weight", "learning_rate",
                    "subsample", "colsample_bytree", "scale_pos_weight"
                )
            },
            "validation_thresholds": {
                "maximum_MCC": best_mcc,
                "maximum_MCC_at_target_sensitivity": best_90,
            },
        }, file, indent=2)

    # Final held-out test evaluation: thresholds are already fixed from validation.
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_results = [
        evaluate_model(
            "XGBoost — max MCC",
            y_test,
            test_probabilities,
            best_mcc["threshold"],
        ),
        evaluate_model(
            "XGBoost — max MCC at >=90% validation sensitivity",
            y_test,
            test_probabilities,
            best_90["threshold"],
        ),
    ]
    test_metrics_df = pd.DataFrame(test_results)
    test_metrics_df.to_csv(output_dir / "xgboost_tuned_test_metrics.csv", index=False)

    pd.DataFrame({
        "smiles": val_df["smiles"].to_numpy(),
        "y_true": y_val,
        "xgboost_probability": best_val_prob,
    }).to_csv(output_dir / "xgboost_tuned_val_predictions.csv", index=False)

    pd.DataFrame({
        "smiles": test_df["smiles"].to_numpy(),
        "y_true": y_test,
        "xgboost_probability": test_probabilities,
    }).to_csv(output_dir / "xgboost_tuned_test_predictions.csv", index=False)

    print("\nHeld-out test metrics:")
    print(test_metrics_df.round(4).to_string(index=False))
    print(f"\nAll outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate Morgan-fingerprint XGBoost for hERG classification."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/scaffoldsplit"),
        help="Directory containing train.csv, val.csv and test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/trees"),
        help="Directory in which models, predictions and metrics will be saved.",
    )
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
