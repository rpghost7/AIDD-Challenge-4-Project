#!/usr/bin/env python3
"""
Chemprop 2.3.1 hERG classification pipeline.

- Trains Chemprop on the supplied training CSV only.
- Predicts probabilities on the fixed validation and test CSVs.
- Selects thresholds on validation data only:
    A) maximum MCC
    B) maximum MCC subject to sensitivity >= 90% (primary strategy)
- Evaluates both fixed thresholds on the held-out test set.
- Saves predictions, threshold scan, metrics, and run configuration.

Important:
The original notebook passed train.csv, val.csv, and test.csv together to
`chemprop train`. Chemprop interpreted these as input data files and applied
its configured random split, so the supplied test set was not kept strictly
held out. This script intentionally passes only train.csv to training.
Chemprop's internal split is used for checkpoint/early-stopping selection;
the fixed validation set is used for threshold selection.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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


def run_command(command: list[str]) -> None:
    """Run a command and stop immediately if it fails."""
    print("\n$", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def load_split(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"smiles", "hERG_blocker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} CSV is missing columns: {sorted(missing)}")
    if df["smiles"].isna().any() or df["hERG_blocker"].isna().any():
        raise ValueError(f"{name} CSV contains missing SMILES or labels.")
    labels = set(pd.to_numeric(df["hERG_blocker"], errors="raise").unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{name} labels must be binary 0/1; found {labels}.")
    return df


def load_predictions(
    input_df: pd.DataFrame, prediction_path: Path, split_name: str
) -> tuple[np.ndarray, np.ndarray]:
    pred_df = pd.read_csv(prediction_path)
    if len(input_df) != len(pred_df):
        raise ValueError(
            f"{split_name}: row count mismatch: input={len(input_df)}, "
            f"predictions={len(pred_df)}."
        )
    if "smiles" in pred_df.columns:
        same_order = (
            input_df["smiles"].astype(str).to_numpy()
            == pred_df["smiles"].astype(str).to_numpy()
        ).all()
        if not same_order:
            raise ValueError(f"{split_name}: SMILES order mismatch in predictions.")

    if "hERG_blocker" not in pred_df.columns:
        raise ValueError(
            f"{prediction_path} has no 'hERG_blocker' probability column. "
            f"Found columns: {pred_df.columns.tolist()}"
        )

    y = pd.to_numeric(input_df["hERG_blocker"], errors="raise").astype(int).to_numpy()
    probabilities = pd.to_numeric(
        pred_df["hERG_blocker"], errors="raise"
    ).to_numpy(dtype=float)

    if not np.isfinite(probabilities).all():
        raise ValueError(f"{split_name} predictions contain NaN or infinite values.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(
            f"{split_name} predictions are not probabilities in [0, 1]."
        )
    return y, probabilities


def threshold_table(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    # Include all distinct model probabilities and endpoints so the threshold
    # scan covers meaningful classification changes.
    thresholds = np.unique(np.concatenate(([0.0], probabilities, [1.0 + 1e-12])))
    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, predicted, labels=[0, 1]
        ).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "MCC": float(matthews_corrcoef(y_true, predicted)),
                "sensitivity": sensitivity,
                "specificity": specificity,
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }
        )
    return pd.DataFrame(rows)


def calculate_test_metrics(
    strategy: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predicted, labels=[0, 1]
    ).ravel()
    return {
        "strategy": strategy,
        "threshold_selected_on_validation": float(threshold),
        "Accuracy": accuracy_score(y_true, predicted),
        "Precision": precision_score(y_true, predicted, zero_division=0),
        "Sensitivity / Recall": recall_score(
            y_true, predicted, zero_division=0
        ),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "F1-score": f1_score(y_true, predicted, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC (Average Precision)": average_precision_score(
            y_true, probabilities
        ),
        "MCC": matthews_corrcoef(y_true, predicted),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def save_predictions(
    input_df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    path: Path,
) -> None:
    output = input_df[["smiles", "hERG_blocker"]].copy()
    output["probability_hERG_blocker"] = probabilities
    output["predicted_label_90_sensitivity_mcc"] = (
        probabilities >= threshold
    ).astype(int)
    output.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Chemprop and select a validation threshold for hERG classification."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/scaffoldsplit"),
        help="Directory containing train.csv, val.csv, and test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/chemprop"),
        help="Directory in which to save the model and results.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sensitivity-target", type=float, default=0.90)
    parser.add_argument(
        "--accelerator",
        choices=["gpu", "cpu", "auto"],
        default="gpu",
    )
    parser.add_argument("--devices", default="1")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and use an existing model under output-dir/model_0/best.pt.",
    )
    args = parser.parse_args()

    if not 0 < args.sensitivity_target <= 1:
        parser.error("--sensitivity-target must be in (0, 1].")

    train_path = args.data_dir / "train.csv"
    val_path = args.data_dir / "val.csv"
    test_path = args.data_dir / "test.csv"
    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Could not find required file: {path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(train_path, "Train")
    val_df = load_split(val_path, "Validation")
    test_df = load_split(test_path, "Test")

    for name, df in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        print(
            f"{name}: {len(df):,} rows; "
            f"class counts={df['hERG_blocker'].value_counts().to_dict()}"
        )

    model_dir = output_dir / "chemprop_model"
    if not args.skip_training:
        if model_dir.exists():
            shutil.rmtree(model_dir)
        train_command = [
            "chemprop", "train",
            "-i", str(train_path.resolve()),
            "-o", str(model_dir),
            "-s", "smiles",
            "--target-columns", "hERG_blocker",
            "-t", "classification",
            "--metrics", "prc", "roc", "binary-mcc",
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "-b", str(args.batch_size),
            "--accelerator", args.accelerator,
            "--devices", args.devices,
            "--pytorch-seed", str(args.seed),
        ]
        run_command(train_command)

    model_path = model_dir / "model_0" / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Expected trained model at {model_path}. "
            "Check Chemprop training output or use the correct --output-dir."
        )

    val_pred_path = output_dir / "chemprop_val_predictions_raw.csv"
    test_pred_path = output_dir / "chemprop_test_predictions_raw.csv"

    for split_path, prediction_path in [
        (val_path, val_pred_path),
        (test_path, test_pred_path),
    ]:
        predict_command = [
            "chemprop", "predict",
            "-i", str(split_path.resolve()),
            "-o", str(prediction_path),
            "-s", "smiles",
            "--model-paths", str(model_path.resolve()),
            "--accelerator", args.accelerator,
            "--devices", args.devices,
        ]
        run_command(predict_command)

    y_val, p_val = load_predictions(val_df, val_pred_path, "Validation")
    y_test, p_test = load_predictions(test_df, test_pred_path, "Test")

    thresholds = threshold_table(y_val, p_val)
    thresholds.to_csv(output_dir / "chemprop_validation_threshold_scan.csv", index=False)

    # Strategy A: unconstrained maximum MCC.
    best_mcc = thresholds.sort_values(
        ["MCC", "threshold"], ascending=[False, False]
    ).iloc[0]

    # Strategy B (primary): maximum MCC among thresholds with sensitivity >= target.
    eligible = thresholds[thresholds["sensitivity"] >= args.sensitivity_target]
    if eligible.empty:
        raise ValueError(
            f"No validation threshold achieved sensitivity >= "
            f"{args.sensitivity_target:.1%}. No constrained threshold was selected."
        )
    best_sensitivity_mcc = eligible.sort_values(
        ["MCC", "specificity", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]

    print("\n===== VALIDATION THRESHOLDS =====")
    print("\nMaximum MCC (comparison):")
    print(best_mcc.to_string())
    print(f"\nMaximum MCC with sensitivity >= {args.sensitivity_target:.1%} (PRIMARY):")
    print(best_sensitivity_mcc.to_string())

    threshold_mcc = float(best_mcc["threshold"])
    threshold_primary = float(best_sensitivity_mcc["threshold"])

    metrics = pd.DataFrame(
        [
            calculate_test_metrics(
                "max_MCC_unconstrained", y_test, p_test, threshold_mcc
            ),
            calculate_test_metrics(
                f"max_MCC_sensitivity_at_least_{args.sensitivity_target:.2f}",
                y_test, p_test, threshold_primary
            ),
        ]
    )
    metrics.to_csv(output_dir / "chemprop_test_metrics.csv", index=False)

    save_predictions(
        val_df, p_val, threshold_primary,
        output_dir / "chemprop_val_predictions.csv"
    )
    save_predictions(
        test_df, p_test, threshold_primary,
        output_dir / "chemprop_test_predictions.csv"
    )

    config = {
        "model": "Chemprop",
        "chemprop_model_path": str(model_path),
        "train_csv": str(train_path.resolve()),
        "validation_csv": str(val_path.resolve()),
        "test_csv": str(test_path.resolve()),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "sensitivity_target": args.sensitivity_target,
        "threshold_max_mcc_comparison": threshold_mcc,
        "threshold_primary_sensitivity_constrained_mcc": threshold_primary,
        "primary_strategy": "maximum validation MCC subject to validation sensitivity >= target",
        "thresholds_selected_using": "validation set only",
        "test_set_used_for_training_or_threshold_selection": False,
    }
    with open(output_dir / "chemprop_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\n===== TEST RESULTS (BOTH THRESHOLDS) =====")
    print(metrics.to_string(index=False))
    print(f"\nPrimary threshold: {threshold_primary:.6f}")
    print(f"Saved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
