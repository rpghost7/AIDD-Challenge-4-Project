#!/usr/bin/env python3
"""
Train and evaluate a Chemprop regression model for median_pIC50.

This script follows the workflow in the Chemprop regression notebook:
- uses the existing train/validation/test regression splits;
- cleans SMILES as a defensive preprocessing step;
- trains Chemprop with validation and test files kept separate;
- predicts validation and test sets using the best checkpoint;
- evaluates RMSE, MAE, R2, Pearson r, and Spearman r;
- saves predictions, metrics, plots, and training outputs.

Requires Chemprop 2.x. Check `chemprop train --help` if using another major version.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_SEED = 123
TARGET_COLUMN = "median_pIC50"
SMILES_COLUMN = "smiles"
ID_COLUMN = "inchi_key"


def clean_smiles(value: object) -> str:
    """Apply the same defensive cleaning rule used elsewhere in the project."""
    if pd.isna(value):
        return ""
    return str(value).split(";")[0].split(" ")[0].strip()


def validate_and_clean_split(path: Path, split_name: str, output_dir: Path) -> pd.DataFrame:
    """Load a regression split, clean SMILES, validate targets, and save a clean copy."""
    if not path.exists():
        raise FileNotFoundError(
            f"{split_name} split not found: {path}\n"
            "Generate the regression splits before running this script."
        )

    df = pd.read_csv(path)
    required = {SMILES_COLUMN, TARGET_COLUMN, ID_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{split_name} file {path} is missing required columns: {sorted(missing)}. "
            f"Found columns: {df.columns.tolist()}"
        )

    df = df.copy()
    df[SMILES_COLUMN] = df[SMILES_COLUMN].map(clean_smiles)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df[ID_COLUMN] = df[ID_COLUMN].astype("string").str.strip()

    invalid_smiles = df[SMILES_COLUMN].eq("")
    invalid_target = ~np.isfinite(df[TARGET_COLUMN].to_numpy(dtype=float, na_value=np.nan))
    missing_id = df[ID_COLUMN].isna() | df[ID_COLUMN].eq("")

    # RDKit validation is an extra safeguard when RDKit is installed.
    rdkit_invalid = pd.Series(False, index=df.index)
    try:
        from rdkit import Chem

        parsed = df[SMILES_COLUMN].map(lambda s: Chem.MolFromSmiles(s) if s else None)
        rdkit_invalid = parsed.isna()
    except ImportError:
        print("WARNING: RDKit is not installed; skipping RDKit SMILES validity check.")

    invalid = invalid_smiles | invalid_target | missing_id | rdkit_invalid
    if invalid.any():
        bad_count = int(invalid.sum())
        bad_path = output_dir / f"{split_name.lower()}_rejected_rows.csv"
        df.loc[invalid].to_csv(bad_path, index=False)
        print(
            f"WARNING: {split_name}: excluding {bad_count} invalid row(s); "
            f"details saved to {bad_path}"
        )
        df = df.loc[~invalid].copy()

    if df.empty:
        raise ValueError(f"No valid rows remain in the {split_name} split after cleaning.")

    # Duplicate InChIKeys can break identity-based one-to-one evaluation and may
    # indicate a split-generation/data-preparation issue, so stop rather than guess.
    if df[ID_COLUMN].duplicated().any():
        examples = df.loc[df[ID_COLUMN].duplicated(keep=False), ID_COLUMN].head(10).tolist()
        raise ValueError(
            f"{split_name} contains duplicate InChIKeys (examples: {examples}). "
            "Resolve duplicates before training/evaluation."
        )

    clean_path = output_dir / f"{split_name.lower()}_regression_clean.csv"
    df.to_csv(clean_path, index=False)
    print(
        f"{split_name}: {len(df):,} valid rows | "
        f"missing/invalid SMILES removed as needed | saved {clean_path}"
    )
    return df.reset_index(drop=True)


def run_command(command: list[str]) -> None:
    """Run a subprocess command and show a useful error if it fails."""
    print("\nRunning:\n  " + " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def safe_corr(function, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return float("nan")
    try:
        return float(function(y_true, y_pred).statistic)
    except (ValueError, FloatingPointError):
        return float("nan")


def evaluate_predictions(
    actual_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    split_name: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Join predictions to actual values by InChIKey and calculate metrics."""
    if ID_COLUMN not in prediction_df.columns:
        raise ValueError(
            f"Chemprop prediction file lacks {ID_COLUMN!r}. "
            "The prediction input must include the compound identifier."
        )

    # Chemprop's output column may be named after the target. Identify it safely.
    possible_prediction_columns = [
        col for col in prediction_df.columns
        if col not in {ID_COLUMN, SMILES_COLUMN, TARGET_COLUMN}
    ]
    if TARGET_COLUMN in prediction_df.columns:
        # If Chemprop preserved the target from its input, it is not a prediction.
        # A prediction-only input is used below, so a target-named output is a prediction.
        prediction_column = TARGET_COLUMN
    elif len(possible_prediction_columns) == 1:
        prediction_column = possible_prediction_columns[0]
    else:
        raise ValueError(
            "Could not identify the Chemprop prediction column. "
            f"Columns found: {prediction_df.columns.tolist()}"
        )

    actual = actual_df[[ID_COLUMN, SMILES_COLUMN, TARGET_COLUMN]].rename(
        columns={TARGET_COLUMN: "actual_pIC50"}
    )
    predicted = prediction_df[[ID_COLUMN, prediction_column]].rename(
        columns={prediction_column: "predicted_pIC50"}
    )

    comparison = actual.merge(
        predicted, on=ID_COLUMN, how="inner", validate="one_to_one"
    )
    comparison["actual_pIC50"] = pd.to_numeric(comparison["actual_pIC50"], errors="coerce")
    comparison["predicted_pIC50"] = pd.to_numeric(
        comparison["predicted_pIC50"], errors="coerce"
    )
    valid = np.isfinite(comparison["actual_pIC50"]) & np.isfinite(
        comparison["predicted_pIC50"]
    )
    comparison = comparison.loc[valid].copy()

    if comparison.empty:
        raise ValueError(
            f"No matched, finite predictions found for {split_name}. "
            "Check Chemprop output and InChIKey matching."
        )

    y_true = comparison["actual_pIC50"].to_numpy(dtype=float)
    y_pred = comparison["predicted_pIC50"].to_numpy(dtype=float)
    metrics = {
        "Split": split_name,
        "N": len(comparison),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "Pearson_r": safe_corr(pearsonr, y_true, y_pred),
        "Spearman_r": safe_corr(spearmanr, y_true, y_pred),
    }
    return metrics, comparison


def make_plots(val_comparison: pd.DataFrame, test_comparison: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib is not installed; skipping plots.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, comparison, name in [
        (axes[0], val_comparison, "Validation"),
        (axes[1], test_comparison, "Test"),
    ]:
        actual = comparison["actual_pIC50"].to_numpy(dtype=float)
        predicted = comparison["predicted_pIC50"].to_numpy(dtype=float)
        ax.scatter(actual, predicted, alpha=0.5)
        lower = float(min(actual.min(), predicted.min()))
        upper = float(max(actual.max(), predicted.max()))
        if math.isclose(lower, upper):
            upper = lower + 1.0
        ax.plot([lower, upper], [lower, upper], linestyle="--")
        ax.set_xlabel("Actual pIC50")
        ax.set_ylabel("Predicted pIC50")
        ax.set_title(f"Chemprop: {name}")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
    plt.tight_layout()
    fig.savefig(out_dir / "actual_vs_predicted.png", dpi=200)
    plt.close(fig)

    residuals = test_comparison["actual_pIC50"] - test_comparison["predicted_pIC50"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(test_comparison["predicted_pIC50"], residuals, alpha=0.5)
    ax.axhline(0, linestyle="--")
    ax.set_xlabel("Predicted pIC50")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Chemprop Test Residuals")
    fig.tight_layout()
    fig.savefig(out_dir / "test_residuals.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Chemprop regression on fixed scaffold splits."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/regression_split"),
        help="Directory containing train_regression.csv, val_regression.csv, test_regression.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/chemprop_regression"),
        help="Directory for model, predictions, metrics, and plots",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=MODEL_SEED)
    parser.add_argument(
        "--accelerator", choices=["auto", "cpu", "gpu"], default="auto",
        help="Use 'auto' to select GPU when available, otherwise CPU.",
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--chemprop-command", default="chemprop",
        help="Chemprop executable name/path (default: chemprop)",
    )
    args = parser.parse_args()

    if shutil.which(args.chemprop_command) is None:
        raise SystemExit(
            f"Cannot find Chemprop executable {args.chemprop_command!r}. "
            "Install Chemprop 2.x or pass --chemprop-command with its executable path."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir = args.output_dir / "cleaned_inputs"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "model"
    results_dir = args.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_df = validate_and_clean_split(
        args.data_dir / "train_regression.csv", "Train", cleaned_dir
    )
    val_df = validate_and_clean_split(
        args.data_dir / "val_regression.csv", "Validation", cleaned_dir
    )
    test_df = validate_and_clean_split(
        args.data_dir / "test_regression.csv", "Test", cleaned_dir
    )

    # Check no compound crosses split boundaries.
    sets = {
        "train": set(train_df[ID_COLUMN].astype(str)),
        "validation": set(val_df[ID_COLUMN].astype(str)),
        "test": set(test_df[ID_COLUMN].astype(str)),
    }
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = sets[left] & sets[right]
        if overlap:
            raise ValueError(
                f"InChIKey leakage between {left} and {right}: "
                f"{len(overlap)} overlapping compound(s)."
            )

    train_path = cleaned_dir / "train_regression_clean.csv"
    val_path = cleaned_dir / "validation_regression_clean.csv"
    test_path = cleaned_dir / "test_regression_clean.csv"

    # Chemprop 2.x accepts separate validation/test paths; do not pass all three
    # filenames as extra arguments to --data-path.
    train_cmd = [
        args.chemprop_command, "train",
        "--data-path", str(train_path),
        "--separate-val-path", str(val_path),
        "--separate-test-path", str(test_path),
        "--smiles-columns", SMILES_COLUMN,
        "--target-columns", TARGET_COLUMN,
        "--task-type", "regression",
        "--metrics", "rmse", "mae", "r2",
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--accelerator", args.accelerator,
        "--devices", str(args.devices),
        "--pytorch-seed", str(args.seed),
        "--output-dir", str(model_dir),
    ]
    run_command(train_cmd)

    # Find the best checkpoint produced by Chemprop.
    candidates = list(model_dir.rglob("best.pt"))
    if not candidates:
        candidates = list(model_dir.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No .pt checkpoint found under {model_dir}. "
            "Inspect Chemprop's training output and your installed Chemprop version."
        )
    best_model = candidates[0]
    print(f"\nUsing checkpoint: {best_model}")

    prediction_files: dict[str, Path] = {}
    for split_name, df, clean_path in [
        ("val", val_df, val_path),
        ("test", test_df, test_path),
    ]:
        # Avoid passing the target to predict, which could cause ambiguity if
        # Chemprop preserves input columns. Keep InChIKey for reliable matching.
        predict_input = results_dir / f"{split_name}_predict_input.csv"
        df[[ID_COLUMN, SMILES_COLUMN]].to_csv(predict_input, index=False)
        pred_path = results_dir / f"{split_name}_predictions.csv"
        predict_cmd = [
            args.chemprop_command, "predict",
            "-i", str(predict_input),
            "-o", str(pred_path),
            "-s", SMILES_COLUMN,
            "--model-paths", str(best_model),
            "--accelerator", args.accelerator,
            "--devices", str(args.devices),
        ]
        run_command(predict_cmd)
        prediction_files[split_name] = pred_path

    val_pred = pd.read_csv(prediction_files["val"])
    test_pred = pd.read_csv(prediction_files["test"])

    val_metrics, val_comparison = evaluate_predictions(val_df, val_pred, "Validation")
    test_metrics, test_comparison = evaluate_predictions(test_df, test_pred, "Test")
    metrics_df = pd.DataFrame([val_metrics, test_metrics])

    metrics_path = results_dir / "regression_metrics.csv"
    val_comparison_path = results_dir / "val_evaluated_predictions.csv"
    test_comparison_path = results_dir / "test_evaluated_predictions.csv"
    metrics_df.to_csv(metrics_path, index=False)
    val_comparison.to_csv(val_comparison_path, index=False)
    test_comparison.to_csv(test_comparison_path, index=False)
    make_plots(val_comparison, test_comparison, results_dir)

    config = {
        "model": "Chemprop",
        "task_type": "regression",
        "target_column": TARGET_COLUMN,
        "smiles_column": SMILES_COLUMN,
        "id_column": ID_COLUMN,
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "best_checkpoint": str(best_model),
        "cleaning_rule": "str(value).split(';')[0].split(' ')[0].strip()",
        "cleaned_input_dir": str(cleaned_dir),
    }
    with (args.output_dir / "chemprop_regression_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\n===== REGRESSION METRICS =====")
    print(metrics_df.round(4).to_string(index=False))
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved validation predictions: {val_comparison_path}")
    print(f"Saved test predictions: {test_comparison_path}")
    print(f"Saved results and plots under: {results_dir}")


if __name__ == "__main__":
    main()
