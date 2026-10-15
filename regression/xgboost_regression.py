#!/usr/bin/env python3
"""Train and evaluate an XGBoost pIC50 regressor on Morgan fingerprints.

Expected input files:
    train_regression.csv
    val_regression.csv
    test_regression.csv

Each CSV must contain `smiles` and `median_pIC50`; `inchi_key` is optional.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from xgboost import XGBRegressor

SEED = 123
FP_SIZE = 2048
FP_RADIUS = 2


def clean_smiles(s: object) -> str:
    """Keep the first SMILES token, stripping suffix annotations after ; or space."""
    if pd.isna(s):
        return ""
    return str(s).split(";")[0].split(" ")[0].strip()


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"smiles", "median_pIC50"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["smiles"] = df["smiles"].map(clean_smiles)
    df["median_pIC50"] = pd.to_numeric(df["median_pIC50"], errors="coerce")

    bad_target = df["median_pIC50"].isna() | ~np.isfinite(df["median_pIC50"])
    empty_smiles = df["smiles"].eq("")
    if bad_target.any() or empty_smiles.any():
        bad_rows = df.index[bad_target | empty_smiles].tolist()[:10]
        raise ValueError(
            f"{split_name} contains missing/invalid SMILES or non-finite targets "
            f"after cleaning. Example row indices: {bad_rows}. Fix the source CSV "
            "rather than silently dropping rows."
        )

    # Fail clearly on invalid molecules instead of letting fingerprint generation fail later.
    invalid = [s for s in df["smiles"].tolist() if Chem.MolFromSmiles(s) is None]
    if invalid:
        raise ValueError(
            f"{split_name} contains {len(invalid)} invalid SMILES after cleaning. "
            f"Examples: {invalid[:5]}"
        )

    print(f"{split_name}: {df.shape}; columns={df.columns.tolist()}")
    return df.reset_index(drop=True)


def smiles_to_fingerprints(smiles_list: list[str]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS,
        fpSize=FP_SIZE,
    )
    fingerprints = []
    for smiles in tqdm(smiles_list, desc="Generating Morgan fingerprints"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES encountered after cleaning: {smiles}")
        fp = generator.GetFingerprint(mol)
        arr = np.zeros(FP_SIZE, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fingerprints.append(arr)
    return np.asarray(fingerprints, dtype=np.uint8)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pearson = pearsonr(y_true, y_pred).statistic if len(y_true) >= 2 else np.nan
    spearman = spearmanr(y_true, y_pred).statistic if len(y_true) >= 2 else np.nan
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson_r": float(pearson),
        "Spearman_r": float(spearman),
    }


def make_prediction_df(df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    output = pd.DataFrame({
        "smiles": df["smiles"].to_numpy(),
        "actual_pIC50": df["median_pIC50"].to_numpy(),
        "predicted_pIC50": predictions,
    })
    if "inchi_key" in df.columns:
        output.insert(0, "inchi_key", df["inchi_key"].to_numpy())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost regression using Morgan/ECFP4 fingerprints."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/regression_split"),
        help="Directory containing train_regression.csv, val_regression.csv, test_regression.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/xgboost_regression"),
        help="Directory for model, metrics, predictions and plots",
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(args.data_dir / "train_regression.csv", "Train")
    val_df = load_split(args.data_dir / "val_regression.csv", "Validation")
    test_df = load_split(args.data_dir / "test_regression.csv", "Test")
    print("\\nRegression splits loaded and SMILES cleaned successfully.")

    X_train = smiles_to_fingerprints(train_df["smiles"].tolist())
    X_val = smiles_to_fingerprints(val_df["smiles"].tolist())
    X_test = smiles_to_fingerprints(test_df["smiles"].tolist())

    y_train = train_df["median_pIC50"].to_numpy(dtype=np.float32)
    y_val = val_df["median_pIC50"].to_numpy(dtype=np.float32)
    y_test = test_df["median_pIC50"].to_numpy(dtype=np.float32)

    print("\\nFeature shapes:")
    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    param_grid = list(product([3, 4, 5], [1, 3], [0.03, 0.05]))
    results = []
    best_model = None
    best_val_rmse = float("inf")
    best_params = None
    best_val_pred = None

    for trial, (max_depth, min_child_weight, learning_rate) in enumerate(param_grid, start=1):
        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=600,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
        )
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        metrics = regression_metrics(y_val, val_pred)
        results.append({
            "trial": trial,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "learning_rate": learning_rate,
            **{f"val_{key}": value for key, value in metrics.items()},
        })
        print(
            f"Trial {trial:02d}/{len(param_grid)} | depth={max_depth} | "
            f"child_weight={min_child_weight} | lr={learning_rate} | "
            f"Val RMSE={metrics['RMSE']:.4f} | Val MAE={metrics['MAE']:.4f} | "
            f"Val R2={metrics['R2']:.4f}"
        )
        if metrics["RMSE"] < best_val_rmse:
            best_val_rmse = metrics["RMSE"]
            best_model = model
            best_params = {
                "max_depth": max_depth,
                "min_child_weight": min_child_weight,
                "learning_rate": learning_rate,
            }
            best_val_pred = val_pred.copy()

    results_df = pd.DataFrame(results).sort_values("val_RMSE")
    results_df.to_csv(args.output_dir / "hyperparameter_search.csv", index=False)

    print("\\nBest hyperparameters:")
    print(best_params)
    print(f"Best validation RMSE: {best_val_rmse:.4f}")

    best_test_pred = best_model.predict(X_test)
    val_metrics = regression_metrics(y_val, best_val_pred)
    test_metrics = regression_metrics(y_test, best_test_pred)
    metrics_df = pd.DataFrame([
        {"Split": "Validation", **val_metrics},
        {"Split": "Test", **test_metrics},
    ])
    print("\\nFinal regression metrics:")
    print(metrics_df.round(4).to_string(index=False))
    metrics_df.to_csv(args.output_dir / "regression_metrics.csv", index=False)

    make_prediction_df(val_df, best_val_pred).to_csv(
        args.output_dir / "val_evaluated_predictions.csv", index=False
    )
    make_prediction_df(test_df, best_test_pred).to_csv(
        args.output_dir / "test_evaluated_predictions.csv", index=False
    )
    best_model.save_model(args.output_dir / "xgboost_regression.json")
    with open(args.output_dir / "xgboost_regression_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED,
            "fingerprint": {"type": "Morgan/ECFP4", "radius": FP_RADIUS, "size": FP_SIZE},
            "best_hyperparameters": best_params,
            "best_validation_RMSE": best_val_rmse,
            "smiles_cleaning": 'str(s).split(";")[0].split(" ")[0].strip()',
        }, f, indent=2)

    # Save actual-vs-predicted plot if matplotlib is installed.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, actual, predicted, split_name in [
            (axes[0], y_val, best_val_pred, "Validation"),
            (axes[1], y_test, best_test_pred, "Test"),
        ]:
            ax.scatter(actual, predicted, alpha=0.45, s=16)
            lower = min(float(actual.min()), float(predicted.min()))
            upper = max(float(actual.max()), float(predicted.max()))
            ax.plot([lower, upper], [lower, upper], linestyle="--")
            ax.set_xlabel("Actual pIC50")
            ax.set_ylabel("Predicted pIC50")
            ax.set_title(f"XGBoost Regression — {split_name}")
            ax.grid(alpha=0.2)
        plt.tight_layout()
        fig.savefig(args.output_dir / "actual_vs_predicted.png", dpi=200)
        plt.close(fig)
    except ImportError:
        print("matplotlib not installed; skipping actual_vs_predicted.png.")

    print(f"\\nSaved outputs to: {args.output_dir.resolve()}")
    for path in sorted(args.output_dir.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
