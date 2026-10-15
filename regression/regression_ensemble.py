#!/usr/bin/env python3
"""Evaluate and combine ChemBERTa, Chemprop, and XGBoost pIC50 regressors.

The script reads prediction CSVs produced by:
  - chemberta_regression.py
  - chemprop_regression.py
  - xgboost_regression.py

Each model should have its own output directory because all three save
val_evaluated_predictions.csv and test_evaluated_predictions.csv.

Weights are fitted using VALIDATION predictions only, by minimizing validation
MSE with non-negative weights summing to one. The selected weights are then
kept fixed for the held-out test evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_NAMES = ["ChemBERTa", "Chemprop", "XGBoost"]
PREDICTION_FILES = {
    "val": "val_evaluated_predictions.csv",
    "test": "test_evaluated_predictions.csv",
}
ID_COL = "inchi_key"
TRUE_COL = "actual_pIC50"
PRED_COL = "predicted_pIC50"


def safe_corr(func, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(func(y_true, y_pred).statistic)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model: str, split: str) -> dict:
    return {
        "Split": split,
        "Model": model,
        "N": int(len(y_true)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        "Pearson_r": safe_corr(pearsonr, y_true, y_pred),
        "Spearman_r": safe_corr(spearmanr, y_true, y_pred),
    }


def read_model_predictions(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} prediction file not found: {path}\n"
            "Run the corresponding regression model first."
        )
    df = pd.read_csv(path)
    required = [ID_COL, TRUE_COL, PRED_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{model_name} file {path} is missing columns {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )
    out = df[required].copy()
    out[ID_COL] = out[ID_COL].astype("string").str.strip()
    out["actual_pIC50"] = pd.to_numeric(out[TRUE_COL], errors="coerce")
    out[f"{model_name}_pred"] = pd.to_numeric(out[PRED_COL], errors="coerce")
    out = out.drop(columns=[TRUE_COL, PRED_COL])
    if out[ID_COL].isna().any() or (out[ID_COL] == "").any():
        raise ValueError(f"{model_name} predictions contain missing/empty InChIKeys.")
    if out[ID_COL].duplicated().any():
        duplicates = out.loc[out[ID_COL].duplicated(), ID_COL].head(5).tolist()
        raise ValueError(f"{model_name} predictions contain duplicate InChIKeys: {duplicates}")
    return out


def align_split(split: str, output_dirs: dict[str, Path]) -> pd.DataFrame:
    frames = {
        name: read_model_predictions(
            output_dirs[name] / PREDICTION_FILES[split], name
        )
        for name in MODEL_NAMES
    }

    # Inner join ensures only compounds with predictions from all three models
    # are evaluated, and the join is based on compound identity, not row order.
    merged = frames["ChemBERTa"]
    for name in ["Chemprop", "XGBoost"]:
        other = frames[name].drop(columns=["actual_pIC50"])
        merged = merged.merge(other, on=ID_COL, how="inner", validate="one_to_one")

    # Check targets agree for every compound that appears in the aligned set.
    for name in ["Chemprop", "XGBoost"]:
        check = merged[[ID_COL, "actual_pIC50"]].merge(
            frames[name][[ID_COL, "actual_pIC50"]].rename(
                columns={"actual_pIC50": f"actual_{name}"}
            ),
            on=ID_COL,
            how="left",
            validate="one_to_one",
        )
        if not np.allclose(
            check["actual_pIC50"].to_numpy(dtype=float),
            check[f"actual_{name}"].to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        ):
            raise ValueError(
                f"{split}: true pIC50 values differ between ChemBERTa and {name}."
            )

    numeric_cols = ["actual_pIC50"] + [f"{name}_pred" for name in MODEL_NAMES]
    merged[numeric_cols] = merged[numeric_cols].replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=numeric_cols).reset_index(drop=True)
    if merged.empty:
        raise ValueError(
            f"No overlapping valid compounds found for {split}. "
            "Check InChIKeys and prediction outputs."
        )
    return merged


def optimize_weights(y_true: np.ndarray, prediction_matrix: np.ndarray) -> np.ndarray:
    n_models = prediction_matrix.shape[1]
    initial = np.full(n_models, 1.0 / n_models)

    def objective(weights: np.ndarray) -> float:
        pred = prediction_matrix @ weights
        return float(mean_squared_error(y_true, pred))

    result = minimize(
        objective,
        x0=initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Validation weight optimization failed: {result.message}")
    weights = np.clip(result.x, 0.0, 1.0)
    weights /= weights.sum()
    return weights


def evaluate_split(df: pd.DataFrame, split: str, weights: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_true = df["actual_pIC50"].to_numpy(dtype=float)
    pred_cols = [f"{name}_pred" for name in MODEL_NAMES]
    matrix = df[pred_cols].to_numpy(dtype=float)

    equal_weights = np.full(len(MODEL_NAMES), 1.0 / len(MODEL_NAMES))
    predictions = {
        name: matrix[:, i] for i, name in enumerate(MODEL_NAMES)
    }
    predictions["Equal-weight ensemble"] = matrix @ equal_weights
    predictions["Validation-weighted ensemble"] = matrix @ weights

    rows = [evaluate(y_true, pred, name, split) for name, pred in predictions.items()]
    results = pd.DataFrame(rows)

    pred_output = df.copy()
    pred_output["equal_weight_ensemble_pred"] = predictions["Equal-weight ensemble"]
    pred_output["weighted_ensemble_pred"] = predictions["Validation-weighted ensemble"]
    pred_output["ensemble_residual"] = (
        pred_output["actual_pIC50"] - pred_output["weighted_ensemble_pred"]
    )
    return results, pred_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine regression predictions from ChemBERTa, Chemprop and XGBoost."
    )
    parser.add_argument("--chemberta-dir", type=Path, default=Path("results/chemberta_regression"))
    parser.add_argument("--chemprop-dir", type=Path, default=Path("results/chemprop_regression"))
    parser.add_argument("--xgboost-dir", type=Path, default=Path("results/xgboost_regression"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/regression_ensemble"))
    args = parser.parse_args()

    output_dirs = {
        "ChemBERTa": args.chemberta_dir,
        "Chemprop": args.chemprop_dir,
        "XGBoost": args.xgboost_dir,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    val_df = align_split("val", output_dirs)
    test_df = align_split("test", output_dirs)

    val_y = val_df["actual_pIC50"].to_numpy(dtype=float)
    val_matrix = val_df[[f"{name}_pred" for name in MODEL_NAMES]].to_numpy(dtype=float)
    weights = optimize_weights(val_y, val_matrix)

    print(f"Aligned validation compounds: {len(val_df)}")
    print(f"Aligned test compounds:       {len(test_df)}")
    print("\\nValidation-optimized weights (non-negative, sum to 1):")
    for name, weight in zip(MODEL_NAMES, weights):
        print(f"  {name}: {weight:.6f}")
    print(f"  Sum: {weights.sum():.6f}")

    val_metrics, val_predictions = evaluate_split(val_df, "Validation", weights)
    test_metrics, test_predictions = evaluate_split(test_df, "Test", weights)
    metrics = pd.concat([val_metrics, test_metrics], ignore_index=True)

    metrics.to_csv(args.output_dir / "regression_ensemble_metrics.csv", index=False)
    val_predictions.to_csv(args.output_dir / "regression_ensemble_val_predictions.csv", index=False)
    test_predictions.to_csv(args.output_dir / "regression_ensemble_test_predictions.csv", index=False)

    config = {
        "models": MODEL_NAMES,
        "weights": {name: float(weight) for name, weight in zip(MODEL_NAMES, weights)},
        "weight_selection": "minimize validation MSE with non-negative weights summing to 1",
        "validation_prediction_file": PREDICTION_FILES["val"],
        "test_prediction_file": PREDICTION_FILES["test"],
        "id_column": ID_COL,
        "target_column": TRUE_COL,
        "prediction_column": PRED_COL,
        "output_directories": {name: str(path) for name, path in output_dirs.items()},
        "aligned_validation_compounds": int(len(val_df)),
        "aligned_test_compounds": int(len(test_df)),
    }
    with (args.output_dir / "regression_ensemble_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\\nValidation metrics:")
    print(val_metrics.sort_values("RMSE").to_string(index=False))
    print("\\nHeld-out test metrics (weights fixed from validation):")
    print(test_metrics.sort_values("RMSE").to_string(index=False))
    print(f"\\nSaved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
