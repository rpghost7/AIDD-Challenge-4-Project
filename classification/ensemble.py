"""
Weighted ensemble for hERG blocker classification.

Loads validation/test probabilities produced by:
  - trees.py       -> results/trees/
  - chemberta.py   -> results/chemberta/
  - chemprop.py    -> results/chemprop/

Uses the same scaffold validation/test splits as the base models.
Uses a fixed, equal-weight probability average (1/3 per model), as requested.
Selects the classification threshold on validation to maximize MCC subject to
validation sensitivity >= the requested target (default 0.90). The held-out
test set is used only for final evaluation. This script contains classification
ensemble code only; regression-dataset preprocessing is intentionally excluded.

Example:
    python ensemble.py \
        --data-dir data/scaffoldsplit \
        --trees-dir results/trees \
        --chemberta-dir results/chemberta \
        --chemprop-dir results/chemprop \
        --output-dir results/ensemble
"""

from __future__ import annotations

import argparse
import json
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


def read_split(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} split not found: {path}")
    df = pd.read_csv(path)
    required = {"smiles", "hERG_blocker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if df["smiles"].isna().any() or df["hERG_blocker"].isna().any():
        raise ValueError(f"{path} contains missing SMILES or labels.")
    labels = set(pd.to_numeric(df["hERG_blocker"], errors="raise").unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{path}: hERG_blocker must contain only 0/1; got {labels}")
    return df.reset_index(drop=True)


def load_probability_file(
    path: Path,
    probability_column: str,
    split_df: pd.DataFrame,
    model_name: str,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"{model_name} predictions not found: {path}\n"
            "Run the corresponding base-model script first."
        )

    pred_df = pd.read_csv(path)
    required = {"smiles", probability_column}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns {sorted(missing)}. "
            f"Available columns: {list(pred_df.columns)}"
        )

    if len(pred_df) != len(split_df):
        raise ValueError(
            f"{model_name}: {len(pred_df)} predictions but "
            f"{len(split_df)} rows in the corresponding split."
        )

    if not pred_df["smiles"].reset_index(drop=True).equals(
        split_df["smiles"].reset_index(drop=True)
    ):
        raise ValueError(
            f"{model_name}: prediction SMILES/order does not match the split. "
            "Do not combine probabilities unless row order is identical."
        )

    probabilities = pd.to_numeric(
        pred_df[probability_column], errors="raise"
    ).to_numpy(dtype=float)

    if not np.isfinite(probabilities).all():
        raise ValueError(f"{model_name}: probabilities contain non-finite values.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{model_name}: probabilities must be within [0, 1].")

    # If prediction files include labels, ensure they match the source split.
    label_column = "y_true" if "y_true" in pred_df.columns else (
        "hERG_blocker" if "hERG_blocker" in pred_df.columns else None
    )
    if label_column is not None:
        pred_labels = pd.to_numeric(
            pred_df[label_column], errors="raise"
        ).astype(int).to_numpy()
        source_labels = split_df["hERG_blocker"].astype(int).to_numpy()
        if not np.array_equal(pred_labels, source_labels):
            raise ValueError(f"{model_name}: labels do not match the source split.")

    return probabilities


def confusion_counts(y_true: np.ndarray, predicted: np.ndarray):
    return confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()


def best_threshold_for_weights(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    target_sensitivity: float,
):
    # Include 0.0 to consider an all-positive prediction.
    thresholds = np.unique(np.concatenate(([0.0], probabilities)))
    candidates = []

    for threshold in thresholds:
        predicted = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_counts(y_true, predicted)

        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        if sensitivity + 1e-12 < target_sensitivity:
            continue

        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        candidates.append({
            "threshold": float(threshold),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "MCC": float(matthews_corrcoef(y_true, predicted)),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        })

    if not candidates:
        return None

    # Main objective: MCC. Tie-breaker: specificity, matching notebook logic.
    candidates.sort(
        key=lambda row: (row["MCC"], row["specificity"]),
        reverse=True,
    )
    return candidates[0]


def evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    weights: np.ndarray,
) -> dict:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_counts(y_true, predicted)

    return {
        "Threshold": float(threshold),
        "ChemBERTa weight": float(weights[0]),
        "Chemprop weight": float(weights[1]),
        "XGBoost weight": float(weights[2]),
        "Accuracy": float(accuracy_score(y_true, predicted)),
        "Precision": float(precision_score(y_true, predicted, zero_division=0)),
        "Sensitivity": float(recall_score(y_true, predicted, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "F1": float(f1_score(y_true, predicted, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, probabilities)),
        "PR-AUC": float(average_precision_score(y_true, probabilities)),
        "MCC": float(matthews_corrcoef(y_true, predicted)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main(args) -> None:
    data_dir = args.data_dir
    trees_dir = args.trees_dir
    chemberta_dir = args.chemberta_dir
    chemprop_dir = args.chemprop_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    val_df = read_split(data_dir / "val.csv", "validation")
    test_df = read_split(data_dir / "test.csv", "test")
    y_val = val_df["hERG_blocker"].astype(int).to_numpy()
    y_test = test_df["hERG_blocker"].astype(int).to_numpy()

    # These filenames and columns match the three standalone model scripts.
    p_cb_val = load_probability_file(
        chemberta_dir / "chemberta_val_predictions.csv",
        "chemberta_probability", val_df, "ChemBERTa validation"
    )
    p_cb_test = load_probability_file(
        chemberta_dir / "chemberta_test_predictions.csv",
        "chemberta_probability", test_df, "ChemBERTa test"
    )

    p_cp_val = load_probability_file(
        chemprop_dir / "chemprop_val_predictions.csv",
        "probability_hERG_blocker", val_df, "Chemprop validation"
    )
    p_cp_test = load_probability_file(
        chemprop_dir / "chemprop_test_predictions.csv",
        "probability_hERG_blocker", test_df, "Chemprop test"
    )

    p_xgb_val = load_probability_file(
        trees_dir / "xgboost_tuned_val_predictions.csv",
        "xgboost_probability", val_df, "XGBoost validation"
    )
    p_xgb_test = load_probability_file(
        trees_dir / "xgboost_tuned_test_predictions.csv",
        "xgboost_probability", test_df, "XGBoost test"
    )

    p_val = np.column_stack([p_cb_val, p_cp_val, p_xgb_val])
    p_test = np.column_stack([p_cb_test, p_cp_test, p_xgb_test])

    print("Prediction arrays aligned:")
    print("  Validation:", p_val.shape)
    print("  Test:", p_test.shape)
    print(f"  Target validation sensitivity: {args.sensitivity_target:.1%}")

    # Fixed equal-weight probability average: do not optimize ensemble weights.
    best_weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    ensemble_val_prob = p_val @ best_weights

    operating_point = best_threshold_for_weights(
        y_val, ensemble_val_prob, args.sensitivity_target
    )
    if operating_point is None:
        raise ValueError(
            "No threshold for the fixed equal-weight ensemble achieved the "
            f"requested validation sensitivity ({args.sensitivity_target:.1%})."
        )
    best_threshold = float(operating_point["threshold"])

    print("\nFixed equal ensemble weights:")
    print(f"  ChemBERTa weight: {best_weights[0]:.6f}")
    print(f"  Chemprop weight:  {best_weights[1]:.6f}")
    print(f"  XGBoost weight:   {best_weights[2]:.6f}")
    print(f"  Validation threshold: {best_threshold:.6f}")
    print(f"  Validation sensitivity: {operating_point['sensitivity']:.4f}")
    print(f"  Validation specificity: {operating_point['specificity']:.4f}")
    print(f"  Validation MCC: {operating_point['MCC']:.4f}")

    # Threshold and weights are now fixed; test is used only for evaluation.
    ensemble_val_prob = p_val @ best_weights
    ensemble_test_prob = p_test @ best_weights
    final_results = evaluate(y_test, ensemble_test_prob, best_threshold, best_weights)
    final_results["Model"] = "Equal-weight ensemble — max validation MCC at target sensitivity"

    pd.DataFrame([{
        "w_ChemBERTa": float(best_weights[0]),
        "w_Chemprop": float(best_weights[1]),
        "w_XGBoost": float(best_weights[2]),
        "threshold": best_threshold,
        "validation_sensitivity": float(operating_point["sensitivity"]),
        "validation_specificity": float(operating_point["specificity"]),
        "validation_MCC": float(operating_point["MCC"]),
    }]).to_csv(output_dir / "ensemble_fixed_weights.csv", index=False)
    pd.DataFrame([final_results]).to_csv(
        output_dir / "weighted_ensemble_test_metrics.csv", index=False
    )

    pd.DataFrame({
        "smiles": val_df["smiles"],
        "y_true": y_val,
        "ensemble_probability": ensemble_val_prob,
        "predicted_label_90_sensitivity_mcc": (
            ensemble_val_prob >= best_threshold
        ).astype(int),
    }).to_csv(output_dir / "weighted_ensemble_val_predictions.csv", index=False)

    pd.DataFrame({
        "smiles": test_df["smiles"],
        "y_true": y_test,
        "ensemble_probability": ensemble_test_prob,
        "predicted_label_90_sensitivity_mcc": (
            ensemble_test_prob >= best_threshold
        ).astype(int),
    }).to_csv(output_dir / "weighted_ensemble_test_predictions.csv", index=False)

    config = {
        "ensemble": "fixed equal-weight probability average",
        "base_models_order": ["ChemBERTa", "Chemprop", "XGBoost"],
        "weights": {
            "ChemBERTa": float(best_weights[0]),
            "Chemprop": float(best_weights[1]),
            "XGBoost": float(best_weights[2]),
        },
        "threshold": best_threshold,
        "target_validation_sensitivity": args.sensitivity_target,
        "validation_sensitivity": float(operating_point["sensitivity"]),
        "validation_specificity": float(operating_point["specificity"]),
        "validation_MCC": float(operating_point["MCC"]),
        "weight_selection": "fixed equal weights; no weight optimization",
        "threshold_selected_on": "validation set only",
        "test_set_used_for_weight_or_threshold_selection": False,
        "prediction_files": {
            "chemberta": {
                "validation": str(chemberta_dir / "chemberta_val_predictions.csv"),
                "test": str(chemberta_dir / "chemberta_test_predictions.csv"),
                "probability_column": "chemberta_probability",
            },
            "chemprop": {
                "validation": str(chemprop_dir / "chemprop_val_predictions.csv"),
                "test": str(chemprop_dir / "chemprop_test_predictions.csv"),
                "probability_column": "probability_hERG_blocker",
            },
            "xgboost": {
                "validation": str(trees_dir / "xgboost_tuned_val_predictions.csv"),
                "test": str(trees_dir / "xgboost_tuned_test_predictions.csv"),
                "probability_column": "xgboost_probability",
            },
        },
    }
    with open(output_dir / "weighted_ensemble_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\nHeld-out test metrics:")
    print(pd.DataFrame([final_results]).round(4).to_string(index=False))
    print(f"\nSaved ensemble artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed equal-weight hERG classifier ensemble."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/scaffoldsplit"))
    parser.add_argument("--trees-dir", type=Path, default=Path("results/trees"))
    parser.add_argument("--chemberta-dir", type=Path, default=Path("results/chemberta"))
    parser.add_argument("--chemprop-dir", type=Path, default=Path("results/chemprop"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ensemble"))
    parser.add_argument("--sensitivity-target", type=float, default=0.90)
    parsed_args = parser.parse_args()

    if not 0 < parsed_args.sensitivity_target <= 1:
        parser.error("--sensitivity-target must be in (0, 1].")

    main(parsed_args)
