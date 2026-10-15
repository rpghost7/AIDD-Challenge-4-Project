#!/usr/bin/env python3
"""
Build the hERG regression dataset by combining potency labels from:
1. herg_final_ML_dataset.csv
2. BindingDB_hERG_labeled_dataset.csv

The procedure follows the regression-dataset construction in the project notebook:
- Standardize the key, SMILES, and median_pIC50 columns.
- Remove records missing an InChIKey or SMILES.
- Combine both sources.
- Aggregate pIC50 labels by InChIKey using the median.
- Keep compounds whose pIC50 label range is <= 0.5.
- Retain one SMILES per InChIKey.

Example:
    python create_regression_dataset.py \
        --herg-csv herg_final_ML_dataset.csv \
        --bindingdb-csv BindingDB_hERG_labeled_dataset.csv \
        --output-csv herg_regression_dataset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine hERG and BindingDB potency data into a regression dataset."
    )
    parser.add_argument(
        "--herg-csv",
        type=Path,
        default=Path("herg_final_ML_dataset.csv"),
        help="CSV containing inchi_key, smiles, and median_pIC50.",
    )
    parser.add_argument(
        "--bindingdb-csv",
        type=Path,
        default=Path("BindingDB_hERG_labeled_dataset.csv"),
        help=(
            "BindingDB CSV containing Ligand InChI Key, Ligand SMILES, "
            "and median_pIC50."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("herg_regression_dataset.csv"),
        help="Path for the generated regression dataset.",
    )
    parser.add_argument(
        "--max-pic50-range",
        type=float,
        default=0.5,
        help="Maximum allowed range among pIC50 labels for a compound (default: 0.5).",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, required: list[str], source_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def standardize_dataset(
    df: pd.DataFrame,
    key_col: str,
    smiles_col: str,
    source_name: str,
) -> pd.DataFrame:
    """Select and standardize the columns used by the notebook pipeline."""
    require_columns(
        df,
        [key_col, smiles_col, "median_pIC50"],
        source_name,
    )

    standardized = df[[key_col, smiles_col, "median_pIC50"]].copy()
    standardized = standardized.rename(
        columns={key_col: "inchi_key", smiles_col: "smiles"}
    )

    standardized["inchi_key"] = standardized["inchi_key"].astype("string").str.strip()
    standardized["smiles"] = standardized["smiles"].astype("string").str.strip()
    standardized["median_pIC50"] = pd.to_numeric(
        standardized["median_pIC50"], errors="coerce"
    )

    standardized.loc[standardized["inchi_key"].eq(""), "inchi_key"] = pd.NA
    standardized.loc[standardized["smiles"].eq(""), "smiles"] = pd.NA

    print(f"\n{source_name} missing values before dropping:")
    print(standardized.isna().sum())

    standardized = standardized.dropna(subset=["inchi_key", "smiles"]).copy()
    print(f"{source_name} rows after removing missing InChIKeys/SMILES: {len(standardized)}")
    return standardized


def build_regression_dataset(
    herg_csv: Path,
    bindingdb_csv: Path,
    max_pic50_range: float = 0.5,
) -> pd.DataFrame:
    """Combine sources and retain compounds with consistent potency labels."""
    if not herg_csv.exists():
        raise FileNotFoundError(f"Could not find hERG dataset: {herg_csv}")
    if not bindingdb_csv.exists():
        raise FileNotFoundError(f"Could not find BindingDB dataset: {bindingdb_csv}")
    if max_pic50_range < 0:
        raise ValueError("--max-pic50-range must be non-negative.")

    first_df = pd.read_csv(herg_csv)
    binding_df = pd.read_csv(bindingdb_csv)

    print("Original hERG dataset shape:", first_df.shape)
    print("Original BindingDB dataset shape:", binding_df.shape)

    herg = standardize_dataset(
        first_df, "inchi_key", "smiles", "hERG dataset"
    )
    binding = standardize_dataset(
        binding_df, "Ligand InChI Key", "Ligand SMILES", "BindingDB dataset"
    )

    # Combine the two sources, retaining all records for label reconciliation.
    combined = pd.concat([herg, binding], ignore_index=True)
    print("\nCombined rows before deduplication:", len(combined))
    print(
        "Unique InChIKeys before deduplication:",
        combined["inchi_key"].nunique(),
    )

    duplicate_mask = combined["inchi_key"].duplicated(keep=False)
    duplicate_records = combined.loc[duplicate_mask].sort_values("inchi_key")
    print("\nRows belonging to duplicated InChIKeys:", len(duplicate_records))
    print(
        "Number of duplicated InChIKeys:",
        duplicate_records["inchi_key"].nunique(),
    )

    # Reconcile potency labels for each compound. Rows with missing/non-numeric
    # pIC50 values do not contribute to the label summary.
    labelled = combined.dropna(subset=["median_pIC50"]).copy()
    potency_summary = (
        labelled.groupby("inchi_key")
        .agg(
            median_pIC50=("median_pIC50", "median"),
            min_pIC50=("median_pIC50", "min"),
            max_pIC50=("median_pIC50", "max"),
            n_labels=("median_pIC50", "count"),
        )
        .reset_index()
    )
    potency_summary["pIC50_range"] = (
        potency_summary["max_pIC50"] - potency_summary["min_pIC50"]
    )

    conflicts = potency_summary[
        potency_summary["pIC50_range"] > max_pic50_range
    ]
    print(
        f"\nCompounds with pIC50 differences greater than {max_pic50_range}:",
        len(conflicts),
    )

    accepted_keys = potency_summary.loc[
        potency_summary["pIC50_range"] <= max_pic50_range,
        "inchi_key",
    ]

    # Keep one SMILES per InChIKey, matching the notebook's first-occurrence
    # behavior, then attach the aggregated potency label.
    smiles_lookup = (
        combined.dropna(subset=["inchi_key", "smiles"])
        .drop_duplicates(subset=["inchi_key"])
        [["inchi_key", "smiles"]]
    )

    regression_df = smiles_lookup.merge(
        potency_summary[["inchi_key", "median_pIC50", "pIC50_range"]],
        on="inchi_key",
        how="inner",
        validate="one_to_one",
    )

    regression_df = (
        regression_df[
            regression_df["inchi_key"].isin(accepted_keys)
        ]
        .drop(columns=["pIC50_range"])
        .reset_index(drop=True)
    )

    # The final output should have one row per compound and a usable target.
    regression_df = regression_df.dropna(
        subset=["inchi_key", "smiles", "median_pIC50"]
    ).copy()
    if not regression_df["inchi_key"].is_unique:
        raise AssertionError("Output contains duplicate InChIKeys.")

    print("\nFinal regression dataset shape:", regression_df.shape)
    print("Unique compounds:", regression_df["inchi_key"].nunique())
    print("Missing values in output:")
    print(regression_df.isna().sum())
    print("\nPreview:")
    print(regression_df.head().to_string(index=False))

    return regression_df


def main() -> None:
    args = parse_args()
    regression_df = build_regression_dataset(
        args.herg_csv,
        args.bindingdb_csv,
        max_pic50_range=args.max_pic50_range,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    regression_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved regression dataset to: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
