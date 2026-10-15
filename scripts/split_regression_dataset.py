#!/usr/bin/env python3
"""Create regression train/validation/test splits aligned to classification splits.

Each regression split contains only compounds whose InChIKeys occur in the
corresponding classification scaffold split. This keeps regression evaluation
aligned with the same train/validation/test molecule allocation used by the
classification models and avoids moving compounds between splits.

Expected classification split files (created by scaffold_split.py):
    data/scaffoldsplit/train.csv
    data/scaffoldsplit/val.csv
    data/scaffoldsplit/test.csv

Expected regression dataset (created by create_regression_dataset.py):
    herg_regression_dataset.csv

Example:
    python split_regression_dataset.py \
        --regression-csv herg_regression_dataset.csv \
        --classification-split-dir data/scaffoldsplit \
        --output-dir data/regression_split
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_REGRESSION_COLUMNS = {"inchi_key", "smiles", "median_pIC50"}
REQUIRED_CLASSIFICATION_COLUMNS = {"inchi_key", "smiles", "hERG_blocker"}


def normalize_inchi_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize InChIKeys for reliable joins without changing other columns."""
    df = df.copy()
    df["inchi_key"] = df["inchi_key"].astype("string").str.strip()
    df.loc[df["inchi_key"].eq(""), "inchi_key"] = pd.NA
    return df


def require_columns(df: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def load_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")
    df = pd.read_csv(path)
    require_columns(df, required, path)
    return normalize_inchi_keys(df)


def create_regression_split(
    classification_split: pd.DataFrame,
    regression_data: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Keep regression-labelled compounds belonging to one classification split."""
    classification_keys = classification_split[["inchi_key"]].dropna().drop_duplicates()

    matched = classification_keys.merge(
        regression_data[["inchi_key", "smiles", "median_pIC50"]],
        on="inchi_key",
        how="inner",
        validate="one_to_one",
    )

    # Guard against any unexpected invalid targets or empty identifiers.
    matched["median_pIC50"] = pd.to_numeric(
        matched["median_pIC50"], errors="coerce"
    )
    matched = matched.dropna(subset=["inchi_key", "smiles", "median_pIC50"])
    matched = matched.reset_index(drop=True)

    if not matched["inchi_key"].is_unique:
        raise AssertionError(f"{split_name}: duplicate InChIKeys in regression split.")

    return matched


def split_regression_dataset(
    regression_csv: Path,
    classification_split_dir: Path,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Build aligned regression splits, verify them, and save the CSVs."""
    regression_data = load_csv(regression_csv, REQUIRED_REGRESSION_COLUMNS)
    regression_data = regression_data.dropna(
        subset=["inchi_key", "smiles", "median_pIC50"]
    ).copy()
    regression_data["median_pIC50"] = pd.to_numeric(
        regression_data["median_pIC50"], errors="coerce"
    )
    regression_data = regression_data.dropna(subset=["median_pIC50"])

    if not regression_data["inchi_key"].is_unique:
        duplicates = int(regression_data["inchi_key"].duplicated().sum())
        raise ValueError(
            f"Regression dataset must contain one row per InChIKey; found "
            f"{duplicates} duplicate rows. Recreate/deduplicate the regression dataset first."
        )

    classification_files = {
        "train": classification_split_dir / "train.csv",
        "val": classification_split_dir / "val.csv",
        "test": classification_split_dir / "test.csv",
    }
    classification_splits = {
        name: load_csv(path, REQUIRED_CLASSIFICATION_COLUMNS)
        for name, path in classification_files.items()
    }

    # Check the classification split files themselves do not assign a compound
    # to multiple partitions. InChIKey is the cross-task compound identifier.
    key_sets = {
        name: set(df["inchi_key"].dropna())
        for name, df in classification_splits.items()
    }
    overlap_pairs = [
        ("train", "val"), ("train", "test"), ("val", "test")
    ]
    for left, right in overlap_pairs:
        overlap = key_sets[left] & key_sets[right]
        print(f"Classification InChIKey overlap ({left} vs {right}): {len(overlap)}")
        if overlap:
            raise ValueError(
                f"Classification splits share {len(overlap)} InChIKey(s) between "
                f"{left} and {right}; refusing to create regression splits."
            )

    regression_splits = {
        name: create_regression_split(
            classification_splits[name], regression_data, name
        )
        for name in ("train", "val", "test")
    }

    # Ensure regression split keys are disjoint and contained in the matching
    # classification split.
    reg_key_sets = {
        name: set(df["inchi_key"].dropna())
        for name, df in regression_splits.items()
    }
    for name in ("train", "val", "test"):
        outside = reg_key_sets[name] - key_sets[name]
        if outside:
            raise AssertionError(
                f"{name}: {len(outside)} regression compound(s) are outside "
                "the corresponding classification split."
            )

    for left, right in overlap_pairs:
        overlap = reg_key_sets[left] & reg_key_sets[right]
        if overlap:
            raise AssertionError(
                f"Regression splits share {len(overlap)} InChIKey(s) between "
                f"{left} and {right}."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in regression_splits.items():
        output_path = output_dir / f"{name}_regression.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path} ({len(df):,} compounds)")

    print("\nRegression split summary")
    print("-" * 68)
    total = sum(len(df) for df in regression_splits.values())
    for name in ("train", "val", "test"):
        cls_count = len(key_sets[name])
        reg_count = len(reg_key_sets[name])
        missing_labels = len(key_sets[name] - reg_key_sets[name])
        print(
            f"{name.capitalize():10s} classification compounds: {cls_count:>8,} | "
            f"regression compounds: {reg_count:>8,} | "
            f"without regression labels: {missing_labels:>8,}"
        )
    print(f"Total regression compounds across splits: {total:,}")
    print("\nRegression InChIKey overlap check: 0 across all split pairs")
    print("Regression split creation completed successfully.")

    return regression_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create regression train/validation/test CSVs aligned to existing "
            "classification scaffold splits."
        )
    )
    parser.add_argument(
        "--regression-csv",
        type=Path,
        default=Path("herg_regression_dataset.csv"),
        help="Regression dataset created by create_regression_dataset.py.",
    )
    parser.add_argument(
        "--classification-split-dir",
        type=Path,
        default=Path("data/scaffoldsplit"),
        help="Directory containing train.csv, val.csv, and test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/regression_split"),
        help="Directory where regression split CSVs will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_regression_dataset(
        regression_csv=args.regression_csv,
        classification_split_dir=args.classification_split_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
