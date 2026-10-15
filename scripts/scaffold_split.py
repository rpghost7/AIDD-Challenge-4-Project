#!/usr/bin/env python3
"""Create reproducible Bemis–Murcko scaffold splits for the hERG dataset.

The split strategy mirrors the original ChemBERTa notebook:
  1. Remove rows missing `smiles` or `hERG_blocker`.
  2. Remove invalid SMILES that cannot be parsed by RDKit.
  3. Group molecules by achiral Bemis–Murcko scaffold.
  4. Hold out 20% of scaffold groups for test.
  5. Hold out 10% of the remaining scaffold groups for validation.

Note: GroupShuffleSplit keeps scaffold groups separate, but split sizes are
approximate because scaffold groups can contain different numbers of rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit


def get_scaffold(smiles: object) -> str | None:
    """Return the achiral Bemis–Murcko scaffold for a SMILES string."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False,
        )
    except Exception:
        return None


def print_class_distribution(name: str, split_df: pd.DataFrame, target: str) -> None:
    """Print class counts and proportions for a split."""
    print(f"\n{name} class distribution:")
    counts = split_df[target].value_counts(dropna=False).sort_index()
    proportions = split_df[target].value_counts(normalize=True, dropna=False).sort_index()
    summary = pd.DataFrame({"count": counts, "proportion": proportions})
    print(summary.to_string())


def make_scaffold_splits(
    input_csv: Path,
    output_dir: Path,
    smiles_column: str = "smiles",
    target_column: str = "hERG_blocker",
    test_size: float = 0.20,
    validation_size: float = 0.10,
    random_state: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create and save scaffold-disjoint train/validation/test CSV files."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required = {smiles_column, target_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required column(s): {', '.join(sorted(missing))}. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )

    initial_rows = len(df)
    missing_rows = df[[smiles_column, target_column]].isna().any(axis=1).sum()
    df = df.dropna(subset=[smiles_column, target_column]).copy()
    print(f"Input rows: {initial_rows}")
    print(f"Rows removed for missing SMILES/labels: {missing_rows}")

    # Validate labels instead of silently truncating non-binary values via astype(int).
    numeric_labels = pd.to_numeric(df[target_column], errors="coerce")
    invalid_label_mask = numeric_labels.isna() | ~numeric_labels.isin([0, 1])
    invalid_label_count = int(invalid_label_mask.sum())
    if invalid_label_count:
        print(f"Rows removed for non-binary/invalid labels: {invalid_label_count}")
        df = df.loc[~invalid_label_mask].copy()
        numeric_labels = numeric_labels.loc[~invalid_label_mask]
    df[target_column] = numeric_labels.astype(int)

    df["_scaffold"] = df[smiles_column].apply(get_scaffold)
    invalid_smiles_count = int(df["_scaffold"].isna().sum())
    print(f"Invalid SMILES removed: {invalid_smiles_count}")
    df = df.dropna(subset=["_scaffold"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows remain after cleaning the input dataset.")
    if df["_scaffold"].nunique() < 3:
        raise ValueError(
            "At least three distinct scaffolds are needed to create train, validation, and test splits."
        )

    groups = df["_scaffold"].to_numpy()
    indices = np.arange(len(df))

    # First split: 80% train+validation, 20% test (approximately).
    test_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    trainval_idx, test_idx = next(
        test_splitter.split(indices, groups=groups)
    )

    # Second split: 10% of the remaining train+validation portion for validation.
    val_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_size,
        random_state=random_state,
    )
    train_rel_idx, val_rel_idx = next(
        val_splitter.split(
            trainval_idx,
            groups=groups[trainval_idx],
        )
    )

    train_df = df.iloc[trainval_idx[train_rel_idx]].copy()
    val_df = df.iloc[trainval_idx[val_rel_idx]].copy()
    test_df = df.iloc[test_idx].copy()

    # Ensure no scaffold is shared across splits before saving.
    train_scaffolds = set(train_df["_scaffold"])
    val_scaffolds = set(val_df["_scaffold"])
    test_scaffolds = set(test_df["_scaffold"])
    overlaps = {
        "Train ∩ Validation": train_scaffolds & val_scaffolds,
        "Train ∩ Test": train_scaffolds & test_scaffolds,
        "Validation ∩ Test": val_scaffolds & test_scaffolds,
    }
    print("\nScaffold overlap check:")
    for name, overlap in overlaps.items():
        print(f"{name}: {len(overlap)}")
    if any(overlaps.values()):
        raise RuntimeError("Scaffold leakage detected; refusing to save splits.")

    # The scaffold is a split-only helper and is not included in the output CSVs.
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train.csv": train_df.drop(columns=["_scaffold"]),
        "val.csv": val_df.drop(columns=["_scaffold"]),
        "test.csv": test_df.drop(columns=["_scaffold"]),
    }
    for filename, split_df in splits.items():
        destination = output_dir / filename
        split_df.to_csv(destination, index=False)
        print(f"Saved {destination} ({len(split_df):,} rows)")

    print("\nFinal split sizes:")
    total = len(df)
    for name, split_df in (("Train", splits["train.csv"]), ("Validation", splits["val.csv"]), ("Test", splits["test.csv"])):
        print(f"{name:10s}: {len(split_df):,} rows ({len(split_df) / total:.2%})")
    print(f"Total valid rows: {total:,}")
    print(f"Distinct scaffolds: {df['_scaffold'].nunique():,}")

    for name, split_df in (("Train", splits["train.csv"]), ("Validation", splits["val.csv"]), ("Test", splits["test.csv"])):
        print_class_distribution(name, split_df, target_column)

    print("\nScaffold split completed successfully.")
    return splits["train.csv"], splits["val.csv"], splits["test.csv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible scaffold-disjoint train/validation/test CSV files."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("merged_dataset.csv"),
        help="Path to merged_dataset.csv (default: ./merged_dataset.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/scaffoldsplit"),
        help="Directory for train.csv, val.csv, and test.csv.",
    )
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--target-column", default="hERG_blocker")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.10,
                        help="Fraction of the train+validation portion held out for validation.")
    parser.add_argument("--random-state", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_scaffold_splits(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        smiles_column=args.smiles_column,
        target_column=args.target_column,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
