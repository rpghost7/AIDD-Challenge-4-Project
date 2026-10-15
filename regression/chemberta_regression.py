#!/usr/bin/env python3
"""
Fine-tune ChemBERTa for hERG pIC50 regression.

Expected input files:
  train_regression.csv
  val_regression.csv
  test_regression.csv

Each CSV must contain: inchi_key, smiles, median_pIC50

SMILES are cleaned both defensively (matching the notebook) and validated with
RDKit when available. By default, invalid/empty SMILES and non-numeric targets
are removed with a warning. The held-out test set is used only for final
evaluation, not checkpoint selection.
"""

import argparse
import os
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

MODEL_NAME = "DeepChem/ChemBERTa-100M-MLM"
SEED = 123
BATCH_SIZE = 32
NUM_EPOCHS = 50
PATIENCE = 2  # matches the active training cell in the notebook
MAX_LENGTH = 512


def clean_smiles(s):
    """Apply the same defensive cleanup used in the notebook."""
    if pd.isna(s):
        return ""
    return str(s).split(";")[0].split(" ")[0].strip()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_and_clean_dataframe(df, split_name):
    required = {"inchi_key", "smiles", "median_pIC50"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{split_name} CSV is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    original_count = len(df)
    df["smiles"] = df["smiles"].map(clean_smiles)
    df["median_pIC50"] = pd.to_numeric(df["median_pIC50"], errors="coerce")

    valid = df["smiles"].ne("") & df["median_pIC50"].notna()
    # Remove non-finite targets too.
    valid &= np.isfinite(df["median_pIC50"].to_numpy(dtype=float))

    # RDKit validation is used if installed; the model's tokenizer is not a
    # chemical validity checker. Fail clearly if RDKit is missing? No: warn and
    # continue with the notebook's string-cleaning behavior.
    try:
        from rdkit import Chem
        rdkit_valid = df["smiles"].map(
            lambda s: Chem.MolFromSmiles(s) is not None if s else False
        )
        invalid_smiles_count = int((~rdkit_valid & valid).sum())
        valid &= rdkit_valid
    except ImportError:
        invalid_smiles_count = None
        print(
            "Warning: RDKit is not installed; SMILES will be cleaned but not "
            "chemically validated. Install rdkit for validity checks."
        )

    removed = original_count - int(valid.sum())
    if removed:
        print(
            f"{split_name}: removed {removed} rows with missing/invalid SMILES "
            "or missing/non-finite pIC50."
        )
    if invalid_smiles_count is not None and invalid_smiles_count:
        print(f"{split_name}: {invalid_smiles_count} invalid SMILES detected by RDKit.")

    df = df.loc[valid].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{split_name} has no usable rows after cleaning.")

    # Keep only the expected columns in a consistent order.
    return df[["inchi_key", "smiles", "median_pIC50"]]


class ChemBERTaRegressionDataset(Dataset):
    def __init__(self, smiles, targets):
        self.smiles = list(smiles)
        self.targets = list(map(float, targets))

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx], self.targets[idx]


class ChemBERTaRegressor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.chemberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.chemberta.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.chemberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        return self.regressor(pooled)


def make_collate_fn(tokenizer, max_length):
    def collate_fn(batch):
        smiles = [item[0] for item in batch]
        targets = torch.tensor(
            [item[1] for item in batch], dtype=torch.float32
        ).unsqueeze(1)
        inputs = tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return inputs["input_ids"], inputs["attention_mask"], targets
    return collate_fn


def evaluate_loss(model, loader, dataset_size, criterion, device, split_name):
    model.eval()
    loss_sum = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, targets in tqdm(
            loader, desc=f"{split_name} loss", leave=False
        ):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, targets)
            loss_sum += loss.item() * len(targets)
    return loss_sum / dataset_size


def predict_loader(model, loader, device):
    predictions = []
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, _ in tqdm(
            loader, desc="Predicting", leave=False
        ):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output = model(input_ids, attention_mask)
            predictions.extend(output.squeeze(1).cpu().numpy().tolist())
    return np.asarray(predictions, dtype=float)


def safe_correlation(fn, actual, predicted):
    if len(actual) < 2 or np.all(actual == actual[0]) or np.all(predicted == predicted[0]):
        return float("nan")
    try:
        result = fn(actual, predicted)
        return float(result.statistic if hasattr(result, "statistic") else result[0])
    except Exception:
        return float("nan")


def regression_metrics(results, split_name):
    actual = results["actual_pIC50"].to_numpy(dtype=float)
    predicted = results["predicted_pIC50"].to_numpy(dtype=float)
    return {
        "Split": split_name,
        "N": len(results),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "R2": float(r2_score(actual, predicted)) if len(actual) >= 2 else float("nan"),
        "Pearson_r": safe_correlation(pearsonr, actual, predicted),
        "Spearman_r": safe_correlation(spearmanr, actual, predicted),
    }


def make_prediction_frame(df, predictions):
    result = df[["inchi_key", "smiles", "median_pIC50"]].copy()
    result = result.rename(columns={"median_pIC50": "actual_pIC50"})
    result["predicted_pIC50"] = predictions
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate ChemBERTa for pIC50 regression."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/regression_split"),
        help="Directory containing train_regression.csv, val_regression.csv, test_regression.csv.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/chemberta_regression"),
        help="Directory where checkpoints, predictions, and metrics are saved.",
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()

    if args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        parser.error("--batch-size, --epochs and --patience must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Read and clean all splits. Cleaning is repeated here as a safety measure.
    split_paths = {
        "Train": args.data_dir / "train_regression.csv",
        "Validation": args.data_dir / "val_regression.csv",
        "Test": args.data_dir / "test_regression.csv",
    }
    frames = {}
    for name, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} split not found: {path}")
        frames[name] = validate_and_clean_dataframe(
            pd.read_csv(path), name
        )
        df = frames[name]
        print(
            f"{name}: {len(df)} rows; pIC50 mean={df['median_pIC50'].mean():.3f}, "
            f"std={df['median_pIC50'].std():.3f}"
        )

    # Do not move compounds across splits. Cleaning only removes unusable rows.
    train_df, val_df, test_df = (
        frames["Train"], frames["Validation"], frames["Test"]
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fn = make_collate_fn(tokenizer, args.max_length)

    datasets = {
        "Train": ChemBERTaRegressionDataset(
            train_df["smiles"], train_df["median_pIC50"]
        ),
        "Validation": ChemBERTaRegressionDataset(
            val_df["smiles"], val_df["median_pIC50"]
        ),
        "Test": ChemBERTaRegressionDataset(
            test_df["smiles"], test_df["median_pIC50"]
        ),
    }

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loaders = {
        "Train": DataLoader(
            datasets["Train"], batch_size=args.batch_size, shuffle=True,
            generator=generator, collate_fn=collate_fn
        ),
        "Validation": DataLoader(
            datasets["Validation"], batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn
        ),
        "Test": DataLoader(
            datasets["Test"], batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn
        ),
    }
    print(
        "Batches — Train: {}, Validation: {}, Test: {}".format(
            len(loaders["Train"]), len(loaders["Validation"]), len(loaders["Test"])
        )
    )

    model = ChemBERTaRegressor(args.model_name).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.chemberta.parameters(), "lr": 2e-5},
            {"params": model.regressor.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    criterion = nn.MSELoss()

    best_checkpoint = args.output_dir / f"best_checkpoint_seed{args.seed}.pt"
    latest_checkpoint = args.output_dir / f"latest_checkpoint_seed{args.seed}.pt"
    history = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        for input_ids, attention_mask, targets in tqdm(
            loaders["Train"], desc=f"Epoch {epoch + 1} training"
        ):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(targets)

        train_loss = train_loss_sum / len(datasets["Train"])
        val_loss = evaluate_loss(
            model, loaders["Validation"], len(datasets["Validation"]),
            criterion, device, "Validation"
        )
        history.append({
            "epoch": epoch + 1,
            "train_mse": train_loss,
            "validation_mse": val_loss,
        })
        print(
            f"Epoch {epoch + 1:02d} | Train MSE: {train_loss:.4f} | "
            f"Validation MSE: {val_loss:.4f}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": min(best_val_loss, val_loss),
            "seed": args.seed,
            "model_name": args.model_name,
        }
        torch.save(checkpoint, latest_checkpoint)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, best_checkpoint)
            print(f"Best checkpoint saved: {best_checkpoint}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("Early stopping.")
            break

    pd.DataFrame(history).to_csv(
        args.output_dir / "training_history.csv", index=False
    )

    # Load best validation-loss checkpoint for final predictions.
    checkpoint = torch.load(
        best_checkpoint, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_predictions = predict_loader(model, loaders["Validation"], device)
    test_predictions = predict_loader(model, loaders["Test"], device)
    val_results = make_prediction_frame(val_df, val_predictions)
    test_results = make_prediction_frame(test_df, test_predictions)

    metrics_df = pd.DataFrame([
        regression_metrics(val_results, "Validation"),
        regression_metrics(test_results, "Test"),
    ])

    print("\nRegression metrics:")
    print(metrics_df.round(4).to_string(index=False))

    val_results.to_csv(
        args.output_dir / "val_evaluated_predictions.csv", index=False
    )
    test_results.to_csv(
        args.output_dir / "test_evaluated_predictions.csv", index=False
    )
    metrics_df.to_csv(
        args.output_dir / "regression_metrics.csv", index=False
    )
    config = {
        "model_name": args.model_name,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_epochs_max": args.epochs,
        "patience": args.patience,
        "max_length": args.max_length,
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_mse": float(checkpoint["best_val_loss"]),
        "device_used": str(device),
        "smiles_cleaning": "str(s).split(';')[0].split(' ')[0].strip()",
    }
    with open(args.output_dir / "chemberta_regression_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
