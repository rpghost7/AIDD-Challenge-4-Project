"""
hERG blocker classification using fine-tuned ChemBERTa.

Expected input files:
    <data_dir>/train.csv
    <data_dir>/val.csv
    <data_dir>/test.csv

Required columns:
    smiles, hERG_blocker

Pipeline follows the ChemBERTa notebook:
    - DeepChem/ChemBERTa-100M-MLM encoder
    - Mean pooling over non-padding tokens
    - MLP classification head: hidden_size -> 512 -> 256 -> 64 -> 1
    - BCEWithLogitsLoss with positive-class weighting
    - AdamW: encoder LR 2e-5, classifier-head LR 1e-3
    - Early stopping on validation loss (patience 7; maximum 50 epochs)
    - Threshold selection on validation data only:
        1. Maximum MCC
        2. Maximum MCC subject to sensitivity >= 90%
    - Final evaluation on the held-out test set using both fixed thresholds.

Example:
    python chemberta.py --data-dir data/scaffoldsplit --output-dir results/chemberta
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
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


RANDOM_STATE = 123
MODEL_NAME = "DeepChem/ChemBERTa-100M-MLM"
TARGET_SENSITIVITY = 0.90
BATCH_SIZE = 32
MAX_LENGTH = 512
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7
ENCODER_LEARNING_RATE = 2e-5
CLASSIFIER_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_splits(data_dir: Path) -> dict[str, pd.DataFrame]:
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
        if len(labels) < 2:
            print(f"Warning: {name} split contains only one class.")

        splits[name] = df.reset_index(drop=True)
        print(
            f"{name.capitalize():>10}: {df.shape}; "
            f"labels={df['hERG_blocker'].value_counts().to_dict()}"
        )
    return splits


class HERGDataset(Dataset):
    def __init__(self, smiles: list[str], labels: np.ndarray):
        self.smiles = [str(s) for s in smiles]
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.smiles[idx], float(self.labels[idx])


class ChemBERTaHERGClassifier(nn.Module):
    """ChemBERTa encoder with masked mean pooling and an MLP head."""

    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        self.chemberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.chemberta.config.hidden_size
        self.classifier = nn.Sequential(
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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.chemberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        masked_hidden = hidden_states * mask
        summed = masked_hidden.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts
        return self.classifier(pooled)


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        smiles = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.float32).unsqueeze(1)
        tokens = tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        return tokens["input_ids"], tokens["attention_mask"], labels
    return collate_fn


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    """Run one training or validation epoch; optimizer=None means evaluation."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for input_ids, attention_mask, labels in tqdm(loader, desc=desc, leave=False):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * input_ids.size(0)

    return total_loss / len(loader.dataset)


def predict_probabilities(model, loader, device):
    """Return probabilities and labels in loader order."""
    model.eval()
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(loader, desc="Predicting", leave=False):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids, attention_mask).squeeze(-1)
            probabilities = torch.sigmoid(logits)
            all_probabilities.extend(probabilities.cpu().numpy().tolist())
            all_labels.extend(labels.view(-1).cpu().numpy().astype(int).tolist())

    return np.asarray(all_probabilities), np.asarray(all_labels, dtype=int)


def choose_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    target_sensitivity: float = TARGET_SENSITIVITY,
) -> tuple[dict, dict | None, pd.DataFrame]:
    """Select maximum-MCC and sensitivity-constrained thresholds on validation data."""
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    rows = []

    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append({
            "threshold": float(threshold),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "MCC": float(matthews_corrcoef(y_true, predictions)),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        })

    threshold_df = pd.DataFrame(rows)
    best_mcc = threshold_df.sort_values(
        ["MCC", "specificity", "threshold"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()

    eligible = threshold_df[threshold_df["sensitivity"] >= target_sensitivity]
    best_high_sensitivity = None
    if not eligible.empty:
        best_high_sensitivity = eligible.sort_values(
            ["MCC", "specificity", "threshold"],
            ascending=[False, False, False],
        ).iloc[0].to_dict()

    return best_mcc, best_high_sensitivity, threshold_df


def evaluate_model(strategy, y_true, probabilities, threshold) -> dict:
    """Calculate test metrics at a threshold selected on validation data."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
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
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def main(data_dir: Path, output_dir: Path, batch_size: int, max_epochs: int, patience: int):
    set_seed()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(data_dir)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    y_train = train_df["hERG_blocker"].astype(int).to_numpy()
    y_val = val_df["hERG_blocker"].astype(int).to_numpy()
    y_test = test_df["hERG_blocker"].astype(int).to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate_fn = make_collate_fn(tokenizer)

    train_dataset = HERGDataset(train_df["smiles"].tolist(), y_train)
    val_dataset = HERGDataset(val_df["smiles"].tolist(), y_val)
    test_dataset = HERGDataset(test_df["smiles"].tolist(), y_test)

    generator = torch.Generator()
    generator.manual_seed(RANDOM_STATE)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        generator=generator, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
    )

    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("Both classes must be present in the training split.")
    positive_weight = negatives / positives
    print(f"Training positives: {positives}; negatives: {negatives}")
    print(f"Positive class weight: {positive_weight:.4f}")

    model = ChemBERTaHERGClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.chemberta.parameters(), "lr": ENCODER_LEARNING_RATE},
            {"params": model.classifier.parameters(), "lr": CLASSIFIER_LEARNING_RATE},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_checkpoint_path = output_dir / "chemberta_best_checkpoint.pt"
    latest_checkpoint_path = output_dir / "chemberta_latest_checkpoint.pt"
    history = []

    print("\nStarting fine-tuning...")
    for epoch in range(max_epochs):
        train_loss = run_epoch(
            model, train_loader, criterion, device, optimizer,
            desc=f"Epoch {epoch + 1}/{max_epochs} training",
        )
        val_loss = run_epoch(
            model, val_loader, criterion, device, optimizer=None,
            desc=f"Epoch {epoch + 1}/{max_epochs} validation",
        )
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "seed": RANDOM_STATE,
                "model_name": MODEL_NAME,
            }, best_checkpoint_path)
        else:
            patience_counter += 1

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
            "seed": RANDOM_STATE,
            "model_name": MODEL_NAME,
        }, latest_checkpoint_path)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
        })
        print(
            f"Epoch {epoch + 1:02d}/{max_epochs} | "
            f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
            f"{'best checkpoint saved' if improved else f'patience {patience_counter}/{patience}'}"
        )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    pd.DataFrame(history).to_csv(output_dir / "chemberta_training_history.csv", index=False)

    # Restore the best checkpoint selected by validation loss.
    try:
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        f"\nRestored best checkpoint from epoch {checkpoint['epoch']} "
        f"(validation loss={checkpoint['best_val_loss']:.4f})."
    )

    val_probabilities, val_labels = predict_probabilities(model, val_loader, device)
    best_mcc, best_90, threshold_df = choose_thresholds(val_labels, val_probabilities)
    threshold_df.to_csv(output_dir / "chemberta_validation_thresholds.csv", index=False)

    print("\nValidation threshold — maximum MCC:")
    print(pd.Series(best_mcc).to_string())
    print("\nValidation threshold — maximum MCC with sensitivity >= 90%:")
    if best_90 is None:
        raise RuntimeError(
            "No threshold achieved >=90% sensitivity on validation data. "
            "Inspect chemberta_validation_thresholds.csv."
        )
    print(pd.Series(best_90).to_string())

    tokenizer.save_pretrained(output_dir / "tokenizer")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": MODEL_NAME,
        "model_class": "ChemBERTaHERGClassifier",
        "max_length": MAX_LENGTH,
        "random_state": RANDOM_STATE,
    }, output_dir / "chemberta_model.pt")

    with open(output_dir / "chemberta_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "random_state": RANDOM_STATE,
            "batch_size": batch_size,
            "max_length": MAX_LENGTH,
            "max_epochs": max_epochs,
            "early_stopping_patience": patience,
            "encoder_learning_rate": ENCODER_LEARNING_RATE,
            "classifier_learning_rate": CLASSIFIER_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "positive_class_weight": positive_weight,
            "best_epoch": int(checkpoint["epoch"]),
            "best_validation_loss": float(checkpoint["best_val_loss"]),
            "target_sensitivity": TARGET_SENSITIVITY,
            "validation_thresholds": {
                "maximum_MCC": best_mcc,
                "maximum_MCC_at_target_sensitivity": best_90,
            },
        }, f, indent=2)

    # Test set is used only after model and thresholds have been selected.
    test_probabilities, test_labels = predict_probabilities(model, test_loader, device)
    test_metrics = pd.DataFrame([
        evaluate_model(
            "ChemBERTa — max MCC",
            test_labels,
            test_probabilities,
            best_mcc["threshold"],
        ),
        evaluate_model(
            "ChemBERTa — max MCC at >=90% validation sensitivity",
            test_labels,
            test_probabilities,
            best_90["threshold"],
        ),
    ])
    test_metrics.to_csv(output_dir / "chemberta_test_metrics.csv", index=False)

    pd.DataFrame({
        "smiles": val_df["smiles"].to_numpy(),
        "y_true": y_val,
        "chemberta_probability": val_probabilities,
    }).to_csv(output_dir / "chemberta_val_predictions.csv", index=False)

    pd.DataFrame({
        "smiles": test_df["smiles"].to_numpy(),
        "y_true": y_test,
        "chemberta_probability": test_probabilities,
    }).to_csv(output_dir / "chemberta_test_predictions.csv", index=False)

    print("\nHeld-out test metrics:")
    print(test_metrics.round(4).to_string(index=False))
    print(f"\nAll outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune ChemBERTa for hERG classification."
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
        default=Path("results/chemberta"),
        help="Directory for model checkpoints, predictions and metrics.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.batch_size, args.max_epochs, args.patience)
