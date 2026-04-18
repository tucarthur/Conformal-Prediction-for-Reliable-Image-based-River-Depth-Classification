"""
train.py
========
Fine-tuning of pre-trained image classification backbones for a 4-class dataset.

This script:
  1. Builds a stratified train / validation / test split from raw image folders.
  2. Defines a reproducible ``Image_Dataset`` and data augmentation pipeline tailored
     for low-light imagery.
  3. Wraps five ImageNet pre-trained backbones (ResNet-50, EfficientNet-V2-L,
     MobileNet-V3-Large, ViT-B/16, Swin-V2-B) under a unified ``Networks`` module.
  4. Runs a stratified k-fold cross-validation grid search to select the best
     hyper-parameters per backbone (``KFoldTrainer``).
  5. Re-trains each backbone on the full training set using the selected
     hyper-parameters, evaluates on the hold-out test set, and saves:
       - Best model checkpoint  → ``<backbone>_best.pth``
       - Per-sample predictions → ``predictions_calib_<backbone>.csv``
       - Summary table          → ``model_results_summary.csv``

Label encoding
--------------
  low → 0 | medium → 1 | high → 2 | (other) → 3

Usage
-----
    python train.py

Configure all paths and hyper-parameters in the ``Configuration`` section below.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    EfficientNet_V2_L_Weights,
    MobileNet_V3_Large_Weights,
    ResNet50_Weights,
    Swin_V2_B_Weights,
    ViT_B_16_Weights,
    efficientnet_v2_l,
    mobilenet_v3_large,
    resnet50,
    swin_v2_b,
    vit_b_16,
)
from tqdm import tqdm

# Required for reproducibility with CUDA >= 10.2
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# =============================================================================
# Configuration
# =============================================================================

# Directories containing the raw images.
TRAIN_FOLDER: Path = Path("/path/to/train")
TEST_FOLDER: Path = Path("/path/to/test")

# Output directory for checkpoints, predictions, and the summary CSV.
OUTPUT_DIR: Path = Path("/path/to/outputs")

# Dataset split ratios (train / val / test = 60 / 20 / 20).
VAL_SIZE: float = 0.4   # fraction held out from full data for val + test
TEST_SIZE: float = 0.5  # fraction of the above held out as the test set

# Global random seed for reproducibility.
SEED: int = 42

# Number of classes.
NUM_CLASSES: int = 4

# Number of DataLoader worker processes.
NUM_WORKERS: int = 4

# Image resolution fed to all backbones.
IMAGE_SIZE: int = 224

# Per-backbone hyper-parameters selected after k-fold cross-validation.
# Edit these values to reflect the best configuration found for your dataset.
BACKBONE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "efficientnet_v2_l":  {"lr": 0.0001, "epochs": 10,  "batch_size": 16},
    "resnet50":           {"lr": 0.0001, "epochs": 50,  "batch_size": 16},
    "mobilenet_v3_large": {"lr": 0.001,  "epochs": 50,  "batch_size": 32},
    "vision_transformer": {"lr": 0.0001, "epochs": 50,  "batch_size": 32},
    "swin_transformer":   {"lr": 0.0001, "epochs": 50,  "batch_size": 16},
}

# Hyper-parameter search space used during k-fold grid search.
KFOLD_HYPERPARAMS: Dict[str, list] = {
    "lr":         [0.01,0.001,0.0001],
    "optimizer":  ["adam", "adamw"],
    "epochs":     [5,10,50],
    "batch_size": [8,16,32],
}

# Backbones to include in the grid search.
KFOLD_BACKBONES: list[str] = [
    "efficientnet_v2_l",
    "resnet50",
    "mobilenet_v3_large",
    "vision_transformer",
    "swin_transformer",
]

# Number of folds for cross-validation.
N_FOLDS: int = 5

# =============================================================================
# Reproducibility
# =============================================================================

def set_all_seeds(seed: int = SEED) -> None:
    """Set all relevant random seeds to ensure reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Per-worker seed initialiser for ``DataLoader``."""
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# =============================================================================
# Data loading and splitting
# =============================================================================

def make_dataframe(folder_path: Path) -> pd.DataFrame:
    """
    Build a DataFrame from a flat folder of JPEG images.

    Assumes the file-name format ``<label>_<rest>.jpg`` where the label is the
    part before the first underscore.
    """
    samples = []
    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            filename = os.path.join(folder_path, file)
            label = file.split("_")[0]
            samples.append([filename, label])
    return pd.DataFrame(samples, columns=["path", "label"])


def build_splits(
    train_folder: Path,
    test_folder: Path,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merge the original train and test folders, then re-split into
    stratified train / validation / test subsets.

    Returns
    -------
    train_df, val_df, test_df
    """
    train_df = make_dataframe(train_folder)
    test_df = make_dataframe(test_folder)

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    train_df, temp_df = train_test_split(
        full_df, test_size=val_size, stratify=full_df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=test_size, stratify=temp_df["label"], random_state=seed
    )

    print(
        f"Split sizes — train: {len(train_df)}, "
        f"val: {len(val_df)}, test: {len(test_df)}"
    )
    return train_df, val_df, test_df


# =============================================================================
# Dataset and transforms
# =============================================================================

_LABEL_MAP: Dict[str, int] = {"low": 0, "medium": 1, "high": 2}


class ImageDataset(Dataset):
    """PyTorch Dataset that reads images from a DataFrame of ``(path, label)`` rows."""

    def __init__(self, dataframe: pd.DataFrame, transform=None) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        image_path = self.dataframe.iloc[idx]["path"]
        image = PIL.Image.open(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        raw_label = self.dataframe.iloc[idx]["label"]
        label = _LABEL_MAP.get(raw_label, 3)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(image_size: int = IMAGE_SIZE):
    """
    Return ``(train_transforms, eval_transforms)`` for ImageNet pre-trained models.

    Training augmentation is designed for low-light / nighttime imagery and
    includes random rotation, colour jitter, random gamma, Gaussian blur,
    and additive Gaussian noise.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.9, contrast=0.8, saturation=0.8, hue=0.4),
        # Random gamma to simulate non-linear light-curve variations
        transforms.RandomApply(
            [transforms.Lambda(
                lambda x: torch.clamp(
                    x ** torch.empty(1).uniform_(0.5, 1.5).item(), 0, 1
                )
            )],
            p=0.5,
        ),
        # Approximate camera sensor issues at low light
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.RandomApply(
            [transforms.Lambda(lambda x: (x + 0.05 * torch.randn_like(x)).clamp(0, 1))],
            p=0.3,
        ),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transforms, eval_transforms


def get_dataloaders(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    batch_size: int = 32,
    image_size: int = IMAGE_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader]:
    """
    Build and return ``(train_loader, eval_loader)`` with deterministic seeding.
    """
    train_transforms, eval_transforms = get_transforms(image_size)

    train_dataset = ImageDataset(dataframe=train_df, transform=train_transforms)
    eval_dataset  = ImageDataset(dataframe=eval_df,  transform=eval_transforms)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, eval_loader


# =============================================================================
# Model definition
# =============================================================================

class Networks(nn.Module):
    """
    Unified wrapper for five ImageNet pre-trained backbones.

    All backbone weights are loaded from torchvision and the final
    classification head is replaced to output ``num_classes`` logits.
    All parameters are kept trainable (full fine-tuning).
    """

    SUPPORTED_BACKBONES = {
        "resnet50":           (resnet50,           ResNet50_Weights.IMAGENET1K_V1),
        "efficientnet_v2_l":  (efficientnet_v2_l,  EfficientNet_V2_L_Weights.IMAGENET1K_V1),
        "mobilenet_v3_large": (mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V1),
        "vision_transformer": (vit_b_16,            ViT_B_16_Weights.IMAGENET1K_V1),
        "swin_transformer":   (swin_v2_b,           Swin_V2_B_Weights.IMAGENET1K_V1),
    }

    def __init__(self, backbone: str, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                f"Choose from: {list(self.SUPPORTED_BACKBONES)}"
            )

        self.backbone_name = backbone
        self.num_classes = num_classes

        backbone_fn, backbone_weights = self.SUPPORTED_BACKBONES[backbone]
        self.backbone = backbone_fn(weights=backbone_weights)

        # Enable full fine-tuning
        for param in self.backbone.parameters():
            param.requires_grad = True

        self._replace_classifier()
        self.history: Dict[str, list] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
        }

    # ------------------------------------------------------------------
    # Classifier replacement
    # ------------------------------------------------------------------

    def _replace_classifier(self) -> None:
        """Replace the pre-trained classification head with a new linear layer."""
        name = self.backbone_name
        if name == "resnet50":
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, self.num_classes)

        elif name == "efficientnet_v2_l":
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Linear(in_features, self.num_classes)

        elif name == "mobilenet_v3_large":
            in_features = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = nn.Linear(in_features, self.num_classes)

        elif name == "vision_transformer":
            if hasattr(self.backbone, "heads"):
                if hasattr(self.backbone.heads, "head"):
                    in_features = self.backbone.heads.head.in_features
                    self.backbone.heads.head = nn.Linear(in_features, self.num_classes)
                else:
                    in_features = self.backbone.heads.in_features
                    self.backbone.heads = nn.Linear(in_features, self.num_classes)
            else:
                in_features = self.backbone.head.in_features
                self.backbone.head = nn.Linear(in_features, self.num_classes)

        elif name == "swin_transformer":
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(in_features, self.num_classes)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        lr: float = 0.001,
        optimizer_type: str = "adamw",
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        save_best: bool = True,
        save_path: str = "best_model.pth",
    ) -> Dict[str, list]:
        """
        Train the model and optionally save the best checkpoint.

        Parameters
        ----------
        train_loader : DataLoader
            Training data.
        val_loader : DataLoader, optional
            Validation data used for early stopping and checkpoint selection.
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        optimizer_type : str
            One of ``'adam'``, ``'adamw'``, ``'sgd'``.
        criterion : nn.Module, optional
            Loss function. Defaults to ``CrossEntropyLoss``.
        device : torch.device, optional
            Training device. Auto-detected if ``None``.
        save_best : bool
            If ``True``, save the checkpoint with the highest validation accuracy.
        save_path : str
            File path for the best model checkpoint.

        Returns
        -------
        dict
            Training history with keys ``train_loss``, ``train_acc``,
            ``val_loss``, ``val_acc``.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        trainable_params = [p for p in self.parameters() if p.requires_grad]

        if optimizer_type.lower() == "adam":
            optimizer = optim.Adam(trainable_params, lr=lr)
        elif optimizer_type.lower() == "sgd":
            optimizer = optim.SGD(trainable_params, lr=lr, momentum=0.9)
        elif optimizer_type.lower() == "adamw":
            optimizer = optim.AdamW(trainable_params, lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: '{optimizer_type}'")

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)

        total_params     = sum(p.numel() for p in self.parameters())
        trainable_count  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Device    : {device}")
        print(f"  Backbone  : {self.backbone_name}")
        print(f"  Parameters: {total_params:,} total, "
              f"{trainable_count:,} trainable "
              f"({trainable_count / total_params * 100:.1f}%)")

        best_val_acc = 0.0

        for epoch in range(epochs):
            t0 = time.time()
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer, device)

            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader, criterion, device)
                scheduler.step(val_loss)

                if save_best and val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_acc": val_acc,
                            "val_loss": val_loss,
                        },
                        save_path,
                    )

                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                print(
                    f"  Epoch {epoch + 1}/{epochs}  [{time.time() - t0:.1f}s]  "
                    f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                    f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                )
            else:
                print(
                    f"  Epoch {epoch + 1}/{epochs}  [{time.time() - t0:.1f}s]  "
                    f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                )

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

        return self.history

    def _train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> tuple[float, float]:
        """Run one training epoch and return ``(avg_loss, accuracy)``."""
        self.train()
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(train_loader, desc="  Training", leave=False) as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = output.max(1)
                total   += target.size(0)
                correct += predicted.eq(target).sum().item()

                pbar.set_postfix(
                    loss=f"{running_loss / (batch_idx + 1):.4f}",
                    acc=f"{100.0 * correct / total:.2f}%",
                )

        return running_loss / len(train_loader), correct / total

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        data_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ) -> tuple[float, float]:
        """
        Evaluate the model on *data_loader*.

        Returns
        -------
        avg_loss : float
        accuracy : float
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad(), tqdm(data_loader, desc="  Evaluating", leave=False) as pbar:
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                output = self(data)
                loss = criterion(output, target)

                running_loss += loss.item()
                _, predicted = output.max(1)
                total   += len(data)
                correct += predicted.eq(target).sum().item()

                pbar.set_postfix(acc=f"{100.0 * correct / total:.2f}%")

        return running_loss / len(data_loader), correct / total

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        data_loader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> tuple[list, list]:
        """
        Generate predictions for all samples in *data_loader*.

        Returns
        -------
        predictions : list of int
        probabilities : list of np.ndarray  (shape: [n_classes])
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eval()
        predictions: list  = []
        probabilities: list = []

        with torch.no_grad(), tqdm(data_loader, desc="  Predicting", leave=False) as pbar:
            for data, _ in pbar:
                data = data.to(device)
                output = self(data)
                probs = torch.softmax(output, dim=1)
                _, predicted = output.max(1)
                predictions.extend(predicted.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())

        return predictions, probabilities

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def load_checkpoint(
        self, checkpoint_path: str, device: Optional[torch.device] = None
    ) -> dict:
        """Load model weights from a checkpoint file."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    def get_model_info(self) -> dict:
        """Return a summary dict of parameter counts."""
        total      = sum(p.numel() for p in self.parameters())
        trainable  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "backbone":              self.backbone_name,
            "num_classes":           self.num_classes,
            "total_parameters":      total,
            "trainable_parameters":  trainable,
            "frozen_parameters":     total - trainable,
            "trainable_percentage":  trainable / total * 100,
        }


# =============================================================================
# K-fold cross-validation trainer
# =============================================================================

class KFoldTrainer:
    """
    Stratified k-fold cross-validation grid search followed by final training.

    Parameters
    ----------
    networks_class : type
        The ``Networks`` class (or any compatible nn.Module subclass).
    get_dataloaders_func : callable
        Function with signature ``(train_df, eval_df, batch_size) -> (train_loader, eval_loader)``.
    """

    def __init__(self, networks_class, get_dataloaders_func) -> None:
        self.Networks = networks_class
        self.get_dataloaders = get_dataloaders_func
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.hyperparams = KFOLD_HYPERPARAMS
        self.backbones   = KFOLD_BACKBONES
        self.n_folds     = N_FOLDS
        self.kfold       = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=SEED
        )

        self.results:       dict = {}
        self.best_params:   dict = {}
        self.final_results: dict = {}

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------

    def _prepare_kfold_data(
        self, full_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, str]:
        label_col = "label" if "label" in full_df.columns else full_df.columns[-1]
        X = full_df.drop(columns=[label_col])
        y = full_df[label_col]
        print(f"  Dataset size : {len(full_df)} samples")
        print(f"  Class counts : {y.value_counts().to_dict()}")
        print(f"  Folds        : {self.n_folds}")
        return X, y, label_col

    def stratified_kfold_search(
        self, backbone: str, full_df: pd.DataFrame
    ) -> tuple[dict, float]:
        """
        Run a stratified k-fold grid search for *backbone* over ``self.hyperparams``.

        Returns
        -------
        best_params : dict
        best_cv_score : float
        """
        print(f"\n{'='*60}")
        print(f"  Grid search — {backbone}")
        print(f"{'='*60}")

        X, y, label_col = self._prepare_kfold_data(full_df)
        param_combinations = list(product(*self.hyperparams.values()))
        param_names        = list(self.hyperparams.keys())

        best_score  = 0.0
        best_params = None
        all_results: list = []

        for i, param_values in enumerate(param_combinations):
            params = dict(zip(param_names, param_values))
            print(f"\n  Combination {i + 1}/{len(param_combinations)}: {params}")

            try:
                fold_accuracies: list = []
                fold_losses:     list = []
                fold_results:    list = []

                for fold, (train_idx, val_idx) in enumerate(self.kfold.split(X, y)):
                    print(f"    Fold {fold + 1}/{self.n_folds} ... ", end="")

                    train_fold_df = pd.concat(
                        [X.iloc[train_idx], y.iloc[train_idx]], axis=1
                    )
                    val_fold_df = pd.concat(
                        [X.iloc[val_idx], y.iloc[val_idx]], axis=1
                    )

                    train_loader, _ = self.get_dataloaders(
                        train_fold_df, train_fold_df, batch_size=params["batch_size"]
                    )
                    _, val_loader = self.get_dataloaders(
                        val_fold_df, val_fold_df, batch_size=len(val_fold_df)
                    )

                    set_all_seeds(SEED)
                    torch.use_deterministic_algorithms(True)

                    model = self.Networks(backbone=backbone, num_classes=NUM_CLASSES)
                    model.to(self.device)

                    history = model.train_model(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        epochs=params["epochs"],
                        lr=params["lr"],
                        optimizer_type=params["optimizer"],
                        device=self.device,
                        save_best=False,
                    )

                    val_loss, val_acc = model.evaluate(val_loader, device=self.device)
                    fold_accuracies.append(val_acc)
                    fold_losses.append(val_loss)
                    fold_results.append(
                        {"fold": fold + 1, "val_acc": val_acc, "val_loss": val_loss}
                    )
                    print(f"acc={val_acc:.4f}")

                mean_acc = float(np.mean(fold_accuracies))
                std_acc  = float(np.std(fold_accuracies))
                mean_loss = float(np.mean(fold_losses))
                std_loss  = float(np.std(fold_losses))

                print(
                    f"  CV — acc={mean_acc:.4f}±{std_acc:.4f}, "
                    f"loss={mean_loss:.4f}±{std_loss:.4f}"
                )

                result = {
                    "params":       params.copy(),
                    "cv_mean_acc":  mean_acc,
                    "cv_std_acc":   std_acc,
                    "cv_mean_loss": mean_loss,
                    "cv_std_loss":  std_loss,
                    "fold_results": fold_results,
                }
                all_results.append(result)

                if mean_acc > best_score:
                    best_score  = mean_acc
                    best_params = params.copy()
                    print(f"  *** New best: {mean_acc:.4f}±{std_acc:.4f} ***")

            except Exception as exc:
                print(f"  FAILED: {exc}")
                all_results.append(
                    {"params": params.copy(), "cv_mean_acc": 0.0, "error": str(exc)}
                )

        self.results[backbone]     = all_results
        self.best_params[backbone] = best_params

        print(f"\n  Best params : {best_params}")
        print(f"  Best CV acc : {best_score:.4f}")
        return best_params, best_score

    # ------------------------------------------------------------------
    # Final training
    # ------------------------------------------------------------------

    def train_final_model(
        self,
        backbone: str,
        best_params: dict,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path,
    ):
        """
        Re-train *backbone* on the full training set using *best_params*,
        then evaluate on *test_df*.

        Returns
        -------
        model : Networks
        test_acc : float
        test_loss : float
        """
        print(f"\n{'='*60}")
        print(f"  Final training — {backbone}")
        print(f"  Params: {best_params}")
        print(f"{'='*60}")

        train_loader, _ = self.get_dataloaders(
            train_df, test_df, batch_size=best_params["batch_size"]
        )
        _, test_loader = self.get_dataloaders(
            train_df, test_df, batch_size=len(test_df)
        )

        set_all_seeds(SEED)
        torch.use_deterministic_algorithms(True)

        model = self.Networks(backbone=backbone, num_classes=NUM_CLASSES)
        model.to(self.device)

        save_path = str(output_dir / f"best_{backbone}_model.pth")
        history = model.train_model(
            train_loader=train_loader,
            val_loader=None,
            epochs=best_params["epochs"],
            lr=best_params["lr"],
            optimizer_type=best_params["optimizer"],
            device=self.device,
            save_best=True,
            save_path=save_path,
        )

        test_loss, test_acc = model.evaluate(test_loader, device=self.device)
        predictions, probabilities = model.predict(test_loader, device=self.device)

        cv_score = self._get_cv_score(backbone)
        self.final_results[backbone] = {
            "test_accuracy": test_acc,
            "test_loss":     test_loss,
            "best_params":   best_params,
            "history":       history,
            "cv_score":      cv_score,
            "model_info":    model.get_model_info(),
        }

        print(f"  Test accuracy : {test_acc:.4f}")
        print(f"  Test loss     : {test_loss:.4f}")
        return model, test_acc, test_loss

    def _get_cv_score(self, backbone: str) -> dict:
        if backbone in self.results:
            best = max(self.results[backbone], key=lambda r: r.get("cv_mean_acc", 0))
            return {"mean_acc": best.get("cv_mean_acc", 0), "std_acc": best.get("cv_std_acc", 0)}
        return {"mean_acc": 0.0, "std_acc": 0.0}

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        full_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path,
    ) -> dict:
        """
        Run the complete pipeline (grid search + final training) for all backbones.

        Returns
        -------
        dict mapping backbone name → trained ``Networks`` instance.
        """
        print(f"{'='*60}")
        print("  K-Fold Training Pipeline")
        print(f"  Device          : {self.device}")
        print(f"  Training samples: {len(full_df)}")
        print(f"  Test samples    : {len(test_df)}")
        print(f"  Folds           : {self.n_folds}")
        print(f"{'='*60}")

        final_models: dict = {}

        for backbone in self.backbones:
            try:
                best_params, _ = self.stratified_kfold_search(backbone, full_df)
                model, _, _    = self.train_final_model(
                    backbone, best_params, full_df, test_df, output_dir
                )
                final_models[backbone] = model
            except Exception as exc:
                print(f"  ERROR [{backbone}]: {exc}")
                self.final_results[backbone] = {
                    "error": str(exc),
                    "test_accuracy": 0.0,
                    "test_loss": float("inf"),
                    "cv_score": {"mean_acc": 0.0, "std_acc": 0.0},
                }

        self._print_summary()
        return final_models

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _print_summary(self) -> None:
        print(f"\n{'='*70}")
        print("  FINAL SUMMARY")
        print(f"{'='*70}")

        rows = []
        for backbone in self.backbones:
            result = self.final_results.get(backbone, {})
            if "error" not in result:
                cv = result.get("cv_score", {})
                rows.append({
                    "Backbone":       backbone,
                    "CV Score":       f"{cv.get('mean_acc', 0):.4f}±{cv.get('std_acc', 0):.3f}",
                    "Test Accuracy":  f"{result['test_accuracy']:.4f}",
                    "Test Loss":      f"{result['test_loss']:.4f}",
                    "LR":             result["best_params"]["lr"],
                    "Optimizer":      result["best_params"]["optimizer"],
                    "Epochs":         result["best_params"]["epochs"],
                    "Batch Size":     result["best_params"]["batch_size"],
                })
            else:
                rows.append({"Backbone": backbone, **{k: "ERROR" for k in
                    ["CV Score", "Test Accuracy", "Test Loss", "LR", "Optimizer", "Epochs", "Batch Size"]}})

        print(pd.DataFrame(rows).to_string(index=False))

        best_backbone, best_acc = None, 0.0
        for backbone, result in self.final_results.items():
            if "error" not in result and result["test_accuracy"] > best_acc:
                best_acc, best_backbone = result["test_accuracy"], backbone

        if best_backbone:
            cv = self.final_results[best_backbone].get("cv_score", {})
            print(f"\n  Best model : {best_backbone}")
            print(f"  CV Score   : {cv.get('mean_acc', 0):.4f}±{cv.get('std_acc', 0):.3f}")
            print(f"  Test Acc   : {best_acc:.4f}")


# =============================================================================
# Final per-backbone training (post grid-search)
# =============================================================================

def train_all_backbones(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Train each backbone in ``BACKBONE_CONFIGS`` using the specified hyper-parameters,
    evaluate on *val_df*, and persist predictions and a summary CSV.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split.
    val_df : pd.DataFrame
        Validation / calibration split used for evaluation and saved predictions.
    output_dir : Path
        Directory where checkpoints, prediction CSVs, and the summary CSV are written.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per backbone.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_rows: list = []

    for backbone, config in BACKBONE_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  Training — {backbone}")
        print(f"  Config  — {config}")
        print(f"{'='*60}")

        train_loader, _ = get_dataloaders(
            train_df, val_df, batch_size=config["batch_size"], image_size=IMAGE_SIZE
        )
        _, eval_loader = get_dataloaders(
            train_df, val_df, batch_size=len(val_df), image_size=IMAGE_SIZE
        )

        torch.use_deterministic_algorithms(True)
        set_all_seeds(SEED)

        model = Networks(backbone=backbone, num_classes=NUM_CLASSES)
        checkpoint_path = str(output_dir / f"{backbone}_best.pth")

        model.train_model(
            train_loader=train_loader,
            val_loader=eval_loader,
            epochs=config["epochs"],
            lr=config["lr"],
            optimizer_type="adam",
            save_best=True,
            save_path=checkpoint_path,
        )

        test_loss, test_acc = model.evaluate(eval_loader, device=device)
        predictions, probabilities = model.predict(eval_loader, device=device)

        # Collect true labels from the dataloader
        true_labels: list = []
        for _, labels in eval_loader:
            true_labels.extend(labels.cpu().numpy())

        # Build and save the prediction CSV
        prob_array = np.array(probabilities)
        pred_df = pd.DataFrame({
            "True Label":      true_labels,
            "Predicted Label": predictions,
        })
        prob_df = pd.DataFrame(
            prob_array,
            columns=[f"Prob_Class_{i}" for i in range(prob_array.shape[1])],
        )
        pred_df = pd.concat([pred_df, prob_df], axis=1)

        pred_path = output_dir / f"predictions_calib_{backbone}.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"  Predictions saved to '{pred_path}'")

        # Accumulate summary row
        info = model.get_model_info()
        summary_rows.append({
            "Backbone":              backbone,
            "Test Accuracy":         test_acc,
            "Test Loss":             test_loss,
            "LR":                    config["lr"],
            "Epochs":                config["epochs"],
            "Batch Size":            config["batch_size"],
            "Trainable Parameters":  info["trainable_parameters"],
            "Total Parameters":      info["total_parameters"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "model_results_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to '{summary_path}'")
    return summary_df


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)

    # ------------------------------------------------------------------
    # 1. Build dataset splits
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Building dataset splits …")
    print("=" * 60)
    train_df, val_df, test_df = build_splits(TRAIN_FOLDER, TEST_FOLDER)

    # ------------------------------------------------------------------
    # 2. K-fold grid search + per-backbone final training
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("K-Fold cross-validation grid search …")
    print("=" * 60)
    trainer = KFoldTrainer(
        networks_class=Networks,
        get_dataloaders_func=get_dataloaders,
    )
    trainer.run_pipeline(full_df=train_df, test_df=test_df, output_dir=OUTPUT_DIR)

    # ------------------------------------------------------------------
    # 3. Final training with pre-selected configs (BACKBONE_CONFIGS)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Final training with selected hyper-parameters …")
    print("=" * 60)
    summary = train_all_backbones(train_df, val_df, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()