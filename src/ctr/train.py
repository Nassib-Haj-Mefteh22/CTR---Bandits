import argparse
import csv
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ctr.config import DATA_DIR, EMBED_DIM, HASH_SIZE, RESULTS_DIR, SEED
from ctr.data import CTRDataset, load_split, make_loader, rehash_ids
from ctr.metrics import ctr_metrics
from ctr.models.dcnv2 import N_CROSS, DCNv2
from ctr.models.deepfm import DeepFM
from ctr.models.fm import FM
from ctr.models.logreg import LogisticRegression

CHECKPOINT_DIR = DATA_DIR / "checkpoints"
METRICS_CSV = RESULTS_DIR / "metrics.csv"
MODELS: dict[str, Callable[..., nn.Module]] = {
    "logreg": LogisticRegression,
    "fm": FM,
    "deepfm": DeepFM,
    "dcnv2": DCNv2,
}


@dataclass
class ModelConfig:
    embed_dim: int = EMBED_DIM
    n_cross: int = N_CROSS
    hash_size: int = HASH_SIZE

    def tag(self) -> str:
        return f"k{self.embed_dim}_c{self.n_cross}_h{self.hash_size.bit_length() - 1}"


def build_model(name: str, mcfg: ModelConfig) -> nn.Module:
    kwargs: dict = {"n_ids": mcfg.hash_size}
    if name != "logreg":
        kwargs["embed_dim"] = mcfg.embed_dim
    if name == "dcnv2":
        kwargs["n_cross"] = mcfg.n_cross
    return MODELS[name](**kwargs)


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 2
    seed: int = SEED


@dataclass
class TrainResult:
    best_epoch: int
    best_val: dict[str, float]
    history: list[dict[str, float]] = field(default_factory=list)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, probs = [], []
    for dense, sparse, label in loader:
        probs.append(torch.sigmoid(model(dense, sparse)).numpy())
        labels.append(label.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module
) -> float:
    model.train()
    total, n = 0.0, 0
    for dense, sparse, label in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(dense, sparse), label)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(label)
        n += len(label)
    return total / n


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    checkpoint: Path,
) -> TrainResult:
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_loss, best_epoch, best_val, bad_epochs = float("inf"), 0, {}, 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val = ctr_metrics(*predict(model, val_loader))
        seconds = time.perf_counter() - start
        history.append({"epoch": epoch, "train_loss": train_loss, **val, "seconds": seconds})
        improved = val["logloss"] < best_loss
        print(
            f"epoch {epoch:2d}  train_loss {train_loss:.5f}  val_logloss {val['logloss']:.5f}  "
            f"val_auc {val['auc']:.5f}  calib {val['calib_ratio']:.3f}  {seconds:5.1f}s"
            + ("  *best" if improved else ""),
            flush=True,
        )
        if improved:
            best_loss, best_epoch, best_val, bad_epochs = val["logloss"], epoch, val, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"early stop: no val improvement for {cfg.patience} epoch(s)")
                break
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    return TrainResult(best_epoch=best_epoch, best_val=best_val, history=history)


def append_metrics_row(row: dict, path: Path = METRICS_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def run(
    name: str,
    cfg: TrainConfig,
    final: bool = True,
    mcfg: ModelConfig | None = None,
    csv_path: Path = METRICS_CSV,
    extra: dict | None = None,
) -> dict:
    mcfg = mcfg or ModelConfig()
    set_seed(cfg.seed)
    splits = [load_split(s) for s in ("train", "val", "test")]
    if mcfg.hash_size != HASH_SIZE:
        splits = [(d, rehash_ids(sp, mcfg.hash_size), y) for d, sp, y in splits]
    train_ds, val_ds, test_ds = (CTRDataset(*s) for s in splits)
    train_loader = make_loader(train_ds, cfg.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = make_loader(val_ds, 65536, shuffle=False)
    test_loader = make_loader(test_ds, 65536, shuffle=False)

    model = build_model(name, mcfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model {name}: {n_params:,} parameters, train rows {len(train_ds):,}, {cfg}, {mcfg}")
    start = time.perf_counter()
    checkpoint = CHECKPOINT_DIR / f"{name}_{mcfg.tag()}_s{cfg.seed}.pt"
    result = fit(model, train_loader, val_loader, cfg, checkpoint)
    train_seconds = time.perf_counter() - start
    if not final:
        return {"model": name, "best_epoch": result.best_epoch, **result.best_val}
    test = ctr_metrics(*predict(model, test_loader))
    print(
        f"TEST  auc {test['auc']:.5f}  logloss {test['logloss']:.5f}  mean_pred "
        f"{test['mean_pred']:.4f} vs ctr {test['true_ctr']:.4f}  (best epoch {result.best_epoch})"
    )
    row = {
        **(extra or {}),
        "model": name,
        "seed": cfg.seed,
        "best_epoch": result.best_epoch,
        "val_logloss": round(result.best_val["logloss"], 5),
        "val_auc": round(result.best_val["auc"], 5),
        "test_auc": round(test["auc"], 5),
        "test_logloss": round(test["logloss"], 5),
        "test_mean_pred": round(test["mean_pred"], 4),
        "test_ctr": round(test["true_ctr"], 4),
        "n_params": n_params,
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "weight_decay": cfg.weight_decay,
        "train_seconds": round(train_seconds, 1),
    }
    append_metrics_row(row, csv_path)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CTR model")
    parser.add_argument("--model", choices=sorted(MODELS), default="logreg")
    defaults = TrainConfig()
    for key, value in vars(defaults).items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    model_defaults = ModelConfig()
    for key, value in vars(model_defaults).items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    parser.add_argument("--tune", action="store_true", help="val only: no test score, no csv row")
    args = parser.parse_args()
    cfg = TrainConfig(**{k: getattr(args, k) for k in vars(defaults)})
    mcfg = ModelConfig(**{k: getattr(args, k) for k in vars(model_defaults)})
    run(args.model, cfg, final=not args.tune, mcfg=mcfg)


if __name__ == "__main__":
    main()
