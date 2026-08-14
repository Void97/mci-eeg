"""Framework-level training primitives shared by the full k-fold trainer
and the hyperparameter pilot search (models_train.train_func). Nothing in
this module knows about k-fold splitting, datasets, or file I/O -- just
"given a model/loader/optimizer/criterion/device, run one training or
evaluation pass" -- so it's usable and testable independent of the rest of
the benchmarking pipeline.
"""

import time
from dataclasses import dataclass
from typing import Any

import torch
from sklearn.metrics import accuracy_score

from models.models_base.models_list import ModelSpec


@dataclass
class RunContext:
    """Identifies *what* is being trained: everything about a single
    (dataset, task, model, iteration) run except the actual tensors."""
    dataset_name: str
    task: str
    model_spec: ModelSpec
    iter: int
    seed: int
    dirs: dict
    k: int


@dataclass
class FoldData:
    """The tensors/arrays a run is trained and evaluated on."""
    dataset: Any
    samples: Any
    targets: Any
    groups: Any


def logits_from_output(out):
    """Every model except MSVTNet returns a plain logits tensor. MSVTNet
    (with its default b_preds=True) returns (main_logits, branch_logits) so
    its auxiliary branches can feed JointCrossEntoryLoss -- this extracts
    just the part predictions/accuracy computation actually needs, without
    needing to know which model produced `out`."""
    return out[0] if isinstance(out, tuple) else out


def captum_forward_fn(model):
    def forward(x):
        return logits_from_output(model(x))
    return forward


def build_optimizer_and_criterion(model, model_spec: ModelSpec, lr, weight_decay):
    optimizer = model_spec.optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = model_spec.criterion_cls()
    return optimizer, criterion


def run_train_epoch(model, loader, optimizer, criterion, device):
    """One epoch of training. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * yb.size(0)
        preds = logits_from_output(out).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    """Forward pass in eval mode, no gradient. Returns (avg_loss, accuracy, preds, labels)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb.long())
            total_loss += loss.item() * yb.size(0)
            preds = logits_from_output(out).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


def run_test_inference(model, loader, device):
    """Timed test-set forward pass. Returns (preds, labels, roc_probs, elapsed_seconds)."""
    model.eval()
    preds, labels, roc = [], [], []
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = logits_from_output(model(xb))
            batch_preds = logits.argmax(dim=1)
            preds.extend(batch_preds.cpu().numpy())
            labels.extend(yb.cpu().numpy())
            roc.extend(torch.sigmoid(logits).cpu().numpy())
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
    return preds, labels, roc, elapsed