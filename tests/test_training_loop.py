"""Tests for models.models_train.training_loop -- the framework-level,
model-agnostic training primitives. All synthetic data, CPU, no k-fold or
dataset knowledge involved (that's tested in test_train_func.py)."""
import torch
from torch.utils.data import DataLoader

from models.models_train.training_loop import (
    build_optimizer_and_criterion, evaluate, logits_from_output,
    run_test_inference, run_train_epoch,
)

DEVICE = torch.device('cpu')


def test_logits_from_output_passes_plain_tensor_through():
    x = torch.randn(4, 2)
    assert logits_from_output(x) is x


def test_logits_from_output_unwraps_tuple():
    main_out = torch.randn(4, 2)
    out = (main_out, [torch.randn(4, 2)])
    assert logits_from_output(out) is main_out


def test_run_train_epoch_reduces_loss(tiny_model_spec, synthetic_fold_data):
    loader = DataLoader(synthetic_fold_data.dataset, batch_size=16, shuffle=True, drop_last=True)
    model = tiny_model_spec.cls(**tiny_model_spec.kwargs).to(DEVICE)
    optimizer, criterion = build_optimizer_and_criterion(model, tiny_model_spec, lr=1e-3, weight_decay=0.0)

    loss0, _ = run_train_epoch(model, loader, optimizer, criterion, DEVICE)
    for _ in range(15):
        loss, acc = run_train_epoch(model, loader, optimizer, criterion, DEVICE)

    assert loss < loss0, "loss should drop after 15 epochs on a trivially-separable synthetic set"
    assert acc > 0.8, f"expected near-perfect accuracy on separable synthetic data, got {acc}"


def test_evaluate_returns_correctly_shaped_outputs(tiny_model_spec, synthetic_fold_data):
    loader = DataLoader(synthetic_fold_data.dataset, batch_size=16, shuffle=False)
    model = tiny_model_spec.cls(**tiny_model_spec.kwargs).to(DEVICE)
    _, criterion = build_optimizer_and_criterion(model, tiny_model_spec, lr=1e-3, weight_decay=0.0)

    loss, acc, preds, labels = evaluate(model, loader, criterion, DEVICE)
    assert len(preds) == len(labels) == len(synthetic_fold_data.targets)
    assert 0.0 <= acc <= 1.0


def test_run_test_inference_returns_correctly_shaped_outputs(tiny_model_spec, synthetic_fold_data):
    loader = DataLoader(synthetic_fold_data.dataset, batch_size=16, shuffle=False)
    model = tiny_model_spec.cls(**tiny_model_spec.kwargs).to(DEVICE)

    preds, labels, roc, elapsed = run_test_inference(model, loader, DEVICE)
    assert len(preds) == len(labels) == len(roc) == len(synthetic_fold_data.targets)
    assert elapsed >= 0