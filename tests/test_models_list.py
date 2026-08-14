"""Tests for models.models_base.models_list -- the model registry."""
import pytest
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, AdamW

from models.models_base.MSVTNet import JointCrossEntoryLoss
from models.models_base.models_list import MODEL_NAMES, ModelSpec, models_list, single_model

NUM_CHANNELS, NUM_CLASSES = 19, 2
# MSVTNet's conv/pooling stack needs a larger time dimension than the other
# models -- see test_msvtnet_forward_returns_tuple below.
TIME_POINTS = 800

# Oh_CNN, SzHNN, and EEG_Deformer expect (batch, C, T) -- no image-channel
# dimension -- unlike every other model, which expects (batch, 1, C, T).
# This mirrors main.py's own special-casing (see its `full_samples =`
# branch), not something introduced by these tests.
NO_CHANNEL_DIM_MODELS = {'Oh_CNN', 'SzHNN', 'EEG_Deformer'}


def make_input(model_name, batch_size=2):
    x = torch.randn(batch_size, NUM_CHANNELS, TIME_POINTS)
    return x if model_name in NO_CHANNEL_DIM_MODELS else x.unsqueeze(1)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_instantiates(model_name):
    spec = single_model(NUM_CLASSES, NUM_CHANNELS, TIME_POINTS, model_name=model_name)
    assert isinstance(spec, ModelSpec)
    model = spec.cls(**spec.kwargs)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_mbszeegnet_uses_sgd():
    spec = single_model(NUM_CLASSES, NUM_CHANNELS, TIME_POINTS, model_name='MBSzEEGNet')
    assert spec.optimizer_cls is SGD
    assert spec.criterion_cls is CrossEntropyLoss


def test_msvtnet_uses_adamw_and_joint_cross_entropy_loss():
    spec = single_model(NUM_CLASSES, NUM_CHANNELS, TIME_POINTS, model_name='MSVTNet')
    assert spec.optimizer_cls is AdamW
    assert spec.criterion_cls is JointCrossEntoryLoss


def test_all_other_models_use_adamw_and_cross_entropy_loss():
    for name in MODEL_NAMES:
        if name in ('MBSzEEGNet', 'MSVTNet'):
            continue
        spec = single_model(NUM_CLASSES, NUM_CHANNELS, TIME_POINTS, model_name=name)
        assert spec.optimizer_cls is AdamW, name
        assert spec.criterion_cls is CrossEntropyLoss, name


def test_msvtnet_forward_returns_tuple():
    """MSVTNet is the only model whose forward() returns a tuple (its
    default b_preds=True) -- everything downstream (logits_from_output,
    captum_forward_fn) assumes this is the *only* one, so if this ever
    stops being true (or another model starts doing it), predictions would
    silently be extracted wrong."""
    spec = single_model(NUM_CLASSES, NUM_CHANNELS, TIME_POINTS, model_name='MSVTNet')
    model = spec.cls(**spec.kwargs)
    out = model(make_input('MSVTNet'))
    assert isinstance(out, tuple)


@pytest.mark.parametrize("model_name", [n for n in MODEL_NAMES if n != 'MSVTNet'])
def test_non_msvtnet_models_do_not_return_tuples(model_name):
    spec = single_model(NUM_CLASSES, NUM_CHANNELS, TIME_POINTS, model_name=model_name)
    model = spec.cls(**spec.kwargs)
    out = model(make_input(model_name))
    assert not isinstance(out, tuple), model_name


def test_single_model_unknown_name_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown model_name"):
        single_model(NUM_CLASSES, NUM_CHANNELS, TIME_POINTS, model_name='NotARealModel')


def test_models_list_returns_all_registered_models():
    specs = models_list(NUM_CLASSES, NUM_CHANNELS, TIME_POINTS)
    assert {s.name for s in specs} == set(MODEL_NAMES)