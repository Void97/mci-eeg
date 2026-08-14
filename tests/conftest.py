"""Shared fixtures for the fast/unit test tier: synthetic data only, no
real EEG files, no GPU required. Everything here should run in well under
a second per test.
"""
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from models.models_base.models_list import single_model
from models.models_train.training_loop import FoldData

NUM_SUBJECTS_PER_CLASS = 12
SEGMENTS_PER_SUBJECT = 5
NUM_CHANNELS = 4
TIME_POINTS = 64


@pytest.fixture
def synthetic_fold_data():
    """~120 segments from 24 subjects (12 per class), enough for
    StratifiedGroupKFold at k=3 without degenerate folds."""
    rng = np.random.RandomState(0)

    groups, targets = [], []
    subject_id = 0
    for label in (0, 1):
        for _ in range(NUM_SUBJECTS_PER_CLASS):
            for _ in range(SEGMENTS_PER_SUBJECT):
                groups.append(subject_id)
                targets.append(label)
            subject_id += 1
    groups = np.array(groups)
    targets = np.array(targets, dtype=np.float64)
    n = len(targets)

    # Make the two classes trivially separable so training actually converges
    # in a handful of epochs -- these tests check plumbing, not model skill.
    samples = rng.randn(n, NUM_CHANNELS, TIME_POINTS).astype(np.float32)
    samples += (targets[:, None, None] * 3.0)

    tensor_samples = torch.from_numpy(samples).unsqueeze(1).float()  # (n, 1, C, T)
    tensor_targets = torch.from_numpy(targets).float()
    dataset = TensorDataset(tensor_samples, tensor_targets)

    return FoldData(dataset=dataset, samples=samples, targets=targets, groups=groups)


@pytest.fixture
def tiny_model_spec():
    """SCCNet at a small shape -- fast to instantiate and train, and (unlike
    MSVTNet) has no minimum-input-size constraint to work around."""
    spec = single_model(num_classes=2, num_channels=NUM_CHANNELS, time_points=TIME_POINTS, model_name='SCCNet')
    return replace(spec, kwargs={**spec.kwargs, 'num_classes': 2})


@pytest.fixture
def fast_training_settings():
    """Same shape as configs/benchmark.yaml's `training:` section, with
    tiny values so tests run in milliseconds."""
    return {
        'patience': 1,
        'full_train_max_epochs': 2,
        'pilot_max_epochs': 2,
        'val_batch_size': 16,
        'test_batch_size': 16,
        'subsample_fraction': 0.5,
        'tsne_n_iter': 250,
        'tsne_random_state': 42,
    }


@pytest.fixture
def scratch_dirs(tmp_path):
    """A `dirs` dict pointing at a pytest-managed temp directory --
    auto-cleaned, never touches the real results/ tree."""
    dirs = {
        'k_fold_dist_logs_dir': str(tmp_path / 'kfold_dist_logs'),
        'best_params_logs_dir': str(tmp_path / 'best_params_logs'),
        'weights_dir': str(tmp_path / 'weights'),
        'predictions_dir': str(tmp_path / 'predictions'),
        'metrics_dir': str(tmp_path / 'metrics'),
        'metrics_logs_dir': str(tmp_path / 'metrics' / 'logs'),
        'tsne_dir': str(tmp_path / 'tsne'),
        'topomaps_dir': str(tmp_path / 'topomaps'),
        'training_inference_time_logs_dir': str(tmp_path / 'timing'),
    }
    for d in dirs.values():
        import os
        os.makedirs(d, exist_ok=True)
    return dirs