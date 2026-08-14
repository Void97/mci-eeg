"""End-to-end smoke test against real GENEEG data: pilot search -> k-fold
training -> checkpoints -> t-SNE -> saliency -> logs. Everything writes to
a pytest tmp_path scratch dir, never the real results/ tree.

Marked slow and skipped automatically if the real preprocessed data isn't
present locally (it's gitignored, multi-GB, and not something a fresh
clone has) -- run explicitly with `pytest -m slow`.
"""
import os

import pytest
import torch
from torch.utils.data import TensorDataset

from dataclasses import replace

from main_func import build, labels_mapping, filter_data
from utils.EEG_preprocess import load_preprocessed
from models.models_base.models_list import single_model
from models.models_train.train_func import train, train_pilot
from models.models_train.training_loop import RunContext, FoldData

DATASET_NAME = 'GENEEG'
TASK = 'MCI vs HC'
GENEEG_PREPROCESSED_DIR = './datasets/preprocessed/GENEEG_preprocessed'

pytestmark = pytest.mark.slow

requires_real_data = pytest.mark.skipif(
    not os.path.isdir(GENEEG_PREPROCESSED_DIR),
    reason=f"real preprocessed GENEEG data not found at {GENEEG_PREPROCESSED_DIR!r} "
           "(gitignored, not present on a fresh clone)",
)


@requires_real_data
def test_full_pipeline_smoke(tmp_path):
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
        os.makedirs(d, exist_ok=True)

    k = 3
    seed = 42
    training = {
        'patience': 1, 'full_train_max_epochs': 2, 'pilot_max_epochs': 2,
        'val_batch_size': 32, 'test_batch_size': 32, 'subsample_fraction': 0.1,
        'tsne_n_iter': 250, 'tsne_random_state': 42,
    }
    hyperparam_grid = {'batch_sizes': [16], 'learning_rates': [0.001], 'l2_weight_decays': [0.0]}

    dataset_path, preprocessed_dir, ch_num, ch_names, show_channel, tasks, metadata, subjects_list, labels_list = build(DATASET_NAME)
    labels_map = labels_mapping(DATASET_NAME, TASK)
    filtered_subjects, filtered_labels = filter_data(subjects_list, labels_list, labels_map)
    label_dict = dict(zip(filtered_subjects, filtered_labels))
    samples, targets, groups = load_preprocessed(preprocessed_dir, labels_map, label_dict)

    model_spec = single_model(num_classes=2, num_channels=ch_num, time_points=samples.shape[2], model_name='SCCNet')
    model_spec = replace(model_spec, kwargs={**model_spec.kwargs, 'num_classes': 2})

    full_samples = torch.from_numpy(samples).unsqueeze(1).float()
    full_targets = torch.from_numpy(targets).float()
    full_dataset = TensorDataset(full_samples, full_targets)
    fold_data = FoldData(dataset=full_dataset, samples=full_samples.numpy(), targets=full_targets.numpy(), groups=groups)

    best_params = train_pilot(model_spec=model_spec, seed=seed, k=k, fold_data=fold_data,
                               hyperparam_grid=hyperparam_grid, training=training)
    assert set(['batch size', 'learning rate', 'L2 weight decay', 'val_acc', 'val_loss']).issubset(best_params.keys())

    ctx = RunContext(dataset_name=DATASET_NAME, task=TASK, model_spec=model_spec, iter=0, seed=seed, dirs=dirs, k=k)
    seg_acc, sub_acc, sub_sens, sub_spec, sub_prec, sub_f1, saliency_maps = train(ctx, fold_data, best_params, training=training)

    assert 0.0 <= seg_acc <= 1.0
    assert 0.0 <= sub_acc <= 1.0
    assert saliency_maps is not None and len(saliency_maps) > 0

    assert len(os.listdir(dirs['weights_dir'])) == k
    assert len(os.listdir(dirs['tsne_dir'])) == k
    assert len(os.listdir(dirs['training_inference_time_logs_dir'])) == 1
    assert len(os.listdir(dirs['k_fold_dist_logs_dir'])) == 1