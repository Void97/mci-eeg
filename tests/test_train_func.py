"""Tests for models.models_train.train_func -- the k-fold CV orchestration.

These are the highest-value tests in the suite: they cover the two real
bugs found during this pipeline's refactor (the checkpoint-reuse eval()
bug, and the training-time-averaging dilution bug) as permanent
regression tests, plus the subject-leakage guarantee the whole k-fold
design depends on.
"""
import json

import pytest
from torch.utils.data import DataLoader

from models.models_train.train_func import (
    evaluate_fold_on_test_set, get_or_train_fold_model, iter_outer_folds,
    select_inner_fold, train,
)
from models.models_train.training_loop import RunContext


K = 3


def _all_folds(fold_data, k=K):
    """Every (fold, split) this dataset produces at both rotation choices
    train() and train_pilot() use, so leakage tests cover both."""
    results = []
    for fold, train_val_idx, test_idx in iter_outer_folds(fold_data, k):
        for inner_fold_index in range(k - 1):
            split = select_inner_fold(fold_data, train_val_idx, test_idx, k, fold, inner_fold_index)
            results.append((fold, inner_fold_index, split))
    return results


def test_no_subject_overlaps_between_train_val_test(synthetic_fold_data):
    groups = synthetic_fold_data.groups
    for fold, inner_fold_index, split in _all_folds(synthetic_fold_data):
        train_subjects = set(groups[split.train_idx])
        val_subjects = set(groups[split.val_idx])
        test_subjects = set(groups[split.test_idx])

        assert not (train_subjects & val_subjects), f"fold={fold} inner={inner_fold_index}: train/val overlap"
        assert not (train_subjects & test_subjects), f"fold={fold} inner={inner_fold_index}: train/test overlap"
        assert not (val_subjects & test_subjects), f"fold={fold} inner={inner_fold_index}: val/test overlap"


def test_fold_split_indices_are_disjoint_and_within_range(synthetic_fold_data):
    n = len(synthetic_fold_data.targets)
    for fold, inner_fold_index, split in _all_folds(synthetic_fold_data):
        all_idx = set(split.train_idx) | set(split.val_idx) | set(split.test_idx)
        assert len(split.train_idx) + len(split.val_idx) + len(split.test_idx) == len(all_idx), \
            "train/val/test indices must be pairwise disjoint"
        assert all(0 <= i < n for i in all_idx)


def test_k_too_small_raises_rather_than_silently_misbehaving(synthetic_fold_data):
    """k=2 makes inner_k = k - 1 = 1, but StratifiedGroupKFold requires
    n_splits >= 2 -- documents the pipeline's actual minimum valid k."""
    with pytest.raises(ValueError):
        for fold, train_val_idx, test_idx in iter_outer_folds(synthetic_fold_data, k=2):
            select_inner_fold(synthetic_fold_data, train_val_idx, test_idx, k=2, fold=fold, inner_fold_index=0)


def test_checkpoint_reuse_gives_identical_metrics_to_fresh_training(
        tiny_model_spec, synthetic_fold_data, fast_training_settings, scratch_dirs):
    """Regression test for the bug this session found: a model reloaded
    from a checkpoint defaults to train() mode until .eval() is called
    explicitly, which (if missing) silently degrades test-time performance
    via BatchNorm/Dropout. Trains once, evaluates; reloads that same
    checkpoint into a fresh model via get_or_train_fold_model, evaluates
    again; asserts identical metrics."""
    ctx = RunContext(dataset_name='SYNTH', task='synthetic', model_spec=tiny_model_spec,
                      iter=0, seed=0, dirs=scratch_dirs, k=K)
    best_params = {'batch size': 16, 'learning rate': 1e-3, 'L2 weight decay': 0.0}

    for fold, train_val_idx, test_idx in iter_outer_folds(synthetic_fold_data, K):
        split = select_inner_fold(synthetic_fold_data, train_val_idx, test_idx, K, fold, inner_fold_index=fold % (K - 1))
        train_loader = DataLoader(synthetic_fold_data.dataset.__class__(
            *[t[split.train_idx] for t in synthetic_fold_data.dataset.tensors]), batch_size=16, shuffle=True, drop_last=True)
        val_loader = DataLoader(synthetic_fold_data.dataset.__class__(
            *[t[split.val_idx] for t in synthetic_fold_data.dataset.tensors]), batch_size=16)
        test_loader = DataLoader(synthetic_fold_data.dataset.__class__(
            *[t[split.test_idx] for t in synthetic_fold_data.dataset.tensors]), batch_size=16)
        test_groups = synthetic_fold_data.groups[split.test_idx].tolist()

        import torch
        # Same device-selection logic train() itself uses. Hardcoding 'cpu'
        # here would mismatch evaluate_fold_on_test_set's unconditional
        # `if torch.cuda.is_available(): torch.cuda.max_memory_allocated(device)`
        # on a machine that has a GPU -- that check is about the machine's
        # capability, not this specific call's device, and the real pipeline
        # never has that mismatch since its own `device` is always derived
        # from the same is_available() check.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model1, state1, trained1, _, _ = get_or_train_fold_model(ctx, fold, train_loader, val_loader, best_params, fast_training_settings, device)
        assert trained1 is True
        metrics1 = evaluate_fold_on_test_set(model1, state1, test_loader, test_groups, tiny_model_spec.name, device)

        # Second call: checkpoint now exists on disk -- must take the "reuse" branch.
        model2, state2, trained2, _, _ = get_or_train_fold_model(ctx, fold, train_loader, val_loader, best_params, fast_training_settings, device)
        assert trained2 is False, "expected the checkpoint-reuse branch on the second call"
        metrics2 = evaluate_fold_on_test_set(model2, state2, test_loader, test_groups, tiny_model_spec.name, device)

        assert metrics1.seg_acc == metrics2.seg_acc
        assert metrics1.sub_acc == metrics2.sub_acc
        break  # one fold is enough to prove the mechanism; keep this test fast


def test_finalize_run_averages_training_time_over_given_list_only(
        tiny_model_spec, scratch_dirs):
    """Direct, deterministic test of finalize_run's aggregation math --
    no real training/timing involved, so nothing here can be flaky. If
    all_avg_train_times already correctly excludes cached folds (train()'s
    job, tested separately below), the average must be exact, not diluted
    by anything: mean([10.0, 12.0]) == 11.0, not mean([10.0, 12.0, 0.0])."""
    from models.models_train.train_func import FoldMetrics, finalize_run

    ctx = RunContext(dataset_name='GENEEG', task='MCI vs HC', model_spec=tiny_model_spec,
                      iter=0, seed=0, dirs=scratch_dirs, k=3)
    fold_metrics_list = [
        FoldMetrics(seg_acc=0.8, sub_acc=0.8, sub_sens=0.8, sub_spec=0.8, sub_prec=0.8, sub_f1=0.8,
                    saliency_maps={}, avg_inference_time=0.01, peak_gpu_memory_mb=None)
        for _ in range(3)
    ]

    finalize_run(ctx, fold_metrics_list, k_fold_subjects_logs=[{}, {}, {}],
                 all_avg_train_times=[10.0, 12.0],  # only 2 of 3 folds trained this run
                 all_avg_inference_times=[0.01, 0.01, 0.01],
                 all_peak_memory=[0, 0, 0],
                 trained_this_run=True, num_params=100)

    timing_log_path = f"{scratch_dirs['training_inference_time_logs_dir']}/{tiny_model_spec.name}_MCI vs HC_timing_log.json"
    with open(timing_log_path) as f:
        log = json.load(f)
    assert log['training_time_(sec/epoch)'] == 11.0  # exactly mean([10.0, 12.0]), not diluted by a 3rd 0.0


def test_finalize_run_handles_all_folds_cached(tiny_model_spec, scratch_dirs):
    """When every fold was cached, all_avg_train_times is legitimately
    empty -- must not raise ZeroDivisionError."""
    from models.models_train.train_func import FoldMetrics, finalize_run

    ctx = RunContext(dataset_name='GENEEG', task='MCI vs HC', model_spec=tiny_model_spec,
                      iter=0, seed=0, dirs=scratch_dirs, k=1)
    fold_metrics_list = [FoldMetrics(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, {}, 0.01, None)]

    finalize_run(ctx, fold_metrics_list, k_fold_subjects_logs=[{}],
                 all_avg_train_times=[],  # nothing trained -- every fold was cached
                 all_avg_inference_times=[0.01], all_peak_memory=[0],
                 trained_this_run=False, num_params=100)

    timing_log_path = f"{scratch_dirs['training_inference_time_logs_dir']}/{tiny_model_spec.name}_MCI vs HC_timing_log.json"
    with open(timing_log_path) as f:
        log = json.load(f)
    assert log['training_time_(sec/epoch)'] is None  # trained_this_run=False -> not reported


def test_train_only_passes_freshly_trained_folds_time_to_finalize_run(
        tiny_model_spec, synthetic_fold_data, fast_training_settings, scratch_dirs, monkeypatch):
    """Regression test for the actual bug: intercepts train()'s call to
    finalize_run and checks *how many* entries all_avg_train_times has --
    not their noisy real-world values -- so a future regression that goes
    back to appending 0.0 for cached folds is caught by a length mismatch,
    not a flaky timing comparison."""
    import models.models_train.train_func as train_func_module

    captured = {}
    real_finalize_run = train_func_module.finalize_run

    def spy_finalize_run(ctx, fold_metrics_list, k_fold_subjects_logs,
                          all_avg_train_times, all_avg_inference_times, all_peak_memory,
                          trained_this_run, num_params):
        captured['all_avg_train_times'] = list(all_avg_train_times)
        return real_finalize_run(ctx, fold_metrics_list, k_fold_subjects_logs,
                                  all_avg_train_times, all_avg_inference_times, all_peak_memory,
                                  trained_this_run, num_params)

    monkeypatch.setattr(train_func_module, 'finalize_run', spy_finalize_run)

    ctx = RunContext(dataset_name='GENEEG', task='MCI vs HC', model_spec=tiny_model_spec,
                      iter=0, seed=0, dirs=scratch_dirs, k=K)
    best_params = {'batch size': 16, 'learning rate': 1e-3, 'L2 weight decay': 0.0}

    train(ctx, synthetic_fold_data, best_params, training=fast_training_settings)
    assert len(captured['all_avg_train_times']) == K, "first run: all K folds train fresh"

    # Cache 1 of K folds, retrain the rest.
    import os
    weight_files = sorted(os.listdir(scratch_dirs['weights_dir']))
    for wf in weight_files[1:]:
        os.remove(os.path.join(scratch_dirs['weights_dir'], wf))

    train(ctx, synthetic_fold_data, best_params, training=fast_training_settings)
    assert len(captured['all_avg_train_times']) == K - 1, (
        "second run: 1 fold was cached, so only K-1 folds' training times "
        "should be passed to finalize_run -- not K entries with a 0.0 for the cached one"
    )