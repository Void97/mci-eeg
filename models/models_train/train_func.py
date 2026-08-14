"""K-fold cross-validated training and hyperparameter pilot search.

The per-epoch training/evaluation mechanics live in
models_train.training_loop (framework-level, model-agnostic); the
interpretability outputs (saliency maps, t-SNE plots) live in
utils.interpretation alongside the PSD/topomap plotting they're the same
kind of artifact as. This module is the k-fold CV orchestration:
splitting, checkpointing, calling those two, and saving logs -- decomposed
into small named steps rather than two long flat functions.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
import torch

from main_func import set_seed, subject_wise_metrics
from models.models_base.models_list import ModelSpec
from models.models_train.k_fold_logging import count_fold_subjects
from models.models_train.training_loop import (
    RunContext, FoldData, build_optimizer_and_criterion,
    run_train_epoch, evaluate, run_test_inference,
)
from utils.interpretation import compute_saliency, save_tsne_plot
from utils.tests import test_overlaps

logger = logging.getLogger(__name__)


@dataclass
class FoldSplit:
    """One fold's train/val/test indices, already resolved to absolute
    positions into a FoldData's samples/targets/groups -- callers never
    need to know about the outer/inner two-level split that produced them."""
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class FoldMetrics:
    """One fold's test-set results."""
    seg_acc: float
    sub_acc: float
    sub_sens: float
    sub_spec: float
    sub_prec: float
    sub_f1: float
    saliency_maps: dict
    avg_inference_time: float
    peak_gpu_memory_mb: Any  # float, or None if no GPU was used


def subsamples(seed, idx, groups, fraction, min_samples=1):
    random_gen = np.random.RandomState(seed)
    idx = np.asarray(idx)
    groups = groups[idx]

    sub_idx = []
    for subject in np.unique(groups):
        mask = (groups == subject)
        subject_idx = idx[mask]
        total_l = len(subject_idx)

        kept_n = max(min_samples, int(np.ceil(total_l * fraction)))
        if kept_n >= total_l:
            chosen_idx = subject_idx
        else:
            chosen_idx = random_gen.choice(subject_idx, size=kept_n, replace=False)
        sub_idx.extend(chosen_idx)
    return np.array(sub_idx)


def iter_outer_folds(fold_data: FoldData, k):
    """Yields (fold, train_val_idx, test_idx) -- the outer
    StratifiedGroupKFold split shared by train() and train_pilot(),
    with the leakage-safety check baked in so it can't be skipped."""
    outer_kf = StratifiedGroupKFold(n_splits=int(k), shuffle=False)
    for fold, (train_val_idx, test_idx) in enumerate(
            outer_kf.split(fold_data.samples, fold_data.targets, groups=fold_data.groups)):
        logger.debug('Global K-fold overlaps check')
        test_overlaps(fold, fold_data.groups[train_val_idx], fold_data.groups[test_idx])
        yield fold, train_val_idx, test_idx


def select_inner_fold(fold_data: FoldData, train_val_idx, test_idx, k, fold, inner_fold_index) -> FoldSplit:
    """Splits train_val_idx into one inner train/val fold and resolves it
    to absolute indices. `inner_fold_index` picks which of the k-1 inner
    folds to use -- train() rotates through them (fold % (k-1)) so every
    fold eventually serves as validation; train_pilot() always takes the
    first (0), a coarser/faster choice suited to sweeping many
    hyperparameter combinations."""
    samples, targets, groups = fold_data.samples, fold_data.targets, fold_data.groups
    inner_k = int(k) - 1
    inner_kf = StratifiedGroupKFold(n_splits=inner_k, shuffle=False)

    train_idx, val_idx = None, None
    for i, (tr_i, val_i) in enumerate(
            inner_kf.split(samples[train_val_idx], targets[train_val_idx], groups=groups[train_val_idx])):
        if i == inner_fold_index:
            train_idx, val_idx = tr_i, val_i
            break

    if train_idx is None or val_idx is None:
        raise RuntimeError("Inner K-fold splitting failed to produce train/val indices.")

    logger.debug('Inner K-fold overlaps check')
    test_overlaps(fold, groups[train_val_idx][train_idx], groups[train_val_idx][val_idx])

    return FoldSplit(
        train_idx=train_val_idx[train_idx],
        val_idx=train_val_idx[val_idx],
        test_idx=test_idx,
    )


def get_or_train_fold_model(ctx: RunContext, fold, train_loader, val_loader, best_params: dict, training: dict, device):
    """Loads this fold's checkpoint if one exists (skipping training
    entirely), otherwise trains from scratch with early stopping.
    Returns (model, best_model_state, trained_this_run, num_params, avg_epoch_train_time)."""
    model_spec = ctx.model_spec
    model_name = model_spec.name

    best_model_dir = ctx.dirs['weights_dir']
    os.makedirs(best_model_dir, exist_ok=True)
    best_model_file_path = os.path.join(
        best_model_dir, f'{ctx.dataset_name}_{ctx.task}_{model_name}_iteration_{ctx.iter}_fold_{fold}_best_weights.pth'
    )

    model = model_spec.cls(**model_spec.kwargs, tsne=False).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Model has %s trainable parameters', f'{num_params:,}')

    if os.path.exists(best_model_file_path):
        logger.info("Found existing weights for %s %s %s iteration %s fold %d. "
                    "Skipping the training and validation parts",
                    ctx.dataset_name, ctx.task, model_name, ctx.iter, fold + 1)
        best_model_state = torch.load(best_model_file_path, map_location=device)
        if model_name == 'EEG_Conformer':
            ckpt_ch = best_model_state['0.shallownet.1.weight'].shape[2]
            expected_ch = model_spec.kwargs['num_channels']
            assert ckpt_ch == expected_ch, (
                f"Checkpoint has spatial kernel ch={ckpt_ch} but dataset has {expected_ch} channels. "
                f"Delete this checkpoint and retrain."
            )
        return model, best_model_state, False, num_params, 0.0

    optimizer, criterion = build_optimizer_and_criterion(
        model, model_spec, lr=best_params['learning rate'], weight_decay=best_params['L2 weight decay']
    )

    best_val_acc = 0
    best_model_state = None
    counter = 0
    epoch_training_times = []
    for epoch in range(training['full_train_max_epochs']):
        epoch_start = time.perf_counter()

        train_loss, train_acc = run_train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        val_f1 = (f1_score(val_labels, val_preds) if model_spec.kwargs['num_classes'] == 2
                  else f1_score(val_labels, val_preds, average='weighted'))

        epoch_training_times.append(time.perf_counter() - epoch_start)

        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d: Train Loss: %.4f, Train Acc: %.4f, "
                        "Validation Loss: %.4f, Validation Acc: %.4f, Validation F1-score: %.4f",
                        epoch + 1, train_loss, train_acc, val_loss, val_acc, val_f1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
        if counter > training['patience']:
            logger.info("Epoch %d: Train Loss: %.4f, Train Acc: %.4f, "
                        "Validation Loss: %.4f, Validation Acc: %.4f, Validation F1-score: %.4f",
                        epoch + 1, train_loss, train_acc, val_loss, val_acc, val_f1)
            break

    torch.save(best_model_state, best_model_file_path)
    logger.info("The model is saved!")

    av_epoch_train_time = sum(epoch_training_times) / len(epoch_training_times)
    logger.info("Average training time per epoch for Fold %d: %.2f seconds", fold + 1, av_epoch_train_time)
    return model, best_model_state, True, num_params, av_epoch_train_time


def evaluate_fold_on_test_set(model, best_model_state, test_loader, test_groups, model_name, device) -> FoldMetrics:
    """Given a trained (or loaded) model, produce this fold's test-set
    metrics and saliency maps."""
    model.load_state_dict(best_model_state)
    model.eval()
    # Explicit eval() here matters most when get_or_train_fold_model reused an existing
    # checkpoint: that branch never calls model.train()/model.eval() itself, so without this
    # line the model stays in the default training-mode state from instantiation (BatchNorm
    # uses per-batch stats, Dropout stays active) for the entire test pass below.

    metrics_device = device
    if model_name == 'SzHNN':
        metrics_device = torch.device('cpu')
    model.to(metrics_device)

    test_preds, test_labels, test_roc, avg_inference_time = run_test_inference(model, test_loader, metrics_device)
    logger.info("Inference time: %.4f sec", avg_inference_time)

    saliency_maps = compute_saliency(model, metrics_device, test_loader, test_preds, test_labels)

    seg_acc = accuracy_score(test_labels, test_preds)
    sub_acc, sub_sens, sub_spec, sub_prec, sub_f1 = subject_wise_metrics(test_labels, test_preds, test_groups)

    logger.info("Test Accuracy (seg.): %.4f\nTest Accuracy (sub.): %.4f\n"
                "Test Sensitivity: %.4f\nTest Specificity: %.4f\nTest Precision: %.4f\nTest F1: %.4f",
                seg_acc, sub_acc, sub_sens, sub_spec, sub_prec, f1_score(test_labels, test_preds))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        logger.info("GPU memory usage: %.2f MB", peak_gpu_memory_mb)
    else:
        peak_gpu_memory_mb = None
        logger.info("No GPU used.")

    return FoldMetrics(seg_acc, sub_acc, sub_sens, sub_spec, sub_prec, sub_f1,
                       saliency_maps, avg_inference_time, peak_gpu_memory_mb)


def update_timing_log(dirs, model_name, task, k, num_params, trained_this_run,
                       overall_avg_train_time, overall_avg_inference_time, average_peak_gpu_memory):
    timing_log_path = os.path.join(dirs['training_inference_time_logs_dir'], f'{model_name}_{task}_timing_log.json')

    if os.path.exists(timing_log_path):
        with open(timing_log_path, 'r') as f:
            existing_log = json.load(f)
        existing_log.pop("inference_time_(ms/segment)", None)
        existing_log.pop("inference_time_(sec/segment)", None)
        if trained_this_run:
            existing_log["training_time_(sec/epoch)"] = round(overall_avg_train_time, 2)
        existing_log["inference_time_(sec)"] = round(overall_avg_inference_time, 4)
        existing_log["average_peak_gpu_memory_MB"] = round(average_peak_gpu_memory, 2)
        with open(timing_log_path, 'w') as f:
            json.dump(existing_log, f, indent=2)
        logger.info("Timing log updated (inference time) at: %s", timing_log_path)
    else:
        timing_log = {
            "model_name": model_name,
            "task": task,
            "num_folds": k,
            "num_trainable_parameters": num_params,
            "training_time_(sec/epoch)": round(overall_avg_train_time, 2) if trained_this_run else None,
            "inference_time_(sec)": round(overall_avg_inference_time, 4),
            "average_peak_gpu_memory_MB": round(average_peak_gpu_memory, 2),
        }
        with open(timing_log_path, 'w') as f:
            json.dump(timing_log, f, indent=2)
        logger.info("Timing log saved to: %s", timing_log_path)


def finalize_run(ctx: RunContext, fold_metrics_list, k_fold_subjects_logs,
                  all_avg_train_times, all_avg_inference_times, all_peak_memory,
                  trained_this_run, num_params):
    dataset_name, task, iter_, dirs, k = ctx.dataset_name, ctx.task, ctx.iter, ctx.dirs, ctx.k
    model_name = ctx.model_spec.name

    overall_avg_train_time = sum(all_avg_train_times) / len(all_avg_train_times)
    overall_avg_inference_time = sum(all_avg_inference_times) / len(all_avg_inference_times)
    average_peak_gpu_memory = sum(all_peak_memory) / len(all_peak_memory)
    logger.info("=== Overall Averages Across %d Folds ===", k)
    logger.info("Average training time per epoch: %.2f seconds", overall_avg_train_time)
    logger.info("Average inference time: %.4f sec", overall_avg_inference_time)
    logger.info("Average peak GPU memory: %s MB", average_peak_gpu_memory)
    update_timing_log(dirs, model_name, task, k, num_params, trained_this_run,
                       overall_avg_train_time, overall_avg_inference_time, average_peak_gpu_memory)

    av_fold_seg_acc = np.round(np.mean([m.seg_acc for m in fold_metrics_list]), 4)
    av_fold_sub_acc = np.round(np.mean([m.sub_acc for m in fold_metrics_list]), 4)
    av_fold_sub_sens = np.round(np.mean([m.sub_sens for m in fold_metrics_list]), 4)
    av_fold_sub_spec = np.round(np.mean([m.sub_spec for m in fold_metrics_list]), 4)
    av_fold_sub_prec = np.round(np.mean([m.sub_prec for m in fold_metrics_list]), 4)
    av_fold_sub_f1 = np.round(np.mean([m.sub_f1 for m in fold_metrics_list]), 4)

    logger.info("=== Overall Metrics for %s %s %s iteration %s across %d Folds ===\n"
                "Accuracy (seg.): %s\nAccuracy (sub.): %s\nSensitivity: %s\n"
                "Specificity: %s\nPrecision: %s\nF1: %s",
                dataset_name, task, model_name, iter_, k,
                av_fold_seg_acc, av_fold_sub_acc, av_fold_sub_sens,
                av_fold_sub_spec, av_fold_sub_prec, av_fold_sub_f1)

    log_path = os.path.join(dirs['k_fold_dist_logs_dir'], f'{dataset_name}_{task}_{model_name}_iteration_{iter_}_kfold_subjects_log.json')
    with open(log_path, 'w') as f:
        json.dump(k_fold_subjects_logs, f, indent=2)

    # Matches this pipeline's existing behavior: only the last fold's saliency maps are
    # returned/used downstream, not an aggregate across folds.
    saliency_maps = fold_metrics_list[-1].saliency_maps

    return av_fold_seg_acc, av_fold_sub_acc, av_fold_sub_sens, av_fold_sub_spec, av_fold_sub_prec, av_fold_sub_f1, saliency_maps


def train(ctx: RunContext, fold_data: FoldData, best_params: dict, training: dict):
    set_seed(ctx.seed)
    subjects_to_label = dict(zip(fold_data.groups, fold_data.targets))

    k_fold_subjects_logs = []
    fold_metrics_list = []
    all_avg_train_times, all_avg_inference_times, all_peak_memory = [], [], []
    trained_this_run = False
    num_params = None

    for fold, train_val_idx, test_idx in iter_outer_folds(fold_data, ctx.k):
        logger.info('Fold %d: %d train / %d test samples', fold + 1, len(train_val_idx), len(test_idx))

        inner_k = int(ctx.k) - 1
        split = select_inner_fold(fold_data, train_val_idx, test_idx, ctx.k, fold, inner_fold_index=fold % inner_k)

        k_fold_subjects_logs.append(
            count_fold_subjects(fold_data.groups, split.train_idx, split.val_idx, split.test_idx, subjects_to_label, fold)
        )

        train_loader = DataLoader(Subset(fold_data.dataset, split.train_idx), batch_size=best_params['batch size'],
                                   shuffle=True, drop_last=True)
        val_loader = DataLoader(Subset(fold_data.dataset, split.val_idx), batch_size=training['val_batch_size'],
                                 shuffle=False, drop_last=False, num_workers=0, pin_memory=False)
        test_loader = DataLoader(Subset(fold_data.dataset, split.test_idx), batch_size=training['test_batch_size'],
                                  shuffle=False, drop_last=False, num_workers=0, pin_memory=False)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.debug("cuda available: %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            logger.debug("name: %s", torch.cuda.get_device_name(0))

        model, best_model_state, trained_here, num_params, av_epoch_train_time = get_or_train_fold_model(
            ctx, fold, train_loader, val_loader, best_params, training, device
        )
        trained_this_run = trained_this_run or trained_here
        all_avg_train_times.append(av_epoch_train_time)

        save_tsne_plot(ctx.model_spec, best_model_state, test_loader, ctx.dirs['tsne_dir'],
                       ctx.dataset_name, ctx.task, ctx.iter, fold, device,
                       n_iter=training['tsne_n_iter'], random_state=training['tsne_random_state'])

        test_groups = fold_data.groups[split.test_idx].tolist()
        fold_metrics = evaluate_fold_on_test_set(model, best_model_state, test_loader, test_groups, ctx.model_spec.name, device)
        fold_metrics_list.append(fold_metrics)
        all_avg_inference_times.append(fold_metrics.avg_inference_time)
        all_peak_memory.append(fold_metrics.peak_gpu_memory_mb if fold_metrics.peak_gpu_memory_mb is not None else 0)

    return finalize_run(ctx, fold_metrics_list, k_fold_subjects_logs,
                         all_avg_train_times, all_avg_inference_times, all_peak_memory,
                         trained_this_run, num_params)


def evaluate_hyperparams(model_spec: ModelSpec, fold_data: FoldData, k, seed, batch_size, lr, l2, training: dict) -> dict:
    """Trains and validates one hyperparameter combination across all k
    folds (train every epoch; validate once per fold after training
    completes -- see select_inner_fold's docstring for why this differs
    from train()'s per-epoch validation). Returns {'val_loss', 'val_acc'}
    aggregated across folds."""
    val_loss_folds = []
    v_p_all, v_l_all = [], []

    for fold, train_val_idx, test_idx in iter_outer_folds(fold_data, k):
        split = select_inner_fold(fold_data, train_val_idx, test_idx, k, fold, inner_fold_index=0)
        logger.debug('Fold %d: %d train / %d validation samples', fold + 1, len(split.train_idx), len(split.val_idx))

        train_subidx = subsamples(seed, split.train_idx, fold_data.groups, fraction=training['subsample_fraction'])
        train_loader = DataLoader(Subset(fold_data.dataset, train_subidx), batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(Subset(fold_data.dataset, split.val_idx), batch_size=training['val_batch_size'], shuffle=True, drop_last=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if fold == 0:
            logger.debug("cuda available: %s", torch.cuda.is_available())
            if torch.cuda.is_available():
                logger.debug("name: %s", torch.cuda.get_device_name(0))

        model = model_spec.cls(**model_spec.kwargs).to(device)
        optimizer, criterion = build_optimizer_and_criterion(model, model_spec, lr=lr, weight_decay=l2)

        # Training runs every epoch; validation intentionally runs once per fold
        # after all epochs complete (not per-epoch), matching the pilot search's
        # coarser, faster evaluation of a hyperparameter combination.
        for epoch in range(training['pilot_max_epochs']):
            train_loss, train_acc = run_train_epoch(model, train_loader, optimizer, criterion, device)
            if (epoch + 1) % 10 == 0:
                logger.info("Epoch %d: Train Loss: %.4f, Train Acc: %.4f", epoch + 1, train_loss, train_acc)

        val_loss, val_acc, v_p, v_l = evaluate(model, val_loader, criterion, device)
        val_loss_folds.append(val_loss)
        v_p_all.extend(v_p)
        v_l_all.extend(v_l)
        logger.info('Validation Loss: %s, Validation Accuracy %s', val_loss, val_acc)

    val_acc_seg = accuracy_score(v_l_all, v_p_all)
    logger.info("Total Validation Accuracy (seg.): %s", val_acc_seg)
    return {'val_loss': np.mean(val_loss_folds), 'val_acc': val_acc_seg}


def train_pilot(model_spec: ModelSpec, seed, k, fold_data: FoldData, hyperparam_grid: dict, training: dict):
    best_params = {'val_loss': float('inf'), 'val_acc': 0}
    set_seed(seed)

    logger.info('Pilot training begins')
    for batch_size in hyperparam_grid['batch_sizes']:
        for lr in hyperparam_grid['learning_rates']:
            for l2 in hyperparam_grid['l2_weight_decays']:
                logger.info('Batch size: %s, Learning rate: %s, Weight decay: %s', batch_size, lr, l2)
                result = evaluate_hyperparams(model_spec, fold_data, k, seed, batch_size, lr, l2, training)

                if result['val_acc'] > best_params['val_acc']:
                    best_params['model'] = model_spec.name
                    best_params['val_loss'] = result['val_loss']
                    best_params['val_acc'] = result['val_acc']
                    best_params['batch size'] = batch_size
                    best_params['learning rate'] = lr
                    best_params['L2 weight decay'] = l2

    logger.info('The best parameters have been found!')
    logger.info("Model: %s \nLoss: %s \nBatch size: %s \nLearning rate: %s \nL2: %s",
                model_spec.name, best_params['val_loss'], best_params['batch size'],
                best_params['learning rate'], best_params['L2 weight decay'])
    return best_params