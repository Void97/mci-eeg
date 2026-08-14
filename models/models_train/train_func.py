"""K-fold cross-validated training and hyperparameter pilot search.

The per-epoch training/evaluation mechanics live in
models_train.training_loop (framework-level, model-agnostic); the
interpretability outputs (saliency maps, t-SNE plots) live in
utils.interpretation alongside the PSD/topomap plotting they're the same
kind of artifact as. This module is just the k-fold CV orchestration:
splitting, checkpointing, calling those two, and saving logs.
"""

import json
import logging
import os
import time

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


def train(ctx: RunContext, fold_data: FoldData, best_params: dict, training: dict):
    dataset_name, task, iter_, seed, dirs, k = (
        ctx.dataset_name, ctx.task, ctx.iter, ctx.seed, ctx.dirs, ctx.k
    )
    model_spec = ctx.model_spec
    model_name = model_spec.name
    dataset, samples, targets, groups = (
        fold_data.dataset, fold_data.samples, fold_data.targets, fold_data.groups
    )

    subjects_to_label = {}
    k_fold_subjects_logs = []

    set_seed(seed)
    for subject, label in zip(groups, targets):
        subjects_to_label[subject] = label

    all_avg_train_times = []
    all_avg_inference_times = []
    all_peak_memory = []
    trained_this_run = False

    all_seg_acc = []
    all_sub_acc = []
    all_sub_prec = []
    all_sub_sens = []
    all_sub_spec = []
    all_sub_f1 = []

    saliency_maps = None

    unique_subjects = np.unique(groups)
    np.random.shuffle(unique_subjects)  # shuffle subjects randomly
    outer_kf = StratifiedGroupKFold(n_splits=int(k), shuffle=False)
    for fold, (train_val_idx, test_idx) in enumerate(outer_kf.split(samples, targets, groups=groups)):

        logger.info('Fold %d: %d train / %d test samples', fold + 1, len(train_val_idx), len(test_idx))
        train_val_groups = groups[train_val_idx]
        test_groups = groups[test_idx].tolist()

        logger.debug('Global K-fold overlaps check')
        test_overlaps(fold, train_val_groups, test_groups)

        inner_k = int(k) - 1
        inner_kf = StratifiedGroupKFold(n_splits=inner_k, shuffle=False)
        target_inner_fold = fold % inner_k
        train_idx, val_idx = None, None

        for inner_fold, (tr_i, val_i) in enumerate(
                inner_kf.split(samples[train_val_idx], targets[train_val_idx],
                                groups=groups[train_val_idx])):

            if inner_fold == target_inner_fold:
                train_idx, val_idx = tr_i, val_i
                break

        if train_idx is None or val_idx is None:
            raise RuntimeError("Inner K-fold splitting failed to produce train/val indeces.")
        logger.debug('Inner K-fold overlaps check')

        test_overlaps(fold,
                      groups[train_val_idx][train_idx],
                      groups[train_val_idx][val_idx])

        one_fold_subject_logs = count_fold_subjects(groups, train_val_idx, train_idx, val_idx, test_idx, subjects_to_label, fold)
        k_fold_subjects_logs.append(one_fold_subject_logs)

        train_subset = Subset(dataset, train_val_idx[train_idx])
        val_subset = Subset(dataset, train_val_idx[val_idx])
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=best_params['batch size'],
                                   shuffle=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=training['val_batch_size'],
                                 shuffle=False, drop_last=False,
                                 num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_subset, batch_size=training['test_batch_size'],
                                  shuffle=False, drop_last=False,
                                  num_workers=0, pin_memory=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.debug("cuda available: %s", torch.cuda.is_available())
        logger.debug("device count: %s", torch.cuda.device_count())
        if torch.cuda.is_available():
            logger.debug("name: %s", torch.cuda.get_device_name(0))
            logger.debug("torch: %s cuda build: %s", torch.__version__, torch.version.cuda)

        best_model_dir = dirs['weights_dir']
        os.makedirs(best_model_dir, exist_ok=True)
        best_model_file_path = os.path.join(
            best_model_dir, f'{dataset_name}_{task}_{model_name}_iteration_{iter_}_fold_{str(fold)}_best_weights.pth'
        )

        model = model_spec.cls(**model_spec.kwargs, tsne=False).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info('Model has %s trainable parameters', f'{num_params:,}')
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        if os.path.exists(best_model_file_path):
            logger.info("Found existing weights for %s %s %s iteration %s fold %d. "
                        "Skipping the training and validation parts",
                        dataset_name, task, model_name, iter_, fold + 1)
            best_model_state = torch.load(best_model_file_path, map_location=device)
            if model_name == 'EEG_Conformer':
                ckpt_ch = best_model_state['0.shallownet.1.weight'].shape[2]
                expected_ch = model_spec.kwargs['num_channels']
                assert ckpt_ch == expected_ch, (
                    f"Checkpoint has spatial kernel ch={ckpt_ch} but dataset has {expected_ch} channels. "
                    f"Delete this checkpoint and retrain."
                )
            av_epoch_train_time = 0.0
            all_avg_train_times.append(av_epoch_train_time)
        else:
            trained_this_run = True
            optimizer, criterion = build_optimizer_and_criterion(
                model, model_spec, lr=best_params['learning rate'], weight_decay=best_params['L2 weight decay']
            )

            best_val_acc = 0
            counter = 0
            epoch_training_times = []
            for epoch in range(training['full_train_max_epochs']):
                epoch_start = time.perf_counter()

                train_loss, train_acc = run_train_epoch(model, train_loader, optimizer, criterion, device)

                val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
                val_f1 = (f1_score(val_labels, val_preds) if model_spec.kwargs['num_classes'] == 2
                          else f1_score(val_labels, val_preds, average='weighted'))

                epoch_end = time.perf_counter()
                epoch_training_times.append(epoch_end - epoch_start)

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
            all_avg_train_times.append(av_epoch_train_time)

        save_tsne_plot(model_spec, best_model_state, test_loader, dirs['tsne_dir'],
                       dataset_name, task, iter_, fold, device,
                       n_iter=training['tsne_n_iter'], random_state=training['tsne_random_state'])

        # Test with the best weights.
        model.load_state_dict(best_model_state)
        model.eval()
        # Explicit eval() here matters most for the "reuse existing checkpoint" branch above:
        # that branch never calls model.train()/model.eval() itself, so without this line the
        # model stays in the default training-mode state from instantiation (BatchNorm uses
        # per-batch stats, Dropout stays active) for the entire test pass below.

        metrics_device = device
        if model_name == 'SzHNN':
            metrics_device = torch.device('cpu')
        model.to(metrics_device)

        test_preds_fold, test_labels_fold, test_roc, avg_inference_time = run_test_inference(
            model, test_loader, metrics_device
        )
        all_avg_inference_times.append(avg_inference_time)
        logger.info("Inference time for Fold %d: %.4f sec", fold + 1, avg_inference_time)

        saliency_maps = compute_saliency(model, metrics_device, test_loader, test_preds_fold, test_labels_fold)

        seg_acc = accuracy_score(test_labels_fold, test_preds_fold)
        sub_acc, sub_sens, sub_spec, sub_prec, sub_f1 = subject_wise_metrics(test_labels_fold, test_preds_fold, test_groups)
        all_seg_acc.append(seg_acc)
        all_sub_acc.append(sub_acc)
        all_sub_sens.append(sub_sens)
        all_sub_spec.append(sub_spec)
        all_sub_prec.append(sub_prec)
        all_sub_f1.append(sub_f1)

        logger.info("Fold %d\nTest Accuracy (seg.): %.4f\nTest Accuracy (sub.): %.4f\n"
                    "Test Sensitivity: %.4f\nTest Specificity: %.4f\nTest Precision: %.4f\nTest F1: %.4f",
                    fold + 1, seg_acc, sub_acc, sub_sens, sub_spec, sub_prec,
                    f1_score(test_labels_fold, test_preds_fold))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            logger.info("GPU memory usage for Fold %d: %.2f MB", fold + 1, peak_gpu_memory_mb)
        else:
            peak_gpu_memory_mb = None
            logger.info("No GPU used.")
        all_peak_memory.append(peak_gpu_memory_mb if peak_gpu_memory_mb is not None else 0)

    overall_avg_train_time = sum(all_avg_train_times) / len(all_avg_train_times)
    overall_avg_inference_time = sum(all_avg_inference_times) / len(all_avg_inference_times)
    average_peak_gpu_memory = sum(all_peak_memory) / len(all_peak_memory)
    logger.info("=== Overall Averages Across %d Folds ===", k)
    logger.info("Average training time per epoch: %.2f seconds", overall_avg_train_time)
    logger.info("Average inference time: %.4f sec", overall_avg_inference_time)
    logger.info("Average peak GPU memory: %s MB", average_peak_gpu_memory)
    update_timing_log(dirs, model_name, task, k, num_params, trained_this_run,
                       overall_avg_train_time, overall_avg_inference_time, average_peak_gpu_memory)

    av_fold_seg_acc = np.round(np.mean(all_seg_acc), 4)
    av_fold_sub_acc = np.round(np.mean(all_sub_acc), 4)
    av_fold_sub_sens = np.round(np.mean(all_sub_sens), 4)
    av_fold_sub_spec = np.round(np.mean(all_sub_spec), 4)
    av_fold_sub_prec = np.round(np.mean(all_sub_prec), 4)
    av_fold_sub_f1 = np.round(np.mean(all_sub_f1), 4)

    logger.info("=== Overall Metrics for %s %s %s iteration %s across %d Folds ===\n"
                "Accuracy (seg.): %s\nAccuracy (sub.): %s\nSensitivity: %s\n"
                "Specificity: %s\nPrecision: %s\nF1: %s",
                dataset_name, task, model_name, iter_, k,
                av_fold_seg_acc, av_fold_sub_acc, av_fold_sub_sens,
                av_fold_sub_spec, av_fold_sub_prec, av_fold_sub_f1)

    log_dir = os.path.join(dirs['k_fold_dist_logs_dir'], f'{dataset_name}_{task}_{model_name}_iteration_{iter_}_kfold_subjects_log.json')
    with open(log_dir, 'w') as f:
        json.dump(k_fold_subjects_logs, f, indent=2)

    return av_fold_seg_acc, av_fold_sub_acc, av_fold_sub_sens, av_fold_sub_spec, av_fold_sub_prec, av_fold_sub_f1, saliency_maps







def train_pilot(model_spec: ModelSpec, seed, k, fold_data: FoldData,
                 hyperparam_grid: dict, training: dict):
    dataset, samples, targets, groups = (
        fold_data.dataset, fold_data.samples, fold_data.targets, fold_data.groups
    )
    model_name = model_spec.name

    best_params = {'val_loss': float('inf'), 'val_acc': 0}
    set_seed(seed)

    outer_kf = StratifiedGroupKFold(n_splits=k, shuffle=False)
    logger.info('Pilot training begins')
    for batch_size in hyperparam_grid['batch_sizes']:
        for lr in hyperparam_grid['learning_rates']:
            for l2 in hyperparam_grid['l2_weight_decays']:
                logger.info('Batch size: %s, Learning rate: %s, Weight decay: %s', batch_size, lr, l2)

                v_p_all, v_l_all = [], []
                for fold, (train_val_idx, test_idx) in enumerate(
                        outer_kf.split(samples, targets, groups=groups)
                ):

                    test_overlaps(fold, groups[train_val_idx], groups[test_idx])

                    inner_kf = StratifiedGroupKFold(n_splits=int(k) - 1, shuffle=False)
                    train_idx, val_idx = next(
                        inner_kf.split(samples[train_val_idx], targets[train_val_idx],
                                        groups=groups[train_val_idx])
                    )
                    train_idx, val_idx = train_val_idx[train_idx], train_val_idx[val_idx]

                    logger.debug('Fold %d: %d train / %d validation samples', fold + 1, len(train_idx), len(val_idx))
                    test_overlaps(fold, groups[train_idx], groups[val_idx])

                    train_subidx = subsamples(seed, train_idx, groups, fraction=training['subsample_fraction'])
                    train_subset = Subset(dataset, train_subidx)
                    val_subset = Subset(dataset, val_idx)
                    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
                    val_loader = DataLoader(val_subset, batch_size=training['val_batch_size'], shuffle=True, drop_last=False)

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    if fold == 0:
                        logger.debug("cuda available: %s", torch.cuda.is_available())
                        logger.debug("device count: %s", torch.cuda.device_count())
                        if torch.cuda.is_available():
                            logger.debug("name: %s", torch.cuda.get_device_name(0))
                        logger.debug("torch: %s cuda build: %s", torch.__version__, torch.version.cuda)

                    model = model_spec.cls(**model_spec.kwargs).to(device)
                    optimizer, criterion = build_optimizer_and_criterion(model, model_spec, lr=lr, weight_decay=l2)

                    val_loss_folds = []
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

                val_loss_av = np.mean(val_loss_folds)
                val_acc_seg = accuracy_score(v_l_all, v_p_all)
                logger.info("Total Validation Accuracy (seg.): %s", val_acc_seg)

                if val_acc_seg > best_params['val_acc']:
                    best_params['model'] = model_name
                    best_params['val_loss'] = val_loss_av
                    best_params['val_acc'] = val_acc_seg
                    best_params['batch size'] = batch_size
                    best_params['learning rate'] = lr
                    best_params['L2 weight decay'] = l2

    logger.info('The best parameters have been found!')
    logger.info("Model: %s \nLoss: %s \nBatch size: %s \nLearning rate: %s \nL2: %s",
                model_name, best_params['val_loss'], best_params['batch size'],
                best_params['learning rate'], best_params['L2 weight decay'])
    return best_params