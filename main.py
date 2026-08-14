import argparse
import json
import os
import captum
import numpy as np
import pandas as pd
import torch
import random
import yaml
from dataclasses import replace

from utils.EEG_preprocess import preprocess, load_preprocessed
from models.models_base.models_list import models_list, single_model, MODEL_NAMES
from models.models_train.train_func import train, train_pilot
from models.models_train.training_loop import RunContext, FoldData
from captum.attr import Saliency
from collections import Counter
from torch.utils.data import TensorDataset


from main_func import *
from utils.interpretation import *

DATASET_NAMES = ['ADvsFTDvsHC', 'GENEEG', 'CAUEEG', 'MCIvsHC']


def parse_args():
    parser = argparse.ArgumentParser(description="Run the EEG benchmarking pipeline.")
    parser.add_argument(
        '--dataset', required=True, choices=DATASET_NAMES,
        help="Which dataset to benchmark.",
    )
    parser.add_argument(
        '--model', required=True, choices=[*MODEL_NAMES, 'all'],
        help="Which model to run, or 'all' to run every model in the registry.",
    )
    parser.add_argument(
        '--config', default='configs/benchmark.yaml',
        help="Path to the YAML config with the non-identity settings (sfreq, "
             "k-folds, hyperparameter grid, etc.). Default: configs/benchmark.yaml",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main(args, cfg):

    sfreq = cfg['sfreq']
    dataset_name = args.dataset
    need_preprocessing = cfg['need_preprocessing']

    results_subj = []
    freq_bands = cfg['freq_bands']
    k = cfg['k_folds']
    n_repeats = cfg['n_repeats']

    dataset_path, preprocessed_dir, ch_num, ch_names, show_channel, tasks, metadata, subjects_list, labels_list = build(dataset_name)
    dirs = makedirs(dataset_name)

    if need_preprocessing:
        if dataset_name == 'GENEEG':
            ch_types = ['eeg'] * len(ch_names)
            ch_num = len(ch_names)

            preprocess(metadata, dataset_path, preprocessed_dir,
                    dataset_name, sfreq,
                    ch_names, ch_types)
        elif dataset_name == 'MCIvsHC' or dataset_name == 'ADvsFTDvsHC':
            preprocess(metadata, dataset_path, preprocessed_dir,
                    dataset_name, new_sfreq=200)
        else:
            preprocess(metadata, dataset_path, preprocessed_dir,
                    dataset_name)

    for task in tasks:

        random_seeds = generate_seeds('benchmark', n_repeats, dataset_name, task)

        labels_map = labels_mapping(dataset_name, task)
        filtered_subjects, filtered_labels = filter_data(subjects_list, labels_list, labels_map)


        label_dict = dict(zip(filtered_subjects, filtered_labels))
        samples, targets, groups = load_preprocessed(preprocessed_dir, labels_map, label_dict)

        if args.model == 'all':
            models = models_list(num_classes=2, num_channels=ch_num, time_points=samples.shape[2])
        else:
            models = [single_model(num_classes=2, num_channels=ch_num, time_points=samples.shape[2], model_name=args.model)]

        for model in models:
            seg_acc_all = []
            acc_all = []
            prec_all = []
            sens_all = []
            spec_all = []
            f1_all = []


            for i in range(n_repeats):
                print(f"Iteration {i+1}")
                seed = random_seeds[i]
                print(f"Random seed: {seed}")
                metrics_exist_file = os.path.join(dirs['metrics_logs_dir'], f"{dataset_name}_{task}_{model.name}_iteration_{i}_metrics.json")
                if os.path.exists(metrics_exist_file):

                    print(f"{dataset_name}_{task}_{model.name}_iteration_{i+1} metrics exist! No need to train.")
                    with open(metrics_exist_file, 'r') as f:
                        metrics = json.load(f)

                    seg_acc, sub_acc, sub_sens, sub_spec, sub_prec, sub_f1, = (
                        metrics['Accuracy (seg.)'],
                        metrics['Accuracy (subj.)'],
                        metrics['Sensitivity (subj.)'],
                        metrics['Specificity (subj.)'],
                        metrics['Precision (subj.)'],
                        metrics['F1 Score (subj.)']
                    )
                    seg_acc_all.append(seg_acc)
                    acc_all.append(sub_acc)
                    sens_all.append(sub_sens)
                    spec_all.append(sub_spec)
                    prec_all.append(sub_prec)
                    f1_all.append(sub_f1)

                else:
                    print(f"{dataset_name}_{task}_{model.name}_iteration_{i+1} metrics doesn't exist. Let's train!")

                    if model.name == 'Oh_CNN' or model.name == 'SzHNN' or model.name == 'EEG_Deformer':
                        full_samples = torch.from_numpy(samples).float()
                    else:
                        full_samples = torch.from_numpy(samples).unsqueeze(1).float()   # one big tensor

                    task_model = replace(model, kwargs={**model.kwargs, 'num_classes': 3 if task == '3-class' else 2})

                    full_targets = torch.from_numpy(targets).float()    # one big tensor
                    full_dataset = TensorDataset(full_samples, full_targets)
                    fold_data = FoldData(dataset=full_dataset, samples=full_samples.numpy(),
                                         targets=full_targets.numpy(), groups=groups)
                    ctx = RunContext(dataset_name=dataset_name, task=task, model_spec=task_model,
                                      iter=i, seed=seed, dirs=dirs, k=k)

                    ###############################################################################################################################

                    params_file_path = os.path.join(dirs['best_params_logs_dir'], f"{dataset_name}_{task}_{model.name}_iteration_{i}_best_params_file.json")

                    if os.path.exists(params_file_path):
                        print(f"Best parameters file for the iteration {i+1} of {task} and {model.name} exists! No need the pilot training.")
                        best_params_file = os.path.join(dirs['best_params_logs_dir'], f"{dataset_name}_{task}_{model.name}_iteration_{i}_best_params_file.json")
                        with open(best_params_file, 'r') as f:
                            best_params = json.load(f)

                        print(f"Task: {task}. Model: {model.name}. Iteration: {i + 1}")
                        seg_acc, sub_acc, sub_sens, sub_spec, sub_prec, sub_f1, gradient = train(
                                ctx, fold_data, best_params, training=cfg['training']
                            )

                        seg_acc_all.append(seg_acc)
                        acc_all.append(sub_acc)
                        sens_all.append(sub_sens)
                        spec_all.append(sub_spec)
                        prec_all.append(sub_prec)
                        f1_all.append(sub_f1)


                        save_iter_metrics(seg_acc, sub_acc, sub_sens, sub_spec, sub_prec, sub_f1, dataset_name, task, model.name, i, dirs['metrics_logs_dir'])

                        xai = Interpretation(dirs['topomaps_dir'], sfreq)
                        xai.plot_psd(dataset_name, model.name, task, gradient, i, 40)
                        xai.plot_psd_topo(dataset_name, model.name, task, gradient, freq_bands, ch_names, show_channel, i)


                    else:
                        print(f"Best parameters file for the iteration {i+1} of {task} and {model.name} doesn't exists! Needs pilot training.")
                        print(f"The pilot training begins. \nTask: {task}. Model: {model.name}. Iteration: {i+1}.")
                        best_params = train_pilot(
                            model_spec=task_model, seed=seed, k=k, fold_data=fold_data,
                            hyperparam_grid=cfg['hyperparam_grid'], training=cfg['training']
                        )

                        best_params_file = os.path.join(dirs['best_params_logs_dir'], f"{dataset_name}_{task}_{model.name}_iteration_{i}_best_params_file.json")
                        with open(best_params_file, 'w') as f:
                            json.dump(best_params, f, indent=2)
                        print(f'The best parameters have been saved!')


                        print(f"Task: {task}. Model: {model.name}. Iteration: {i + 1}")
                        seg_acc, sub_acc, sub_sens, sub_spec, sub_prec, sub_f1, gradient = train(
                                ctx, fold_data, best_params, training=cfg['training']
                            )

                        seg_acc_all.append(seg_acc)
                        acc_all.append(sub_acc)
                        sens_all.append(sub_sens)
                        spec_all.append(sub_spec)
                        prec_all.append(sub_prec)
                        f1_all.append(sub_f1)


                        save_iter_metrics(seg_acc, sub_acc, sub_sens, sub_spec, sub_prec, sub_f1, dataset_name, task, model.name, i, dirs['metrics_logs_dir'])


                        xai = Interpretation(dirs['topomaps_dir'], sfreq)
                        xai.plot_psd(dataset_name, model.name, task, gradient, i, 40)
                        xai.plot_psd_topo(dataset_name, model.name, task, gradient, freq_bands, ch_names, show_channel, i)


            seg_acc_av, seg_acc_std = round(np.mean(seg_acc_all),4), round(np.std(seg_acc_all),4)
            acc_av, acc_std = round(np.mean(acc_all),4), round(np.std(acc_all),4)
            sens_av, sens_std = round(np.mean(sens_all),4), round(np.std(sens_all),4)
            spec_av, spec_std = round(np.mean(spec_all),4), round(np.std(spec_all),4)
            prec_av, prec_std = round(np.mean(prec_all),4), round(np.std(prec_all),4)
            f1_av, f1_std = round(np.mean(f1_all),4), round(np.std(f1_all),4)

            save_metrics_logs(dirs['metrics_logs_dir'], dataset_name, task, model.name,
                    seg_acc_av, seg_acc_std, acc_av, acc_std,
                    sens_av, sens_std,
                    spec_av, spec_std,
                    prec_av, prec_std,
                    f1_av, f1_std)

            result_subj = {
                    "Task": task,
                    "Model": model.name,
                    "Accuracy (seg.)": f'{seg_acc_av} ± {seg_acc_std}',
                    "Accuracy (subj.)": f'{acc_av} ± {acc_std}',
                    "Sensitivity": f'{sens_av} ± {sens_std}',
                    "Specificity": f'{spec_av} ± {spec_std}',
                    "Precision": f'{prec_av} ± {prec_std}',
                    "F1 Score": f'{f1_av} ± {f1_std}',
                    }


            results_subj.append(result_subj)

    metrics_subj = pd.DataFrame(results_subj)
    if not os.path.exists(dirs['metrics_dir']):
        os.makedirs(dirs['metrics_dir'])
    metrics_subj_filename = f"{dataset_name}_metrics_overall_({cfg['run_label']}).xlsx"
    metrics_subj_path = os.path.join(dirs['metrics_dir'], metrics_subj_filename)
    metrics_subj.to_excel(metrics_subj_path)
    print(f"✅ All models trained and results saved to {metrics_subj_filename}")


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args.config)
    main(args, cfg)