import os
import json
import numpy as np
import torch
import argparse

from torch.utils.data import TensorDataset

from main_func_111125 import build, generate_seeds, labels_mapping, filter_data
from models.models_base.models_list import single_model
from utils.EEG_preprocess import load_preprocessed
from clustering_utils.best_iteration import find_best_iteration
from clustering_utils.inference_saliency import inference, GRADIENT_METHODS
from clustering_utils.clustering_20260424 import Pipeline
from clustering_utils.cluster_interpretation import plot_cluster_psd_topomaps, plot_cluster_subject_psd_curves

def main():

    parser = argparse.ArgumentParser(description='Subgroup Clustering for MCI EEG data')
    parser.add_argument('--dataset', type=str, help='Name of the dataset to use')
    parser.add_argument('--gradient', type=str, default='vanilla', choices=GRADIENT_METHODS,
                        help=f'Gradient attribution method for saliency maps. '
                             f'Available: {", ".join(GRADIENT_METHODS)}')

    args = parser.parse_args()

    dataset_name = args.dataset
    gradient_method = args.gradient

    bands = {
        'delta': [0, 4],
        'theta': [4, 8],
        'alpha': [8, 12],
        'beta': [12, 30],
        'gamma': [30, 45]
    }

    dataset_path, preprocessed_dir, ch_num, ch_names, show_channel, tasks, metadata, subjects_list, labels_list = build(dataset_name)
    print(f'The dataset {dataset_name} is loaded successfully!')

    results_path = 'results/clustering/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    n_repeats = 5
    for task in tasks:

        random_seeds = generate_seeds('clustering', n_repeats, dataset_name, task)
        labels_map = labels_mapping(dataset_name, task)
        filtered_subjects, filtered_labels = filter_data(subjects_list, labels_list, labels_map)

        label_dict = dict(zip(filtered_subjects, filtered_labels))

        samples, targets, groups = load_preprocessed(preprocessed_dir, labels_map, label_dict)
        print(f'Samples size: {samples.shape}, Targets size: {targets.shape}, Groups size: {groups.shape}')
        print(groups)
        print(f"Types - Samples: {type(samples)}, Targets: {type(targets)}, Groups: {type(groups)}")


        model = single_model(num_classes=2, num_channels=ch_num, time_points=samples.shape[2])
        print(f"The model {model['name']} is built successfully!")

        best_iteration = find_best_iteration(dataset_name, task, model['name'])
        best_params_path = f'results/best params logs/{dataset_name}/{dataset_name}_{task}_{model["name"]}_iteration_{best_iteration}_best_params_file.json'
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
        print(f"Best params for dataset: {dataset_name}, task: {task}, model: {model['name']} - {best_params}")
        print('---------------------------------------------')

        gradients_list, negative_gradients_list, subject_ids_list, negative_subject_ids_list = inference(dataset_name, task, model,
                                                                                                        samples, targets, groups,
                                                                                                        best_iteration, best_params, random_seeds,
                                                                                                        gradient_method=gradient_method)

        clustering_pipeline = Pipeline(dataset_name, task, model['name'], best_iteration, bands, gradient_method=gradient_method)
        cluster_labels = clustering_pipeline.run(gradients_list, subject_ids_list)

        clustering_label = f'hierarchical_{gradient_method}'
        plot_cluster_psd_topomaps(dataset_name, task, model['name'], best_iteration,
                                  clustering_label, gradients_list, cluster_labels,
                                  ch_names, show_channel, subject_ids_list, func='topo', sfreq=200)
        plot_cluster_subject_psd_curves(dataset_name, task, model['name'], best_iteration,
                                        clustering_label, gradients_list, negative_gradients_list,
                                        cluster_labels, ch_names, subject_ids_list,
                                        negative_subject_ids_list, func='curve', sfreq=200)

if __name__ == "__main__":
    main()