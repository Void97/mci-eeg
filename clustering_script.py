import argparse

from main_func_111125 import build, labels_mapping, filter_data
from models.models_base.models_list import single_model
from utils.EEG_preprocess import load_preprocessed
from clustering_utils.best_iteration import find_best_iteration
from clustering_utils.constants import BANDS
from clustering_utils.inference_saliency import inference, GRADIENT_METHODS
from clustering_utils.clustering_20260424 import Pipeline
from clustering_utils.cluster_interpretation import plot_cluster_psd_topomaps, plot_cluster_subject_psd_curves
from clustering_utils.faithfulness import FaithfulnessEvaluator, compute_spearman_consistency, EXP_KEYS


def main():
    parser = argparse.ArgumentParser(description='Subgroup clustering for EEG data')
    parser.add_argument('--dataset',  type=str, required=True)
    parser.add_argument('--gradient', type=str, default='vanilla', choices=GRADIENT_METHODS)
    args = parser.parse_args()

    dataset_name     = args.dataset
    gradient_method  = args.gradient

    _, preprocessed_dir, ch_num, ch_names, show_channel, tasks, \
        _, subjects_list, labels_list = build(dataset_name)
    print(f'Dataset {dataset_name} loaded.')

    for task in tasks:
        labels_map = labels_mapping(dataset_name, task)
        filtered_subjects, filtered_labels = filter_data(subjects_list, labels_list, labels_map)
        label_dict = dict(zip(filtered_subjects, filtered_labels))

        samples, targets, groups = load_preprocessed(preprocessed_dir, labels_map, label_dict)
        print(f'Samples: {samples.shape}  Targets: {targets.shape}')

        model          = single_model(num_classes=2, num_channels=ch_num,
                                      time_points=samples.shape[2])
        best_iteration = find_best_iteration(dataset_name, task, model['name'])
        print(f"Model: {model['name']}  best iteration: {best_iteration}")

        (gradients_list, negative_gradients_list,
         subject_ids_list, negative_subject_ids_list,
         subject_fold_map) = inference(
            dataset_name, task, model, samples, targets, groups,
            best_iteration, gradient_method=gradient_method,
        )

        clustering_pipeline = Pipeline(dataset_name, task, model['name'],
                                       best_iteration, BANDS,
                                       gradient_method=gradient_method)
        cluster_labels = clustering_pipeline.run(gradients_list, subject_ids_list)

        FaithfulnessEvaluator(BANDS, ch_names).evaluate(
            model, samples, targets, groups,
            gradients_list, cluster_labels, subject_ids_list,
            dataset_name, task, best_iteration, gradient_method,
            subject_fold_map=subject_fold_map,
        )

        results_path = 'results/clustering/'
        for exp_key in EXP_KEYS:
            compute_spearman_consistency(
                results_path, dataset_name, task, model['name'],
                best_iteration, GRADIENT_METHODS, exp_key=exp_key,
            )

        clustering_label = f'hierarchical_{gradient_method}'
        plot_cluster_psd_topomaps(
            dataset_name, task, model['name'], best_iteration,
            clustering_label, gradients_list, cluster_labels,
            ch_names, show_channel, subject_ids_list, func='topo', sfreq=200,
        )
        plot_cluster_subject_psd_curves(
            dataset_name, task, model['name'], best_iteration,
            clustering_label, gradients_list, negative_gradients_list,
            cluster_labels, ch_names, subject_ids_list,
            negative_subject_ids_list, func='curve', sfreq=200,
        )


if __name__ == '__main__':
    main()
