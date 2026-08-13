import argparse

from main_func import build, labels_mapping, filter_data
from models.models_base.models_list import single_model, MODEL_NAMES
from utils.EEG_preprocess import load_preprocessed
from clustering_utils.best_iteration import find_best_iteration
from clustering_utils.constants import BANDS, LINKAGE_METHODS
from clustering_utils.inference_saliency import GRADIENT_METHODS
from clustering_utils.faithfulness import EXP_KEYS
from clustering_utils.cluster_interpretation import plot_masking_curves


def main():
    parser = argparse.ArgumentParser(
        description='Plot MoRF/LeRF masking curves from faithfulness results.')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--exp_key', type=str, default=None, choices=EXP_KEYS,
                        help='Experiment type to plot (default: all 4)')
    parser.add_argument('--method', type=str, default='complete', choices=LINKAGE_METHODS,
                        help='Hierarchical clustering linkage method')
    parser.add_argument('--model', type=str, default='SCCNet', choices=MODEL_NAMES,
                        help='Classifier model (must already have trained weights saved)')
    args = parser.parse_args()

    dataset_name   = args.dataset
    exp_keys       = [args.exp_key] if args.exp_key else EXP_KEYS
    linkage_method = args.method
    model_name     = args.model

    _, preprocessed_dir, ch_num, _, _, tasks, _, subjects_list, labels_list = \
        build(dataset_name)

    for task in tasks:
        labels_map = labels_mapping(dataset_name, task)
        filtered_subjects, filtered_labels = filter_data(subjects_list, labels_list, labels_map)
        label_dict = dict(zip(filtered_subjects, filtered_labels))
        samples, _, _ = load_preprocessed(preprocessed_dir, labels_map, label_dict)

        model          = single_model(num_classes=2, num_channels=ch_num,
                                      time_points=samples.shape[2], model_name=model_name)
        best_iteration = find_best_iteration(dataset_name, task, model['name'])

        for exp_key in exp_keys:
            print(f"\n--- {dataset_name} | {task} | {exp_key} ---")
            plot_masking_curves(
                dataset_name, task, model['name'], best_iteration,
                GRADIENT_METHODS, exp_key,
                linkage_method=linkage_method,
            )


if __name__ == '__main__':
    main()
