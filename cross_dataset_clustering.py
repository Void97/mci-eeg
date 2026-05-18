import os
import json
import types
import numpy as np
import argparse
from collections import defaultdict, Counter
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt

from clustering_utils.inference_saliency import GRADIENT_METHODS
from clustering_utils.clustering_20260424 import Pipeline, PSDConverter

# Intersection of all dataset channel sets (GENEEG is the smallest at 17 ch)
COMMON_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8',
    'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'T3', 'T4', 'Fz', 'Cz', 'Pz'
]

# 'diagnosis' = the positive class label for each source (what the saliency maps represent)
SOURCES = [
    {
        'dataset': 'GENEEG', 'task': 'MCI vs HC', 'diagnosis': 'MCI',
        'model': 'SCCNet', 'iteration': 2,
        'ch_names': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'C3', 'C4', 'P3', 'P4',
                     'O1', 'O2', 'T3', 'T4', 'Fz', 'Cz', 'Pz'],
    },
    {
        'dataset': 'MCIvsHC', 'task': 'MCI vs HC', 'diagnosis': 'MCI',
        'model': 'SCCNet', 'iteration': 1,
        'ch_names': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
                     'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'],
    },
    {
        'dataset': 'CAUEEG', 'task': 'Dementia vs Normal', 'diagnosis': 'Dementia',
        'model': 'SCCNet', 'iteration': 5,
        'ch_names': ['Fp1', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F4', 'C4', 'P4', 'O2',
                     'F7', 'T3', 'T5', 'F8', 'T4', 'T6', 'Fz', 'Cz', 'Pz'],
    },
    {
        'dataset': 'CAUEEG', 'task': 'MCI vs Normal', 'diagnosis': 'MCI',
        'model': 'SCCNet', 'iteration': 5,
        'ch_names': ['Fp1', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F4', 'C4', 'P4', 'O2',
                     'F7', 'T3', 'T5', 'F8', 'T4', 'T6', 'Fz', 'Cz', 'Pz'],
    },
    {
        'dataset': 'ADvsFTDvsHC', 'task': 'AD vs HC', 'diagnosis': 'AD',
        'model': 'SCCNet', 'iteration': 7,
        'ch_names': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
                     'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'],
    },
    {
        'dataset': 'ADvsFTDvsHC', 'task': 'FTD vs HC', 'diagnosis': 'FTD',
        'model': 'SCCNet', 'iteration': 13,
        'ch_names': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
                     'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'],
    },
]


def _make_visualizer(all_diagnoses):
    def _visualize_named(self, features, cluster_labels, subject_ids_list):
        cluster_names = _name_clusters(cluster_labels, all_diagnoses)

        perplexity = min(5, len(features) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        features_2d = tsne.fit_transform(features)

        unique_names = sorted(set(cluster_names.values()))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_names)))
        name_to_color = dict(zip(unique_names, colors))

        plt.figure(figsize=(10, 7))
        for name in unique_names:
            mask = np.array([cluster_names[c] == name for c in cluster_labels])
            plt.scatter(
                features_2d[mask, 0], features_2d[mask, 1],
                color=name_to_color[name], label=name, alpha=0.7, s=30
            )
        plt.legend(title='Clusters', loc='upper right',
                   bbox_to_anchor=(1.18, 1), borderaxespad=0)

        filepath = os.path.join(
            self._get_base_path(),
            f'{self.dataset_name}_{self.task}_{self.model_name}'
            f'_iteration_{self.best_iteration}_hierarchical_{self.gradient_method}.png'
        )
        plt.title(
            f'Hierarchical Clustering: {self.dataset_name}, {self.task}, '
            f'{self.model_name} (Iteration: {self.best_iteration})'
        )
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True)
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Saved clustering visualization to {filepath}")
        plt.close()

    return _visualize_named


def _name_clusters(cluster_labels, all_diagnoses, threshold=0.5):
    """Assign a human-readable name to each cluster based on its dominant diagnosis."""
    raw_names = {}
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        counts = Counter(all_diagnoses[mask])
        dominant, dominant_count = counts.most_common(1)[0]
        total = mask.sum()
        raw_names[cluster_id] = dominant if dominant_count / total >= threshold else 'Mixed'

    # Add numeric suffix when multiple clusters share the same dominant label
    name_freq = Counter(raw_names.values())
    suffix_idx = defaultdict(int)
    named = {}
    for cluster_id in sorted(raw_names):
        base = raw_names[cluster_id]
        if name_freq[base] > 1:
            named[cluster_id] = f'{base}-{suffix_idx[base]}'
            suffix_idx[base] += 1
        else:
            named[cluster_id] = base
    return named


def _save_cluster_composition(all_labels, all_diagnoses, cluster_labels, gradient_method):
    cluster_names = _name_clusters(cluster_labels, all_diagnoses)

    by_source = defaultdict(lambda: defaultdict(int))
    by_diagnosis = defaultdict(lambda: defaultdict(int))
    label_map = defaultdict(list)

    for label, diagnosis, cluster in zip(all_labels, all_diagnoses, cluster_labels):
        source_key = '_'.join(label.split('_')[:-1])
        by_source[int(cluster)][source_key] += 1
        by_diagnosis[int(cluster)][diagnosis] += 1
        label_map[int(cluster)].append(label)

    log = {}
    for cluster_id in sorted(by_source):
        name = cluster_names[cluster_id]
        log[name] = {
            'cluster_id': cluster_id,
            'total': sum(by_source[cluster_id].values()),
            'by_diagnosis': dict(by_diagnosis[cluster_id]),
            'by_source': dict(by_source[cluster_id]),
            'subject_labels': label_map[cluster_id],
        }

    log_dir = 'results/clustering/aggregated/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f'aggregated_MCI-related_SCCNet_iteration_0_hierarchical_{gradient_method}_composition.json'
    )
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=4)
    print(f"Saved cluster composition to {log_path}")

    for cluster_id in sorted(by_source):
        name = cluster_names[cluster_id]
        total = sum(by_source[cluster_id].values())
        diag_str = ', '.join(f'{k}: {v}' for k, v in sorted(by_diagnosis[cluster_id].items()))
        src_str  = ', '.join(f'{k}: {v}' for k, v in sorted(by_source[cluster_id].items()))
        print(f"  [{name}] ({total})  diagnosis=[{diag_str}]  source=[{src_str}]")


def _channel_indices(source_ch_names, target_ch_names):
    return [source_ch_names.index(ch) for ch in target_ch_names]


def load_saliency_maps(source, gradient_method, saliency_dir='results/saliency_maps'):
    dataset = source['dataset']
    task = source['task']
    model = source['model']
    iteration = source['iteration']
    base = os.path.join(saliency_dir, dataset)

    path = os.path.join(
        base,
        f'{dataset}_{task}_{model}_iteration_{iteration}_{gradient_method}_saliency_maps.npy'
    )
    if not os.path.exists(path):
        print(f"[skip] {dataset} / {task}: file not found ({os.path.basename(path)})")
        return None
    return np.load(path)


def main():
    parser = argparse.ArgumentParser(description='Cross-dataset saliency clustering')
    parser.add_argument(
        '--gradient', type=str, default='vanilla', choices=GRADIENT_METHODS,
        help=f'Gradient attribution method. Available: {", ".join(GRADIENT_METHODS)}'
    )
    args = parser.parse_args()
    gradient_method = args.gradient

    bands = {
        'delta': [0, 4],
        'theta': [4, 8],
        'alpha': [8, 12],
        'beta': [12, 30],
        'gamma': [30, 45]
    }

    psd_converter = PSDConverter()
    all_psd = []
    all_labels = []
    all_diagnoses = []

    for source in SOURCES:
        maps = load_saliency_maps(source, gradient_method)
        if maps is None:
            continue

        ch_idx = _channel_indices(source['ch_names'], COMMON_CHANNELS)
        maps = maps[:, ch_idx, :]

        psd = psd_converter.convert(maps, bands)   # (n, 17, 5)
        flat = psd.reshape(len(psd), -1)           # (n, 85)
        flat = flat - flat.mean(axis=0)            # center per source to remove baseline offset

        prefix = f"{source['dataset']}_{source['task'].replace(' ', '')}"
        labels = np.array([f"{prefix}_{i}" for i in range(len(maps))])

        all_psd.append(flat)
        all_labels.append(labels)
        all_diagnoses.extend([source['diagnosis']] * len(maps))
        print(f"Loaded {len(maps)} subjects from {source['dataset']} / {source['task']} [{source['diagnosis']}]")

    if not all_psd:
        print("No saliency maps found. Exiting.")
        return

    all_psd = np.concatenate(all_psd, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_diagnoses = np.array(all_diagnoses)
    all_features = normalize(all_psd, norm='l2')
    print(f"\nTotal: {len(all_features)} subjects | features shape: {all_features.shape}")

    pipeline = Pipeline(
        dataset_name='aggregated',
        task='MCI-related',
        model_name='SCCNet',
        best_iteration=0,
        bands=bands,
        gradient_method=gradient_method
    )
    pipeline.visualizer.visualize_clusters = types.MethodType(
        _make_visualizer(all_diagnoses), pipeline.visualizer
    )

    cluster_labels = pipeline.run(None, all_labels, precomputed_features=all_features)
    _save_cluster_composition(all_labels, all_diagnoses, cluster_labels, gradient_method)


if __name__ == '__main__':
    main()
