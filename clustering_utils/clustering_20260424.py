import os
import numpy as np
import json

from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from matplotlib import legend, pyplot as plt
from scipy.signal import welch
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram as scipy_dendrogram
from abc import ABC, abstractmethod


class PSDConverter:
    def __init__(self, sfreq=200):
        self.sfreq = sfreq

    def extract_band_psd(self, freqs, psd, bands):
        band_psd_sub = []
        for band in bands:
            fmin, fmax = bands[band]
            if band == 'all':
                band_mask = (freqs >= fmin) & (freqs <= fmax)
            else:
                band_mask = (freqs >= fmin) & (freqs < fmax)
            band_psd_sub.append(psd[:, band_mask].sum(axis=1))
        return np.array(band_psd_sub).T

    def convert(self, saliency_maps, bands):
        welch_saliency_maps = []
        for saliency_map in saliency_maps:
            freqs, psd = welch(
                saliency_map,
                fs=self.sfreq,
                axis=1,
                nperseg=self.sfreq,
                noverlap=self.sfreq // 2
            )
            psd = psd / (psd.sum() + 1e-8)
            band_psd_sub = self.extract_band_psd(freqs, psd, bands)
            welch_saliency_maps.append(band_psd_sub)
        return np.array(welch_saliency_maps)


class FeaturePreprocessor:
    def preprocess(self, saliency_maps):
        flat = saliency_maps.reshape(saliency_maps.shape[0], -1)
        print(f"Flat saliency maps shape: {flat.shape}")
        return normalize(flat, norm='l2')


class Visualizer:
    def __init__(self, dataset_name, task, model_name,
                 best_iteration, gradient_method='vanilla', output_dir='results/clustering'):
        self.dataset_name = dataset_name
        self.task = task
        self.model_name = model_name
        self.best_iteration = best_iteration
        self.gradient_method = gradient_method
        self.output_dir = output_dir

    def _get_base_path(self):
        path = f'{self.output_dir}/{self.dataset_name}'
        os.makedirs(path, exist_ok=True)
        return path

    def save_dendrogram(self, linkage_matrix, subject_ids, n_clusters, cut_threshold, cluster_labels):
        path = os.path.join(self._get_base_path(), 'dendrograms')
        os.makedirs(path, exist_ok=True)

        n = len(subject_ids)
        fig_width = max(12, n * 0.6)
        fig, ax = plt.subplots(figsize=(fig_width, 8))

        labels = [str(s.item() if hasattr(s, 'item') else s) for s in subject_ids]
        dend = scipy_dendrogram(
            linkage_matrix,
            labels=labels,
            ax=ax,
            leaf_rotation=90,
            color_threshold=cut_threshold,
            above_threshold_color='gray'
        )

        # Color each leaf label by its flat cluster assignment
        cmap = plt.cm.tab10
        leaf_order = dend['leaves']  # original indices left→right in the dendrogram
        for tick, leaf_idx in zip(ax.get_xticklabels(), leaf_order):
            tick.set_color(cmap(cluster_labels[leaf_idx] / max(n_clusters - 1, 1)))

        ax.axhline(
            y=cut_threshold, color='red', linestyle='--', linewidth=1.5,
            label='Cut threshold'
        )

        cluster_handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=cmap(c / max(n_clusters - 1, 1)),
                       markersize=10, label=f'Cluster {c}')
            for c in range(n_clusters)
        ]
        cluster_handles.append(
            plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
                       label=f'Cut threshold ({n_clusters} clusters)')
        )
        ax.legend(handles=cluster_handles, fontsize=10)

        ax.set_title(
            f'Hierarchical Clustering Dendrogram\n'
            f'{self.dataset_name} — {self.task} ({n_clusters} clusters)',
            fontsize=14
        )
        ax.set_xlabel('Subject ID')
        ax.set_ylabel('Distance')

        savepath = os.path.join(
            path,
            f'{self.dataset_name}_{self.task}_{self.model_name}'
            f'_iteration_{self.best_iteration}_{self.gradient_method}_dendrogram.png'
        )
        plt.tight_layout()
        plt.savefig(savepath, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved dendrogram to {savepath}")

    def visualize_clusters(self, features, cluster_labels, subject_ids_list):
        perplexity = min(5, len(features) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        features_2d = tsne.fit_transform(features)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            features_2d[:, 0], features_2d[:, 1],
            c=cluster_labels, cmap='viridis', alpha=0.7
        )

        for i, txt in enumerate(subject_ids_list):
            plt.annotate(
                text=str(txt.item() if hasattr(txt, 'item') else txt),
                xy=(features_2d[i, 0], features_2d[i, 1]),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                fontsize=8,
                alpha=0.8
            )

        leg = plt.legend(
            *scatter.legend_elements(),
            title='Clusters',
            loc='upper right',
            bbox_to_anchor=(1.12, 1),
            borderaxespad=0
        )
        plt.gca().add_artist(leg)

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
        plt.savefig(filepath)
        print(f"Saved clustering visualization to {filepath}")
        plt.close()


class ClusteringStrategy(ABC):
    @abstractmethod
    def cluster(self, features, dataset_name=None):
        pass

    @abstractmethod
    def get_metrics(self, features, labels):
        pass


class HierarchicalClustering(ClusteringStrategy):
    """
    Ward hierarchical clustering via scipy.cluster.hierarchy.
    Cluster count is determined automatically from the largest gap
    between consecutive merge distances in the dendrogram.
    """

    def __init__(self, method='complete', metric='cosine'):
        self.method = method
        self.metric = metric
        self.linkage_matrix = None
        self.n_clusters = None
        self.cut_threshold = None

    def _find_n_clusters(self, linkage_matrix):
        n = len(linkage_matrix) + 1
        if n <= 3:
            return 2

        distances = linkage_matrix[:, 2]
        # Second derivative of merge distances: finds where growth accelerates most,
        # rather than just the single largest gap (which almost always gives 2 clusters)
        acceleration = np.diff(distances, 2)
        # Argmax from the right: large distances = few clusters
        rev_idx = int(np.argmax(acceleration[::-1]))
        k = rev_idx + 2
        return max(2, min(k, 10))

    def cluster(self, features, dataset_name=None):
        self.linkage_matrix = linkage(features, method=self.method, metric=self.metric)
        self.n_clusters = self._find_n_clusters(self.linkage_matrix)

        # Cut threshold: midpoint between the two merges that bound the chosen cut
        distances = self.linkage_matrix[:, 2]
        n = len(distances) + 1
        cut_lower = n - self.n_clusters - 1
        cut_upper = n - self.n_clusters
        if 0 <= cut_lower and cut_upper < len(distances):
            self.cut_threshold = (distances[cut_lower] + distances[cut_upper]) / 2
        else:
            self.cut_threshold = distances[-self.n_clusters + 1]

        cluster_labels = fcluster(
            self.linkage_matrix, t=self.cut_threshold, criterion='distance'
        ) - 1  # 0-indexed

        actual_k = len(np.unique(cluster_labels))
        if actual_k != self.n_clusters:
            print(f"Note: expected {self.n_clusters} clusters, got {actual_k}.")
            self.n_clusters = actual_k

        print(f"Hierarchical clustering: {self.n_clusters} clusters "
              f"(cut threshold: {self.cut_threshold:.4f})")
        return cluster_labels

    def get_metrics(self, features, labels):
        if len(set(labels)) >= 2:
            return {
                'n_clusters': int(len(set(labels))),
                'silhouette_score': float(silhouette_score(features, labels, metric='cosine')),
                'davies_bouldin_score': float(davies_bouldin_score(features, labels)),
                'calinski_harabasz_score': float(calinski_harabasz_score(features, labels))
            }
        return {}


class Logs:
    def __init__(self):
        self.logs = {}

    def analyze_clusters(self, cluster_labels):
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        results = {}
        for label, count in zip(unique_labels, counts):
            print(f'Cluster {int(label)}: {int(count)} subjects')
            results[f'cluster_{int(label)}_size'] = int(count)
        return results

    def save_logs(self, dataset_name, task, model_name, best_iteration,
                  clustering_method, metrics):
        logs_savedir = f'results/clustering/{dataset_name}/logs'
        os.makedirs(logs_savedir, exist_ok=True)

        log_entry = {
            'dataset_name': dataset_name,
            'task': task,
            'model_name': model_name,
            'best_iteration': int(best_iteration),
            'clustering_method': clustering_method,
            **metrics
        }
        log_path = os.path.join(
            logs_savedir,
            f'{dataset_name}_{task}_{model_name}_iteration_{best_iteration}'
            f'_{clustering_method}_clustering_logs.json'
        )
        with open(log_path, 'w') as f:
            json.dump(log_entry, f, indent=4)
        print(f"Saved clustering logs to {log_path}")


class Pipeline:
    def __init__(self, dataset_name, task, model_name, best_iteration, bands, gradient_method='vanilla'):
        self.dataset_name = dataset_name
        self.task = task
        self.model_name = model_name
        self.best_iteration = best_iteration
        self.bands = bands
        self.gradient_method = gradient_method

        self.psd_converter = PSDConverter()
        self.feature_preprocessor = FeaturePreprocessor()
        self.visualizer = Visualizer(dataset_name, task, model_name, best_iteration, gradient_method)
        self.logs = Logs()
        self.clustering_strategy = HierarchicalClustering()

    def run(self, saliency_maps_list, subject_ids_list):
        print(f"Hierarchical clustering for: {self.dataset_name}, {self.task}, "
              f"model: {self.model_name}, iteration: {self.best_iteration}")

        psd_features = self.psd_converter.convert(saliency_maps_list, self.bands)
        print(f"PSD features shape: {psd_features.shape}")

        processed_features = self.feature_preprocessor.preprocess(psd_features)
        print(f"Processed features shape: {processed_features.shape}")

        cluster_labels = self.clustering_strategy.cluster(processed_features)

        self.visualizer.save_dendrogram(
            self.clustering_strategy.linkage_matrix,
            subject_ids_list,
            self.clustering_strategy.n_clusters,
            self.clustering_strategy.cut_threshold,
            cluster_labels
        )

        metrics = self.clustering_strategy.get_metrics(processed_features, cluster_labels)
        cluster_counts = self.logs.analyze_clusters(cluster_labels)
        metrics.update(cluster_counts)

        self.logs.save_logs(self.dataset_name, self.task, self.model_name,
                            self.best_iteration, f'hierarchical_{self.gradient_method}', metrics)

        self.visualizer.visualize_clusters(processed_features, cluster_labels, subject_ids_list)

        return cluster_labels
