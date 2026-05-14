import os
import numpy as np
import hdbscan
import json


from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from matplotlib import legend, pyplot as plt
from scipy.signal import welch
from abc import ABC, abstractmethod

from clustering_utils import best_iteration

class PSDConverter:
    #Convert saliency maps to PSD features
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
            #minmax
            #psd = (psd - psd.min()) / (psd.max() - psd.min() + 1e-8)
            band_psd_sub = self.extract_band_psd(freqs, psd, bands)
            welch_saliency_maps.append(band_psd_sub)
        return np.array(welch_saliency_maps)

class FeaturePreprocessor:
    #Preprocess PSD features: flattening, scaling, PCA, L2 normalization
    def __init__(self, pca_variance=0.9):
        self.pca_variance = pca_variance
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.pca_variance, random_state=42)
    
    def preprocess(self, saliency_maps):
        
        flat_saliency_maps = saliency_maps.reshape(saliency_maps.shape[0], -1)
        print(f"Flat saliency maps shape: {flat_saliency_maps.shape}")
        saliency_maps_scaled = self.scaler.fit_transform(flat_saliency_maps)
        saliency_maps_reduced = self.pca.fit_transform(saliency_maps_scaled)
        #saliency_maps_normalized = normalize(saliency_maps_reduced, norm='l2')
        return saliency_maps_reduced, saliency_maps_reduced

class Visualizer:

    def __init__(self, dataset_name, task, model_name,
                 best_iteration, clustering_method, output_dir='results/clustering'):
        
        self.dataset_name = dataset_name
        self.task = task
        self.model_name = model_name
        self.best_iteration = best_iteration
        self.clustering_method = clustering_method
        self.output_dir = output_dir

    def get_save_path(self):
        path = f'{self.output_dir}/{self.dataset_name}'
        if not os.path.exists(path):
            os.makedirs(path)
        filepath = os.path.join(path, f'{self.dataset_name}_{self.task}_{self.model_name}_iteration_{self.best_iteration}_{self.clustering_method}.png')
        return filepath
    
    def visualize_clusters(self, features, cluster_labels, 
                           subjects_ids_list):
        
        perplexity = min(5, len(features) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        features_2d = tsne.fit_transform(features)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                              c=cluster_labels, cmap='viridis', alpha=0.7)

        for i, txt in enumerate(subjects_ids_list):
            plt.annotate(
                text=str(txt.item() if hasattr(txt, 'item') else txt),
                xy=(features_2d[i, 0], features_2d[i, 1]),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                fontsize=8,
                alpha=0.8)
                
        
        legend = plt.legend(
            *scatter.legend_elements(),
            title='Clusters',
            loc='upper right',
            bbox_to_anchor=(1.12, 1),
            borderaxespad=0
        )
        plt.gca().add_artist(legend)

        savepath = self.get_save_path()
        plt.title(f'Clustering {self.clustering_method}: {self.dataset_name}, {self.task}, {self.model_name} (Iteration: {self.best_iteration})') 
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        #plt.colorbar(scatter, label='Cluster Label')
        plt.grid(True)
        #plt_path = f'results/clustering/{self.dataset_name}/{self.dataset_name}_{self.task}_{self.model_name}_iteration_{self.best_iteration}_fold_{self.fold}_{self.clustering_method}.png'
        plt.savefig(savepath)
        print(f"Saved clustering visualization to {savepath}")
        plt.close()

class ClusteringStrategy(ABC):

    @abstractmethod
    def cluster(self, features, dataset_name=None):
        pass

    @abstractmethod
    def get_metrics(self, features, labels):
        pass

class HDBSCAN(ClusteringStrategy):
    def __init__(self, default_mincluster_size = 3, default_min_samples = 3):
        self.default_mincluster_size = default_mincluster_size
        self.default_min_samples = default_min_samples 
        self.noise_count = 0
    
    def cluster(self, features, dataset_name=None):

        if dataset_name == 'CAUEEG':
            min_cluster_size = 9
            min_samples = 5
        else:
            min_cluster_size = self.default_mincluster_size
            min_samples = self.default_min_samples

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        cluster_labels = clusterer.fit_predict(features)
        self.noise_count = np.sum(cluster_labels == -1)
        return cluster_labels

    def get_metrics(self, features, cluster_labels):

        valid_mask = cluster_labels != -1
        valid_labels = cluster_labels[valid_mask]
        n_clusters = len(np.unique(valid_labels))
        n_noise = np.sum(cluster_labels == -1)
        print(f"HDBSCAN results for all folds - Number of clusters: {n_clusters}, Number of noise points: {n_noise}")
        
        if n_clusters >= 2 and valid_mask.sum() > 1:
            return {
                'n_clusters': int(n_clusters),
                'n_noise': int(n_noise),
                'silhouette_score': float(silhouette_score(features[valid_mask], valid_labels, metric='euclidean')),
                'davies_bouldin_score': float(davies_bouldin_score(features[valid_mask], valid_labels)),
                'calinski_harabasz_score': float(calinski_harabasz_score(features[valid_mask], valid_labels))
            }
        else:
            return {}

class Agglomerative(ClusteringStrategy):
    def __init__(self, max_cluster=10, linkage='average'):
        self.max_cluster = max_cluster
        self.linkage = linkage
        self.best_k = None
        self.best_score = -1
    
    def cluster(self, features, dataset_name=None):
        
        max_k = min(10, len(features) - 1)
        for k in range(2, max_k + 1):
            clusters = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage=self.linkage)
            cluster_labels = clusters.fit_predict(features)
            score = silhouette_score(features, cluster_labels, metric='euclidean')
            print(f"Number of clusters: {k}, Silhouette Score: {score:.4f}")
            if score > self.best_score:
                self.best_score = score
                self.best_k = k
            print(f"Best number of clusters updated to: {self.best_k} with Silhouette Score: {self.best_score:.4f}")
        final_clusters = AgglomerativeClustering(n_clusters=self.best_k, metric='euclidean', linkage=self.linkage)
        return final_clusters.fit_predict(features)
    
    def get_metrics(self, features, labels):
        if len(set(labels)) >= 2:
            return {
                'n_clusters': int(len(set(labels))),
                'silhouette_score': float(silhouette_score(features, labels, metric='euclidean')),
                'davies_bouldin_score': float(davies_bouldin_score(features, labels)),
                'calinski_harabasz_score': float(calinski_harabasz_score(features, labels))
            }
        return {}

class GMM(ClusteringStrategy):
    def __init__(self, max_components=10, covariance_type = 'spherical'):
        self.max_components = max_components
        self.covariance_type = str(covariance_type)
        self.best_gmm = None
        self.lowest_bic = np.inf

    def cluster(self, features, dataset_name=None):
        
        for n_components in range(1, self.max_components):
            gmm = GaussianMixture(n_components=n_components, 
                                  covariance_type=self.covariance_type)
            gmm.fit(features)
            bic = gmm.bic(features)
            print(f'Number of components: {n_components}, BIC: {bic:.4f}')
            if bic < self.lowest_bic:
                self.lowest_bic = bic
                self.best_gmm = gmm
                print(f'Best number of components updated to: {n_components} with BIC: {self.lowest_bic:.4f}')
        return self.best_gmm.predict(features)

    def get_metrics(self, features, labels):
        
        if self.best_gmm.n_components >= 2:
            return {
                'n_clusters': int(self.best_gmm.n_components),
                'bic': float(self.lowest_bic),
                'silhouette_score': float(silhouette_score(features, labels, metric='euclidean')),
                'davies_bouldin_score': float(davies_bouldin_score(features, labels)),
                'calinski_harabasz_score': float(calinski_harabasz_score(features, labels))
            }
        return {}

class Logs:

    def __init__(self):
        self.logs = {}
    
    def analyze_clusters(self, cluster_labels):
        
        unique_labels, counts = np.unique(cluster_labels,return_counts=True)
        results = {}
        
        for label, count in enumerate(zip(unique_labels, counts)):
        
            if label == -1:
                print(f'Noise (Outliers): {count} subjects')
                results['n_noise'] = int(count[1])
                results[f'cluster_{int(label)}_size'] = int(count[1])
            else:
                print(f'Cluster {label}: {count} subjects')
                results[f'cluster_{int(label)}_size'] = int(count[1])
        return results

    
    def save_logs(self, dataset_name, task, model_name, best_iteration,
                  clustering_method, metrics):
        
        logs_savedir = f'results/clustering/{dataset_name}/logs'
        if not os.path.exists(logs_savedir):
            os.makedirs(logs_savedir)
        
        log_entry = {
            'dataset_name': dataset_name,
            'task': task,
            'model_name': model_name,
            'best_iteration': int(best_iteration),
            'clustering_method': clustering_method,
            **metrics
        }

        log_path = os.path.join(logs_savedir, 
                                f'{dataset_name}_{task}_{model_name}_iteration_{best_iteration}_{clustering_method}_clustering_logs.json')
        with open(log_path, 'w') as f:
            json.dump(log_entry, f, indent=4)
        print(f"Saved clustering logs to {log_path}")


class Pipeline:
    def __init__(self, dataset_name, task, model_name, best_iteration, 
                 clustering_method, bands):
        self.dataset_name = dataset_name
        self.task = task
        self.model_name = model_name
        self.best_iteration = best_iteration
        self.clustering_method = clustering_method
        self.bands = bands

        self.psd_converter = PSDConverter()
        self.feature_preprocessor = FeaturePreprocessor()
        self.visualizer = Visualizer(dataset_name, task, model_name, best_iteration, clustering_method)
        self.logs = Logs()
        self.clustering_strategy = self._get_clustering_strategy()

    def _get_clustering_strategy(self):
        strategies = {
            'hdbscan': HDBSCAN(),
            'agglomerative': Agglomerative(),
            'gaussian': GMM()
        }

        if self.clustering_method not in strategies:
            raise ValueError(f"Unsupported clustering method: {self.clustering_method}")
        
        return strategies[self.clustering_method]
    
    def run(self, saliency_maps_list, subject_ids_list):

        print(f"Clustering for dataset: {self.dataset_name}, task: {self.task}, "
              f"model: {self.model_name}, iteration: {self.best_iteration}")
        
        psd_features = self.psd_converter.convert(saliency_maps_list, self.bands)
        print(f"Saliency maps converted to PSD with shape: {psd_features.shape}")
        processed_features, reduced_features = self.feature_preprocessor.preprocess(psd_features)
        print(f"Processed features shape: {processed_features.shape}")

        cluster_labels = self.clustering_strategy.cluster(processed_features, self.dataset_name)
        metrics = self.clustering_strategy.get_metrics(processed_features, cluster_labels)
        cluster_counts = self.logs.analyze_clusters(cluster_labels)
        metrics.update(cluster_counts)

        self.logs.save_logs(self.dataset_name, self.task, self.model_name,
                            self.best_iteration, self.clustering_method, metrics)
        
        self.visualizer.visualize_clusters(reduced_features, cluster_labels, subject_ids_list)

        return cluster_labels
