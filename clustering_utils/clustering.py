import os
import numpy as np
import hdbscan
import json
import umap

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from scipy.signal import welch

def time_to_psd(saliency_maps_list, bands, sfreq=200):

    welch_saliency_maps_list = []
    for saliency_map in saliency_maps_list:
        freqs, psd = welch(
            saliency_map,
            fs=sfreq,
            axis=1,
            nperseg=sfreq,
            noverlap=sfreq // 2
        )
        
        psd = psd / (psd.sum() + 1e-8)

        band_psd_sub = []
        for band in bands:
            fmin, fmax = bands[band]
            if band == 'all':
                band_mask = (freqs >= fmin) & (freqs <= fmax)
                #continue
            else:
                band_mask = (freqs >= fmin) & (freqs < fmax)
            band_psd = psd[:, band_mask].mean(axis=1)
            band_psd_sub.append(band_psd)
        band_psd_sub = np.array(band_psd_sub).T
        welch_saliency_maps_list.append(band_psd_sub)
    return np.array(welch_saliency_maps_list)

def file_path(dataset_name):
    save_path = f'results/clustering/{dataset_name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path

def calc_subjects(cluster_labels, clustering_logs):

    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    for label, count in enumerate(zip(unique_labels, counts)):
        if label == -1:
            print(f'Noise (Outliers): {count} subjects')
            clustering_logs['n_noise'] = int(count[1])
        else:
            print(f'Cluster {label}: {count} subjects')
            clustering_logs[f'cluster_{label}_size'] = int(count[1])


def visualize_tsne(dataset_name, task, model_name, best_iteration, fold, embeddings, cluster_labels, subject_ids_list, clustering_method):
    perplexity = min(5, len(embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
    # umap_reducer = umap.UMAP()
    # embeddings_2d = umap_reducer.fit_transform(embeddings)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=cluster_labels, 
        cmap='viridis', 
        alpha=0.7)
    
    for i, txt in enumerate(subject_ids_list):
        plt.annotate(
            text=str(txt.item() if hasattr(txt, "item") else txt),
            xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center',
            fontsize=8,
            alpha=0.8
        )

    legend = plt.legend(
        *scatter.legend_elements(),
        title='Clusters',
        loc='upper right',
        bbox_to_anchor=(1.12, 1),
        borderaxespad=0
    )
    plt.gca().add_artist(legend)
    
    filepath = file_path(dataset_name)
    plt_path = os.path.join(filepath,f'{dataset_name}_{task}_{model_name}_iteration_{best_iteration}_fold_{fold}_{clustering_method}.png')
    plt.title(f'Clustering {clustering_method}: {dataset_name}, {task}, {model_name} (Iteration: {best_iteration}), {fold}') 
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    #plt.colorbar(scatter, label='Cluster Label')
    plt.grid(True)
    #plt_path = f'results/clustering/{dataset_name}/{dataset_name}_{task}_{model_name}_iteration_{best_iteration}_fold_{fold}_{clustering_method}.png'
    plt.savefig(plt_path)
    print(f"Saved clustering visualization to {plt_path}")
    plt.close()


#merge embeddings and cluster labels from all folds
def cluster(dataset_name, task, model_name, best_iteration, saliency_maps_list, subject_ids_list, clustering_method, bands):
    
    clustering_logs = {}
    clustering_logs['dataset_name'] = dataset_name
    clustering_logs['task'] = task
    clustering_logs['model_name'] = model_name
    clustering_logs['best_iteration'] = best_iteration
    clustering_logs['clustering_method'] = clustering_method
    
    logs_savedir = f'results/clustering/{dataset_name}/logs'
    if not os.path.exists(logs_savedir):
        os.makedirs(logs_savedir)

    print("Clustering across all folds")
    print(f"Clustering for dataset: {dataset_name}, task: {task}, model: {model_name}, iteration: {best_iteration}, fold: all folds")

    saliency_maps_list = time_to_psd(saliency_maps_list, bands)
    print(f"Saliency maps converted to PSD with shape: {saliency_maps_list.shape}")
    print('---------------------------------------------')

    flat_saliency_maps = saliency_maps_list.reshape(saliency_maps_list.shape[0], -1)
    print(f'Flat saliency maps shape: {flat_saliency_maps.shape}')
    
    scaler = StandardScaler()
    flat_saliency_maps = scaler.fit_transform(flat_saliency_maps)
    pca = PCA(n_components=0.7, random_state=42)
    flat_saliency_maps_no_norm = pca.fit_transform(flat_saliency_maps)
    # umap_reducer = umap.UMAP(n_components=10)
    # flat_saliency_maps_no_norm = umap_reducer.fit_transform(flat_saliency_maps)
    flat_saliency_maps = normalize(flat_saliency_maps_no_norm, norm='l2')
    #print(f'Flat saliency maps after PCA shape: {flat_saliency_maps.shape}')
    print(f'Flat saliency maps after UMAP shape: {flat_saliency_maps.shape}')

    #exit()

    if clustering_method == 'kmeans':
        print('Using K-means clustering for all folds')
        max_k = min(10, len(flat_saliency_maps) - 1)
        best_score = -1
        best_k = None
        for k in range(2, max_k + 1):
            clusters = KMeans(n_clusters=k, n_init='auto', random_state=42)
            cluster_labels = clusters.fit_predict(flat_saliency_maps)
            score = silhouette_score(flat_saliency_maps, cluster_labels, metric='cosine')
            print(f"Number of clusters: {k}, Silhouette Score: {score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k
                print(f"Best number of clusters updated to: {best_k} with Silhouette Score: {best_score:.4f}")

        clusters = KMeans(n_clusters=best_k, n_init='auto', random_state=42)
        cluster_labels = clusters.fit_predict(flat_saliency_maps)
        if best_k >= 2:
            score = silhouette_score(flat_saliency_maps, cluster_labels, metric='cosine')
            print(f"K-means Silhouette Score: {score:.4f}")
            clustering_logs['n_clusters'] = int(best_k)
            clustering_logs['silhouette_score'] = float(f"{score:.4f}")
            calc_subjects(cluster_labels, clustering_logs)
            print(f"Final clustering results for all folds - Number of clusters: {best_k}, Silhouette Score: {best_score:.4f}")
            #calc_subjects(cluster_labels)


    elif clustering_method == 'agglomerative':
        print("Using Agglomerative Clustering for all folds")
        # Determine the optimal number of clusters using the silhouette score method
        max_k = min(10, len(flat_saliency_maps) - 1)
        best_score = -1
        best_k = None
        for k in range(2, max_k + 1):
            clusters = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
            cluster_labels = clusters.fit_predict(flat_saliency_maps)
            score = silhouette_score(flat_saliency_maps, cluster_labels, metric='cosine')
            print(f"Number of clusters: {k}, Silhouette Score: {score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k
                print(f"Best number of clusters updated to: {best_k} with Silhouette Score: {best_score:.4f}")
        # Perform KMeans clustering with the optimal number of clusters
        clusters = AgglomerativeClustering(n_clusters=best_k, metric='cosine', linkage='average')
        cluster_labels = clusters.fit_predict(flat_saliency_maps)
        if best_k >= 2:
            score = silhouette_score(flat_saliency_maps, cluster_labels, metric='cosine')
            score_db = davies_bouldin_score(flat_saliency_maps, cluster_labels)
            score_ch = calinski_harabasz_score(flat_saliency_maps, cluster_labels)
            print(f"Agglomerative Clustering Silhouette Score: {score:.4f}")
            clustering_logs['n_clusters'] = int(best_k)
            clustering_logs['silhouette_score'] = float(f"{score:.4f}")
            clustering_logs['davies_bouldin_score'] = float(f"{score_db:.4f}")
            clustering_logs['calinski_harabasz_score'] = float(f"{score_ch:.4f}")
            calc_subjects(cluster_labels, clustering_logs)
        print(f"Final clustering results for all folds - Number of clusters: {best_k}, Silhouette Score: {best_score:.4f}")
        

    elif clustering_method == 'hdbscan':
        print("Using HDBSCAN for all folds")
        if dataset_name == 'CAUEEG':
            clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=15)
        else:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=3)
        cluster_labels = clusterer.fit_predict(flat_saliency_maps)
        # clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=3)
        # cluster_labels = clusterer.fit_predict(flat_saliency_maps)

        valid_mask = cluster_labels != -1
        valid_labels = cluster_labels[valid_mask]
        n_clusters = len(np.unique(valid_labels))
        n_noise = np.sum(cluster_labels == -1)
        print(f"HDBSCAN results for all folds - Number of clusters: {n_clusters}, Number of noise points: {n_noise}")

        if n_clusters >= 2 and valid_mask.sum() > 1:
            score = silhouette_score(flat_saliency_maps[valid_mask], valid_labels, metric='cosine')
            score_db = davies_bouldin_score(flat_saliency_maps[valid_mask], valid_labels)
            score_ch = calinski_harabasz_score(flat_saliency_maps[valid_mask], valid_labels)
            print(f"HDBSCAN Silhouette Score (excluding noise): {score:.4f}")
            
            clustering_logs['n_clusters'] = int(n_clusters)
            clustering_logs['n_noise'] = int(n_noise)
            clustering_logs['silhouette_score'] = float(f"{score:.4f}")
            clustering_logs['davies_bouldin_score'] = float(f"{score_db:.4f}")
            clustering_logs['calinski_harabasz_score'] = float(f"{score_ch:.4f}")
            calc_subjects(cluster_labels, clustering_logs)
    
    elif clustering_method == 'gaussian':
        print("Using Gaussian Mixture Model for all folds")
        # Determine the optimal number of clusters using BIC
        lowest_bic = np.inf
        best_gmm = None
        n_components_range = range(1, 10)
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, covariance_type='spherical', random_state=42)
            gmm.fit(flat_saliency_maps)
            bic = gmm.bic(flat_saliency_maps)
            print(f'Number of components: {n_components}, BIC: {bic:.4f}')
            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = gmm
                print(f'Best number of components updated to: {n_components} with BIC: {lowest_bic:.4f}')
        cluster_labels = best_gmm.predict(flat_saliency_maps)
        if n_components >= 2:
            score = silhouette_score(flat_saliency_maps, cluster_labels, metric='cosine')
            score_db = davies_bouldin_score(flat_saliency_maps, cluster_labels)
            score_ch = calinski_harabasz_score(flat_saliency_maps, cluster_labels)
            print(f"Gaussian Mixture Model Silhouette Score: {score:.4f}")
            clustering_logs['n_clusters'] = int(best_gmm.n_components)
            clustering_logs['bic'] = float(f"{lowest_bic:.4f}")
            clustering_logs['silhouette_score'] = float(f"{score:.4f}")
            clustering_logs['davies_bouldin_score'] = float(f"{score_db:.4f}")
            clustering_logs['calinski_harabasz_score'] = float(f"{score_ch:.4f}")
            calc_subjects(cluster_labels, clustering_logs)
        print(f"Final clustering results for all folds - Best number of components: {best_gmm.n_components}, BIC: {lowest_bic:.4f}, Silhouette Score: {score:.4f}")

    
    elif clustering_method == 'spectral':
        print("Using Spectral Clustering for all folds")
        # Determine the optimal number of clusters using the eigengap heuristic
        max_k = min(10, len(flat_saliency_maps) - 1)
        best_score = -1
        best_k = None
        for k in range(2, max_k + 1):
            clusters = SpectralClustering(n_clusters=k)
            cluster_labels = clusters.fit_predict(flat_saliency_maps)
            score = silhouette_score(flat_saliency_maps, cluster_labels, metric='cosine')
            print(f"Number of clusters: {k}, Silhouette Score: {score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k
                print(f"Best number of clusters updated to: {best_k} with Silhouette Score: {best_score:.4f}")
        clusters = SpectralClustering(n_clusters=best_k, affinity='nearest_neighbors', random_state=42)
        cluster_labels = clusters.fit_predict(flat_saliency_maps)
        if best_k >= 2:
            score = silhouette_score(flat_saliency_maps, cluster_labels, metric='cosine')
            print(f"Spectral Clustering Silhouette Score: {score:.4f}")
            clustering_logs['n_clusters'] = int(best_k)
            clustering_logs['silhouette_score'] = float(f"{score:.4f}")
            calc_subjects(cluster_labels, clustering_logs)
        print(f"Final clustering results for all folds - Number of clusters: {best_k}, Silhouette Score: {best_score:.4f}")
        print("Cluster sizes:", np.bincount(cluster_labels))      
    else:
        raise ValueError(f"Unsupported clustering method: {clustering_method}")
    
    #save clustering logs
    logs_path = os.path.join(logs_savedir, f'{dataset_name}_{task}_{model_name}_iteration_{best_iteration}_{clustering_method}_clustering_logs.json')
    with open(logs_path, 'w') as f:
        json.dump(clustering_logs, f, indent=4)
    print(f"Saved clustering logs to {logs_path}")

    visualize_tsne(dataset_name, task, model_name, best_iteration, 'all_folds', flat_saliency_maps_no_norm, cluster_labels, subject_ids_list, clustering_method)

    # return cluster_mean_saliency_maps(saliency_maps_list, cluster_labels)
    return cluster_labels


