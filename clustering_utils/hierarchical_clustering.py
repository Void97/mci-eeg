import os
import numpy as np
import hdbscan
import json
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

from helpers import PSDConverter, FeaturePreprocessor

class Clustering:
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

    def get_save_path(self, output_dir='results/clustering/'):
        path = f'{output_dir}/{self.dataset_name}'
        if not os.path.exists(path):
            os.makedirs(path)
        filepath = os.path.join(path, f'{self.dataset_name}_{self.task}_{self.model_name}_iteration_{self.best_iteration}_{self.clustering_method}.png')
        return filepath

    def visualize_tsne(self, features, cluster_labels, 
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
        

    
