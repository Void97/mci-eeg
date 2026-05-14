import numpy as np
from scipy.signal import welch, normalize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
        # saliency_maps_scaled = self.scaler.fit_transform(flat_saliency_maps)
        # saliency_maps_reduced = self.pca.fit_transform(saliency_maps_scaled)
        saliency_maps_normalized = normalize(flat_saliency_maps, norm='l2')
        return saliency_maps_normalized