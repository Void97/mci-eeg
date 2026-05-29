"""
Shared constants and helpers used across the clustering pipeline.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder


BANDS = {
    'delta': [0,  4],
    'theta': [4,  8],
    'alpha': [8,  12],
    'beta':  [12, 30],
    'gamma': [30, 45],
}

N_FOLDS = 5
SFREQ   = 200


def encode_groups(dataset_name, groups):
    """
    Encode subject-ID groups to integers, matching the convention used
    throughout the pipeline.

    ADvsFTDvsHC has non-numeric participant IDs (e.g. 'sub-001') →
    LabelEncoder → 1-indexed integers.
    All other datasets have numeric string IDs → direct int cast.
    """
    if dataset_name == 'ADvsFTDvsHC':
        enc = LabelEncoder()
        return enc.fit_transform(groups).astype(np.int64) + 1
    return groups.astype(np.int64)


def get_fold_splits(samples, targets, groups):
    """
    Return a dict {fold_index: test_indices} for a 5-fold
    StratifiedGroupKFold split — the single authoritative split used by
    both inference_saliency and faithfulness evaluation.
    """
    kf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=False)
    return {
        fold: test_idx
        for fold, (_, test_idx) in enumerate(
            kf.split(samples, targets, groups=groups)
        )
    }
