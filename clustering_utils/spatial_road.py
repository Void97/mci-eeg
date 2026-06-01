"""
Exact port of XAI_tools_auto's noisy_spatial_imputer and dataset adjacency
structures for spatial ROAD masking.

Source: XAI_tools_auto/masking_saveData/spatial_road_saveData.py
"""
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

# Direct / indirect neighbour weights (same as XAI_tools_auto global)
_WEIGHTS = [1 / 6, 1 / 12]


# ─────────────────────────────────────────────────────────────
# Dataset adjacency structures (exact copies from XAI_tools_auto)
# ─────────────────────────────────────────────────────────────

class _Dataset:
    def __init__(self):
        self.dn_id  = None   # direct neighbour relative indices per channel
        self.idn_id = None   # indirect (diagonal) neighbour relative indices per channel


class GENEEG_HC(_Dataset):
    """17-channel GENEEG layout."""
    def __init__(self):
        self.dn_id = [
            (1, 2),           # 0: Fp1
            (-1, 2),          # 1: Fp2
            (-2, 2, 3, 5),    # 2: F3
            (-2, 1, 3, 6),    # 3: F4
            (-2, -1, 4),      # 4: Fz
            (-3, 5),          # 5: F7
            (-3, 8),          # 6: F8
            (-5, 1, 3, 4),    # 7: C3
            (-4, -1, 1, 4),   # 8: Cz
            (-6, -1, 4, 5),   # 9: C4
            (-5, -3),         # 10: T3
            (-4, 1, 4),       # 11: P3
            (-4, -1, 1),      # 12: Pz
            (-4, -1, 3),      # 13: P4
            (-8, -5),         # 14: T4
            (-4, 1),          # 15: O1
            (-3, -1),         # 16: O2
        ]
        self.idn_id = [
            (4, 5),           # 0: Fp1
            (3, 5),           # 1: Fp2
            (-1, 6, 8),       # 2: F3
            (-3, 5, 11),      # 3: F4
            (-4, -3, 3, 5),   # 4: Fz
            (-5, 2),          # 5: F7
            (-5, 3),          # 6: F8
            (-2, -3, 5),      # 7: C3
            (-6, -5, 3, 5),   # 8: Cz
            (-5, -3, 3),      # 9: C4
            (-8, 1),          # 10: T3
            (-1, -3, 5),      # 11: P3
            (-5, -3, 3, 4),   # 12: Pz
            (-5, 1, 2),       # 13: P4
            (-11, -1),        # 14: T4
            (-3,),            # 15: O1
            (-4,),            # 16: O2
        ]


class _Standard19(_Dataset):
    """Standard 10-20 19-channel layout (FTD_HC, AD_HC, FTD_AD, Irani, MCIvsHC)."""
    def __init__(self):
        self.dn_id = [
            (1, 3),           # 0: Fp1
            (-1, 5),          # 1: Fp2
            (1, 5),           # 2: F7
            (-1, 1, 5),       # 3: F3
            (-1, 1, 5),       # 4: Fz
            (-1, 1, 5),       # 5: F4
            (-1, 5),          # 6: F8
            (1, 5),           # 7: T3
            (-1, 1, 5),       # 8: C3
            (-1, 1, 5),       # 9: Cz
            (-1, 1, 5),       # 10: C4
            (-1, 5),          # 11: T4
            (1, 5),           # 12: T5
            (-1, 1, 4),       # 13: P3
            (-1, 1, 3),       # 14: Pz
            (-1, 1, 3),       # 15: P4
            (-1, 2),          # 16: T6
            (1,),             # 17: O1
            (-1,),            # 18: O2
        ]
        self.idn_id = [
            (2, 4),           # 0: Fp1
            (3, 5),           # 1: Fp2
            (4, 6),           # 2: F7
            (-3, -2, 4, 6),   # 3: F3
            (-4, -3, 4, 5),   # 4: Fz
            (-5, -4, 4, 6),   # 5: F4
            (-5, 4),          # 6: F8
            (4, 6),           # 7: T3
            (-6, -4, 4, 6),   # 8: C3
            (-6, -4, 4, 5),   # 9: Cz
            (-6, -4, 4, 6),   # 10: C4
            (-6, 4),          # 11: T4
            (4, 6),           # 12: T5
            (-6, -5, 4, 5),   # 13: P3
            (-6, -5, 3, 4),   # 14: Pz
            (-6, -5, 2, 3),   # 15: P4
            (-6, 2),          # 16: T6
            (),               # 17: O1
            (),               # 18: O2
        ]


class CAUEEG(_Dataset):
    """19-channel CAUEEG layout (different channel order from standard 10-20)."""
    def __init__(self):
        self.dn_id = [
            (5, 1),              # 0: Fp1
            (9, 15, -1, 1),      # 1: F3
            (9, 15, -1, 1),      # 2: C3
            (9, 15, -1, 1),      # 3: P3
            (5, -1),             # 4: O1
            (-5, 1),             # 5: Fp2
            (10, 7, -1, 1),      # 6: F4
            (10, 7, -1, 1),      # 7: C4
            (10, 7, -1, 1),      # 8: P4
            (-5, -1),            # 9: O2
            (-9, 1),             # 10: F7
            (-9, -1, 1),         # 11: T3
            (-9, -1),            # 12: T5
            (-7, 1),             # 13: F8
            (-7, -1, 1),         # 14: T4
            (-7, -1),            # 15: T6
            (-15, -10, 1),       # 16: Fz
            (-15, -10, -1, 1),   # 17: Cz
            (-15, -10, -1),      # 18: Pz
        ]
        self.idn_id = [
            (10, 16),            # 0: Fp1
            (4, 10, 16),         # 1: F3
            (8, 14, 10, 16),     # 2: C3
            (8, 14, 6),          # 3: P3
            (8, 14),             # 4: O1
            (11, 8),             # 5: Fp2
            (-6, 11, 8),         # 6: F4
            (9, 6, 11, 8),       # 7: C4
            (9, 6, -4),          # 8: P4
            (9, 6),              # 9: O2
            (-10, -8),           # 10: F7
            (-10, -8),           # 11: T3
            (-10, -8),           # 12: T5
            (-8, -6),            # 13: F8
            (-8, -6),            # 14: T4
            (-8, -6),            # 15: T6
            (-16, -11, -14, -9), # 16: Fz
            (-16, -11, -14, -9), # 17: Cz
            (-16, -11, -14, -9), # 18: Pz
        ]


def get_datastruct(dataset_name, task):
    """Return the correct dataset adjacency structure for the given dataset/task."""
    if dataset_name == 'GENEEG':
        return GENEEG_HC()
    elif dataset_name == 'MCIvsHC':
        return _Standard19()
    elif dataset_name == 'ADvsFTDvsHC':
        return _Standard19()
    elif dataset_name == 'CAUEEG':
        return CAUEEG()
    else:
        raise ValueError(f"No spatial adjacency structure defined for dataset '{dataset_name}'")


# ─────────────────────────────────────────────────────────────
# noisy_spatial_imputer — exact port from XAI_tools_auto
# ─────────────────────────────────────────────────────────────

class noisy_spatial_imputer:
    """
    Exact port of XAI_tools_auto's noisy_spatial_imputer.

    Solves a weighted linear system where each masked channel equals a
    weighted sum of its spatial neighbours, then adds IQR-scaled Gaussian
    noise and rescales to match the original data range.

    mask    : list of channel indices to impute
    dataset : adjacency structure (_Dataset subclass)
    noise   : noise standard deviation (use IQR std of the trial)
    """

    def __init__(self, mask, dataset, noise=0.01):
        self.dataset    = dataset
        self.noise      = noise
        self.imputed_id = list(mask)
        self.n_imputed  = len(mask)

        self.valid = np.ones(len(mask))
        for i, t in enumerate(self.imputed_id):
            dn_arr  = np.asarray(self.dataset.dn_id[t],  dtype=np.int64) \
                      if len(self.dataset.dn_id[t])  > 0 else np.asarray([], dtype=np.int64)
            idn_arr = np.asarray(self.dataset.idn_id[t], dtype=np.int64) \
                      if len(self.dataset.idn_id[t]) > 0 else np.asarray([], dtype=np.int64)

            arrays_to_concat = [a for a in (dn_arr, idn_arr) if a.size > 0]
            neighbors = np.concatenate(arrays_to_concat) if arrays_to_concat \
                        else np.asarray([], dtype=np.int64)
            if neighbors.size > 0:
                neighbors = neighbors + int(t)

            if np.any(np.isin(neighbors, self.imputed_id)):
                self.valid[i] = 0

    def _neighbor(self, idx):
        dnw  = 4 / len(self.dataset.dn_id[idx])  * _WEIGHTS[0] \
               if len(self.dataset.dn_id[idx])  > 0 else 0
        idnw = 4 / len(self.dataset.idn_id[idx]) * _WEIGHTS[1] \
               if len(self.dataset.idn_id[idx]) > 0 else 0

        neighbors  = [( dnw, dn  + idx) for dn  in self.dataset.dn_id[idx]]
        neighbors += [(idnw, idn + idx) for idn in self.dataset.idn_id[idx]]
        return neighbors

    def _construct_eqsys(self, trial):
        coords_to_vidx = np.zeros(trial.shape[0], dtype=np.int32)
        coords_to_vidx[self.imputed_id] = np.arange(self.n_imputed)

        A            = lil_matrix((self.n_imputed, self.n_imputed))
        b            = np.zeros((self.n_imputed, trial.shape[1]))
        sum_neighbors = np.ones(self.n_imputed)

        for i, target in enumerate(self.imputed_id):
            for weight, offset in self._neighbor(target):
                b[i, :] -= weight * trial[offset, :]
            if not self.valid[i]:
                A[i, coords_to_vidx[i]] = weight
                sum_neighbors[i]        = sum_neighbors[i] - weight

        A[np.arange(self.n_imputed), np.arange(self.n_imputed)] = -sum_neighbors
        return A, b

    def _return_imputed(self, trial):
        perturbed       = trial.copy()
        A, b            = self._construct_eqsys(trial)
        res             = np.array(spsolve(csc_matrix(A), b))
        perturbed[self.imputed_id, :] = res + np.random.randn(*res.shape) * self.noise

        # Rescale to match original data range (XAI_tools_auto convention)
        pert_q25, pert_q75   = np.percentile(perturbed, 25), np.percentile(perturbed, 75)
        pert_mask            = (perturbed >= pert_q25) & (perturbed <= pert_q75)
        pert_min, pert_max   = perturbed[pert_mask].min(), perturbed[pert_mask].max()

        trial_q25, trial_q75 = np.percentile(trial, 25), np.percentile(trial, 75)
        trial_mask           = (trial >= trial_q25) & (trial <= trial_q75)
        trial_min, trial_max = trial[trial_mask].min(), trial[trial_mask].max()

        pert_range = (perturbed.max() - perturbed.min()) or 1e-8
        perturbed  = (perturbed - pert_min) / pert_range
        perturbed  = perturbed * (trial_max - trial_min) + trial_min
        return perturbed
