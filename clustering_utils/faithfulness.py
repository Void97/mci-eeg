import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from numpy.fft import rfft, irfft, rfftfreq
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

from clustering_utils.constants import encode_groups
EXP_KEYS = ['spatial_ar', 'frequency_ar']

# Frequency masking: 20 steps at 5 % spectral-energy increments (matching XAI_tools_auto)
_N_FREQ_STEPS_AR   = 20   # frequency AR: 20 bins, k=1..20 (100% coverage)
_SQRT2 = np.sqrt(2)

# Log-frequency bin boundaries for frequency AR — matches XAI_tools_auto exactly:
#   F_MIN=1, F_MAX=100, N_BINS=20 → f_i = 1 * (100/1)^(i/20)
_F_MIN, _F_MAX = 1.0, 100.0
_LOG_BIN_BOUNDARIES = _F_MIN * (_F_MAX / _F_MIN) ** (np.arange(_N_FREQ_STEPS_AR + 1) / _N_FREQ_STEPS_AR)


def _compute_log_bins(n_time, sfreq):
    """
    Map positive rfft frequency indices (excluding DC and Nyquist) into 20
    log-spaced bins over [F_MIN, F_MAX) Hz — matches XAI_tools_auto's
    compute_log_grouping.  Returns list of 20 index arrays (ragged).
    """
    freqs   = rfftfreq(n_time, d=1.0 / sfreq)
    uq      = len(freqs) - 1                       # exclude Nyquist
    pos_idx = np.arange(1, uq)                     # positive freqs, no DC/Nyquist
    pos_freq = freqs[pos_idx]
    in_range = (pos_freq >= _F_MIN) & (pos_freq < _F_MAX)
    pos_idx  = pos_idx[in_range]
    pos_freq = pos_freq[in_range]
    assign   = np.clip(
        np.searchsorted(_LOG_BIN_BOUNDARIES, pos_freq, side='right') - 1,
        0, _N_FREQ_STEPS_AR - 1,
    )
    return [pos_idx[assign == i] for i in range(_N_FREQ_STEPS_AR)]


def _rank_logbins(centroid_fq, sub_bin_indices, mode):
    """
    Rank 20 log-bins by channel-mean gradient energy of the cluster centroid.
    Returns sorted bin indices (most → least salient for mode='mo').
    """
    grad_mag   = np.abs(centroid_fq).mean(axis=0)  # (n_freqs,)
    bin_energy = np.array([grad_mag[idx].sum() for idx in sub_bin_indices])
    return np.argsort(-bin_energy) if mode == 'mo' else np.argsort(bin_energy)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _weighted_avg(values, weights):
    total = sum(weights)
    result = None
    for v, w in zip(values, weights):
        term = v * (w / total)
        result = term if result is None else result + term
    return result


# ─────────────────────────────────────────────────────────────
# XAI_tools_auto find_neighbors (exact port)
# ─────────────────────────────────────────────────────────────

def _find_neighbors(den, grad, ratio, mode):
    """
    Exact port of XAI_tools_auto's find_neighbors (vectorised prefix-sum version).

    den  : (n_ch, n_freqs) — signal amplitude spectrum, positive freqs, DC excluded
    grad : (n_ch, n_freqs) — |gradient| spectrum, same shape
    ratio: float — fraction of total spectral energy to include in neighbourhood
    mode : 'mo' → argmax neighbourhood gradient (MoRF)
           'le' → argmin neighbourhood gradient (LeRF)

    Returns (center_idx, (lo, hi), edge_case) where lo/hi are 1-based absolute
    indices into the full rfft spectrum (matching XAI_tools_auto's convention).
    """
    den_avg  = np.abs(den).mean(axis=0)  if den.ndim  > 1 else np.abs(den)
    grad_avg = np.abs(grad).mean(axis=0) if grad.ndim > 1 else np.abs(grad)
    den_len  = den_avg.shape[-1]

    target    = den_avg.sum() * ratio
    target_hv = target / 2

    S = np.concatenate(([0.0], np.cumsum(den_avg)))
    G = np.concatenate(([0.0], np.cumsum(grad_avg)))
    c_idx = np.arange(den_len)

    accus = np.ones((den_len, 4)) * -den_len
    accus[:, 0], accus[:, 2] = den_avg, den_avg
    accus[:, 1][den_avg >= target] = 0
    accus[:, 3][den_avg >= target] = 0
    grad_accu = np.zeros((den_len, 2))
    grad_accu[:, 0] = grad_avg

    # Left side
    threshold_L = S[1:] - target_hv
    s_L         = np.searchsorted(S, threshold_L, side='right') - 1
    lid_natural = c_idx - s_L

    quirk     = (den_avg >= target_hv) & (den_avg < target)
    reached_L = s_L >= 0
    normal_L  = (den_avg < target_hv) & reached_L
    invalid_L = (den_avg < target_hv) & (~reached_L)

    accus[quirk,    1] = 1
    accus[normal_L, 1] = lid_natural[normal_L]
    accus[normal_L, 0]  = S[c_idx[normal_L] + 1] - S[c_idx[normal_L] - lid_natural[normal_L]]
    accus[invalid_L, 0] = S[c_idx[invalid_L] + 1]
    grad_accu[normal_L,  0] = G[c_idx[normal_L]  + 1] - G[c_idx[normal_L]  - lid_natural[normal_L]]
    grad_accu[invalid_L, 0] = G[c_idx[invalid_L] + 1]

    # Right side
    threshold_R = S[:-1] + target_hv
    s_R         = np.searchsorted(S, threshold_R, side='left')
    rid_natural = s_R - c_idx - 1

    reached_R = s_R <= den_len
    normal_R  = (den_avg < target_hv) & reached_R
    invalid_R = (den_avg < target_hv) & (~reached_R)

    accus[quirk,    3] = 1
    accus[normal_R, 3] = rid_natural[normal_R]
    accus[normal_R, 2]  = S[c_idx[normal_R] + rid_natural[normal_R] + 1] - S[c_idx[normal_R]]
    accus[invalid_R, 2] = S[-1] - S[c_idx[invalid_R]]
    grad_accu[normal_R,  1] = G[c_idx[normal_R]  + rid_natural[normal_R] + 1] - G[c_idx[normal_R]  + 1]
    grad_accu[invalid_R, 1] = G[-1] - G[c_idx[invalid_R] + 1]

    # Edge fallback
    for il in np.where(invalid_L)[0]:
        ll = 1
        while accus[il, 0] < target and il + ll < den_len:
            accus[il, 0] = S[il + ll]
            accus[il, 1] = ll * -1
            grad_accu[il, 0] = G[il + ll]
            ll += 1
    for ir in np.where(invalid_R)[0]:
        rr = 1
        while accus[ir, 2] < target and rr <= ir:
            accus[ir, 2] = S[-1] - S[ir - rr]
            accus[ir, 3] = rr * -1
            grad_accu[ir, 1] = G[-1] - G[ir - rr]
            rr += 1

    valid_lr = (accus[:, 3] >= 0) & (accus[:, 1] >= 0)
    neighborhood = np.zeros(den_len)
    neighborhood[valid_lr] = (grad_accu.sum(axis=-1)[valid_lr]
                               / (accus[valid_lr, 1] + accus[valid_lr, 3] + 1))
    for il in np.where(accus[:, 1] < 0)[0]:
        neighborhood[il] = grad_accu[il, 0] / (accus[il, 1] * -1 + il + 1)
    for ir in np.where(accus[:, 3] < 0)[0]:
        neighborhood[ir] = grad_accu[ir, 1] / (accus[ir, 3] * -1 + den_len - ir)

    m_id = int(neighborhood.argmax() if mode == 'mo' else neighborhood.argmin())

    if accus[m_id, 1] < 0:
        return m_id + 1, (1, m_id - int(accus[m_id, 1]) + 1), 0
    elif accus[m_id, 3] < 0:
        return m_id + 1, (m_id + int(accus[m_id, 3]) + 1, den_len), 1
    else:
        return m_id + 1, (m_id - int(accus[m_id, 1]) + 1,
                           m_id + int(accus[m_id, 3]) + 1), 2


# ─────────────────────────────────────────────────────────────
# AR replacement — PGD adversarial
# ─────────────────────────────────────────────────────────────

def _pgd_attack(net, x, y, epsilon, n_iter=20):
    """
    L∞ PGD matching XAI_tools_auto (mask_utils.pgd):
      - step = epsilon / n_iter
      - clamp to original data range [x_min, x_max] each step

    x: (n, 1, n_ch, n_time) on device.
    Returns adversarial examples as CPU numpy array, same shape as x.
    """
    x_min = x.min().item()
    x_max = x.max().item()
    step  = epsilon / n_iter
    x_adv = x.clone().detach()

    for _ in range(n_iter):
        x_adv = x_adv.detach().requires_grad_(True)
        out   = net(x_adv)
        if isinstance(out, tuple):
            out = out[0]
        F.cross_entropy(out, y).backward()

        with torch.no_grad():
            x_adv = torch.clamp(x_adv + step * x_adv.grad.sign(), x_min, x_max)

    return x_adv.detach().cpu()


def _auto_epsilon(net, X_f, y_f, device, n_classes, batch_size=256):
    """
    Find minimum epsilon driving trial-level accuracy to chance (1/n_classes).
    Exact port of XAI_tools_auto's exponential probe + binary search.
    Calibrated on cluster fold subjects to match evaluation scope.
    Returns (adv_numpy (n, n_ch, n_time), epsilon).
    """
    chance    = 1.0 / n_classes
    tolerance = 0.01

    def run(eps):
        x = torch.FloatTensor(X_f).unsqueeze(1).to(device)
        y = torch.LongTensor(y_f).to(device)
        adv = _pgd_attack(net, x, y, epsilon=eps).squeeze(1).numpy()
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(adv).unsqueeze(1), torch.LongTensor(y_f)),
            batch_size=batch_size,
        )
        correct = total = 0
        with torch.no_grad():
            for bx, by in loader:
                out = net(bx.to(device))
                if isinstance(out, tuple):
                    out = out[0]
                correct += (out.argmax(1).cpu() == by).sum().item()
                total   += len(by)
        return adv, correct / total

    eps, eps_max = 1e-4, 10.0
    best_adv = best_eps = acc = None

    # Phase 1: exponential probe
    while eps <= eps_max:
        adv, acc = run(eps)
        print(f"  [eps probe] eps={eps:.2e}  trial_acc={acc:.3f}")
        if abs(acc - chance) <= tolerance:
            best_adv, best_eps = adv, eps
            break
        if acc <= chance:
            break
        eps *= 2

    # Phase 2: binary search
    if best_adv is None and eps <= eps_max:
        lo, hi = eps / 2, eps
        for i in range(15):
            mid = (lo + hi) / 2
            adv, acc = run(mid)
            print(f"  [eps search {i+1}] eps={mid:.2e}  trial_acc={acc:.3f}")
            if abs(acc - chance) <= tolerance:
                best_adv, best_eps = adv, mid
                break
            lo, hi = (mid, hi) if acc > chance else (lo, mid)
        if best_adv is None:
            best_adv, best_eps = adv, mid

    if best_adv is None:
        raise RuntimeError(f"Auto epsilon search failed: exceeded eps_max={eps_max}")

    print(f"  Auto epsilon: {best_eps:.2e}  (trial_acc={acc:.3f})")
    return best_adv, best_eps


# ─────────────────────────────────────────────────────────────
# Masking application — spatial AR
# ─────────────────────────────────────────────────────────────

def _apply_spatial_ar(samples, adv_samples, channels):
    result = samples.copy()
    result[:, channels, :] = adv_samples[:, channels, :]
    return result


# ─────────────────────────────────────────────────────────────
# Frequency AR — log-bin approach (matches XAI_tools_auto exactly)
# ─────────────────────────────────────────────────────────────

def _apply_frequency_ar(samples, adv_samples, sub_bin_indices, bin_ranking, k):
    """
    Cumulatively mask the top-k ranked log-frequency bins with adversarial
    spectrum — exact port of XAI_tools_auto's freq_interp_test_log.

    sub_bin_indices: list of 20 rfft index arrays (from _compute_log_bins)
    bin_ranking:     (20,) sorted bin indices, most-salient first (from _rank_logbins)
    k:               number of bins to mask (1 … 20)
    """
    n_time = samples.shape[-1]

    # DC removal before FFT — matches XAI_tools_auto convention
    dc     = samples.mean(axis=-1,     keepdims=True)
    adv_dc = adv_samples.mean(axis=-1, keepdims=True)
    specs     = rfft(samples     - dc,     axis=-1).copy()
    adv_specs = rfft(adv_samples - adv_dc, axis=-1)

    for bin_id in bin_ranking[:k]:
        pos = sub_bin_indices[bin_id]
        specs[:, :, pos] = adv_specs[:, :, pos]

    return irfft(specs, n=n_time, axis=-1).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Accuracy functions
# ─────────────────────────────────────────────────────────────

def _subject_accuracy(net, samples_np, targets_np, groups, subject_ids, device,
                      batch_size=64):
    """Subject-level majority-vote accuracy."""
    net.eval()
    x = torch.FloatTensor(samples_np).unsqueeze(1)
    y = torch.LongTensor(targets_np)
    preds_all = []
    with torch.no_grad():
        for bx, _ in DataLoader(TensorDataset(x, y), batch_size=batch_size):
            out = net(bx.to(device))
            if isinstance(out, tuple):
                out = out[0]
            preds_all.append(out.argmax(dim=1).cpu().numpy())
    preds_all = np.concatenate(preds_all)

    correct = 0
    for sid in subject_ids:
        mask = groups == sid
        if not mask.any():
            continue
        pred = int(np.bincount(preds_all[mask]).argmax())
        correct += int(pred == int(targets_np[mask][0]))
    return correct / len(subject_ids)



# ─────────────────────────────────────────────────────────────
# Spearman consistency
# ─────────────────────────────────────────────────────────────

def _spearman_per_step(morf, lerf):
    """morf / lerf: (n_methods, n_steps) → per-step Spearman rho."""
    rhos = []
    for k in range(morf.shape[1]):
        rho = spearmanr(morf[:, k], lerf[:, k])[0]
        rhos.append(0.0 if np.isnan(rho) else float(rho))
    return rhos


def compute_spearman_consistency(results_path, dataset_name, task, model_name,
                                 best_iteration, gradient_methods,
                                 exp_key='spatial_ar'):
    """Load per-method faithfulness JSONs and compute Spearman consistency."""
    faith_dir = os.path.join(results_path, dataset_name, 'faithfulness')
    prefix    = f'{dataset_name}_{task}_{model_name}_iteration_{best_iteration}'

    morfs, lerfs, available = [], [], []
    for gm in gradient_methods:
        path = os.path.join(faith_dir, f'{prefix}_{gm}_{exp_key}_faithfulness.json')
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            data = json.load(fh)
        cluster_keys = [k for k in data if k != 'overall']
        if not cluster_keys:
            continue

        total_subj = sum(data[k]['n_subjects'] for k in cluster_keys)
        weights    = [data[k]['n_subjects'] / total_subj for k in cluster_keys]
        morf_avg   = _weighted_avg([np.array(data[k]['morf_curve']) for k in cluster_keys], weights)
        lerf_avg   = _weighted_avg([np.array(data[k]['lerf_curve']) for k in cluster_keys], weights)
        morfs.append(morf_avg);  lerfs.append(lerf_avg);  available.append(gm)

    if len(available) < 2:
        print(f"Spearman [{exp_key}]: fewer than 2 methods available, skipping.")
        return

    rhos = _spearman_per_step(np.stack(morfs), np.stack(lerfs))
    mean_rho, std_rho = float(np.mean(rhos)), float(np.std(rhos))
    print(f"Spearman [{exp_key}] ({len(available)} methods): rho = {mean_rho:.3f} ± {std_rho:.3f}")

    path = os.path.join(faith_dir, f'{prefix}_{exp_key}_spearman.json')
    with open(path, 'w') as fh:
        json.dump({'exp_key': exp_key, 'methods': available,
                   'spearman_mean': mean_rho, 'spearman_std': std_rho,
                   'spearman_per_k': rhos}, fh, indent=4)
    print(f"Saved Spearman results to {path}")


# ─────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────

class FaithfulnessEvaluator:
    """
    Faithfulness evaluation for EEG subgroup clusters.

    Feature ranking uses raw gradient saliency (not PSD), enabling bin-level
    frequency masking via XAI_tools_auto's find_neighbors algorithm.

    Experiments:
      • spatial_ar   — mask channels, k = 1 … n_channels
      • frequency_ar — mask 20 log-spaced frequency bins

    Two baselines per fold:
      • baseline_cluster: subject-level accuracy on cluster subjects (= 1.0 by design)
      • baseline_fold:    trial-level accuracy on positive-class fold test subjects
    """

    def __init__(self, ch_names, sfreq=200):
        self.ch_names   = ch_names
        self.sfreq      = sfreq
        self.n_channels = len(ch_names)

    def _rank_channels(self, centroid_fq):
        """centroid_fq: (n_ch, n_freqs) complex FFT centroid → channel indices sorted most-salient first."""
        return np.argsort(np.abs(centroid_fq).mean(axis=-1))[::-1].tolist()

    def _compute_pgd(self, net, X_f, y_f, device):
        """Run auto epsilon search then return adversarial examples (n, n_ch, n_time)."""
        n_classes = 2  # binary classification throughout
        adv_f, _ = _auto_epsilon(net, X_f, y_f, device, n_classes)
        return adv_f

    def _compute_metrics(self, morf_curve, lerf_curve, baseline, n_classes):
        chance = 1.0 / n_classes
        mc = np.clip(morf_curve, chance, baseline)
        lc = np.clip(lerf_curve, chance, baseline)
        return (float(np.mean(1.0 - mc)),
                float(np.mean(lc)),
                float(np.mean(np.clip(lc - mc, 0, None))))

    def evaluate(self, model, samples, targets, groups,
                 gradients_list, cluster_labels, subject_ids_list,
                 dataset_name, task, best_iteration, gradient_method,
                 subject_fold_map=None,
                 output_dir='results/clustering'):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Split BEFORE encoding to match training fold assignments
        groups = encode_groups(dataset_name, groups)
        subject_ids_list  = np.array(subject_ids_list, dtype=np.int64)
        cluster_labels    = np.array(cluster_labels)

        sfm = ({int(k): int(v) for k, v in subject_fold_map.items()}
               if subject_fold_map is not None
               else {int(sid): 0 for sid in subject_ids_list})

        # Load one model per fold
        folds_needed = sorted({sfm[int(sid)] for sid in subject_ids_list if int(sid) in sfm})
        fold_nets = {}
        for fold in folds_needed:
            net_f = model['class'](**model['kwargs'], tsne=False)
            wp = (f'results/weights/{dataset_name}/'
                  f'{dataset_name}_{task}_{model["name"]}_iteration_{best_iteration}'
                  f'_fold_{fold}_best_weights.pth')
            net_f.load_state_dict(torch.load(wp, map_location=device))
            net_f.to(device).eval()
            fold_nets[fold] = net_f
            print(f"Loaded fold {fold} weights from {wp}")


        # Cluster centroids in RAW gradient space (n_ch, n_time)
        unique_clusters  = np.unique(cluster_labels)
        subj_to_cluster  = dict(zip(subject_ids_list, cluster_labels))
        # Cluster centroids: FFT of mean gradient per cluster (matches XAI_tools_auto
        # which uses fft(grad) per trial; abs is taken internally by find_neighbors
        # and rank_logbins, so the sign of the input does not affect ranking).
        centroids = {
            c: rfft(gradients_list[cluster_labels == c].mean(axis=0), axis=-1)
            for c in unique_clusters
        }  # (n_ch, n_rfft_freqs) — complex FFT of mean gradient

        # Log-frequency bin layout — computed once per dataset/time resolution
        sub_bin_indices = _compute_log_bins(samples.shape[-1], self.sfreq)

        trial_mask = np.isin(groups, subject_ids_list)
        samples_c, targets_c, groups_c = (samples[trial_mask],
                                           targets[trial_mask],
                                           groups[trial_mask])
        n_classes   = len(np.unique(targets))
        all_results = {k: {} for k in EXP_KEYS}

        for c in unique_clusters:
            cluster_subjects = [sid for sid in subject_ids_list if subj_to_cluster[sid] == c]
            centroid_fq    = centroids[c]              # (n_ch, n_rfft_freqs) complex
            ranked_ch      = self._rank_channels(centroid_fq)
            bin_ranking_mo = _rank_logbins(centroid_fq, sub_bin_indices, 'mo')
            bin_ranking_le = _rank_logbins(centroid_fq, sub_bin_indices, 'le')

            fold_to_subjects = defaultdict(list)
            for sid in cluster_subjects:
                fold_to_subjects[sfm.get(int(sid), 0)].append(int(sid))
            print(f"Cluster {c}: {len(cluster_subjects)} subjects "
                  f"across {len(fold_to_subjects)} fold(s)")

            fold_data = {k: dict(morf=[], lerf=[], weights=[], bc=[])
                         for k in EXP_KEYS}

            curves_dir = os.path.join(output_dir, dataset_name, 'faithfulness_curves')
            os.makedirs(curves_dir, exist_ok=True)

            for fold in sorted(fold_to_subjects):
                fold_subs = fold_to_subjects[fold]

                # ── Curve cache (skip ALL masking+inference on hit) ─────────
                curve_cache = os.path.join(
                    curves_dir,
                    f'{dataset_name}_{task}_{model["name"]}_iter_{best_iteration}'
                    f'_{gradient_method}_cluster{c}_fold{fold}.json'
                )
                if os.path.exists(curve_cache):
                    with open(curve_cache) as fh:
                        cached = json.load(fh)
                    for key in EXP_KEYS:
                        fd = fold_data[key]
                        fd['morf'].append(np.array(cached[key]['morf']))
                        fd['lerf'].append(np.array(cached[key]['lerf']))
                        fd['weights'].append(len(fold_subs))
                        fd['bc'].append(cached[key]['bc'])
                    print(f"  Loaded curves from cache: cluster {c}, fold {fold}")
                    continue
                # ────────────────────────────────────────────────────────────

                net_f = fold_nets[fold]
                f_mask         = np.isin(groups_c, fold_subs)
                X_f, y_f, g_f = samples_c[f_mask], targets_c[f_mask], groups_c[f_mask]

                bc    = _subject_accuracy(net_f, X_f, y_f, g_f, fold_subs, device)
                adv_f = self._compute_pgd(net_f, X_f, y_f, device)

                f_sp = rfft(X_f, axis=-1)

                maskers = [
                    ('spatial_ar',   self.n_channels,
                     lambda k: _apply_spatial_ar(X_f, adv_f, ranked_ch[:k]),
                     lambda k: _apply_spatial_ar(X_f, adv_f, ranked_ch[-k:])),
                    ('frequency_ar', _N_FREQ_STEPS_AR,
                     lambda k: _apply_frequency_ar(X_f, adv_f,
                                                   sub_bin_indices, bin_ranking_mo, k),
                     lambda k: _apply_frequency_ar(X_f, adv_f,
                                                   sub_bin_indices, bin_ranking_le, k)),
                ]

                fold_curves = {}
                for key, n_steps, morf_fn, lerf_fn in maskers:
                    morf_list, lerf_list = [], []
                    for k in range(1, n_steps + 1):
                        morf_list.append(_subject_accuracy(net_f, morf_fn(k),
                                                           y_f, g_f, fold_subs, device))
                        lerf_list.append(_subject_accuracy(net_f, lerf_fn(k),
                                                           y_f, g_f, fold_subs, device))
                    fold_curves[key] = {'morf': morf_list, 'lerf': lerf_list, 'bc': bc}
                    fd = fold_data[key]
                    fd['morf'].append(np.array(morf_list))
                    fd['lerf'].append(np.array(lerf_list))
                    fd['weights'].append(len(fold_subs))
                    fd['bc'].append(bc)

                # Save curve cache for this fold
                with open(curve_cache, 'w') as fh:
                    json.dump(fold_curves, fh)
                print(f"  Saved curves cache: cluster {c}, fold {fold}")

            # Aggregate folds → cluster result
            for key in EXP_KEYS:
                fd  = fold_data[key]
                ws  = fd['weights']
                morf_curve = _weighted_avg(fd['morf'], ws).tolist()
                lerf_curve = _weighted_avg(fd['lerf'], ws).tolist()
                baseline   = float(_weighted_avg(fd['bc'], ws))

                aoc, auc, abc = self._compute_metrics(morf_curve, lerf_curve, baseline, n_classes)

                all_results[key][int(c)] = {
                    'AOC': aoc, 'AUC': auc, 'ABC': abc,
                    'baseline': baseline,
                    'morf_curve': [float(v) for v in morf_curve],
                    'lerf_curve': [float(v) for v in lerf_curve],
                    'n_subjects': len(cluster_subjects),
                }
                print(f"  [{key}] Cluster {c}: "
                      f"AOC={aoc:.3f}  AUC={auc:.3f}  ABC={abc:.3f}  "
                      f"(n={len(cluster_subjects)}, baseline={baseline:.3f})")

        # Overall weighted average
        for key in EXP_KEYS:
            total = sum(all_results[key][int(c)]['n_subjects'] for c in unique_clusters)
            all_results[key]['overall'] = {
                metric: float(sum(
                    all_results[key][int(c)][metric]
                    * all_results[key][int(c)]['n_subjects'] / total
                    for c in unique_clusters
                ))
                for metric in ('AOC', 'AUC', 'ABC')
            }
            ov = all_results[key]['overall']
            print(f"  [{key}] Overall: AOC={ov['AOC']:.3f}  AUC={ov['AUC']:.3f}  ABC={ov['ABC']:.3f}")

        # Save — one JSON per experiment type
        save_dir = os.path.join(output_dir, dataset_name, 'faithfulness')
        os.makedirs(save_dir, exist_ok=True)
        prefix = (f'{dataset_name}_{task}_{model["name"]}'
                  f'_iteration_{best_iteration}_{gradient_method}')
        for key in EXP_KEYS:
            path = os.path.join(save_dir, f'{prefix}_{key}_faithfulness.json')
            with open(path, 'w') as fh:
                json.dump(all_results[key], fh, indent=4)
        print(f"Saved faithfulness results to {save_dir}/")

        return all_results
