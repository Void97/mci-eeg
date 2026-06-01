import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from numpy.fft import rfft, irfft, rfftfreq
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
from torch.utils.data import DataLoader, TensorDataset

from clustering_utils.constants import encode_groups, get_fold_splits

EXP_KEYS = ['spatial_ar', 'spatial_road', 'frequency_ar', 'frequency_road']

# Frequency masking: 20 steps at 5 % spectral-energy increments (matching XAI_tools_auto)
_N_FREQ_STEPS = 20
_SQRT2 = np.sqrt(2)

# Log-frequency bin boundaries for frequency AR — matches XAI_tools_auto exactly:
#   F_MIN=1, F_MAX=100, N_BINS=20 → f_i = 1 * (100/1)^(i/20)
_F_MIN, _F_MAX = 1.0, 100.0
_LOG_BIN_BOUNDARIES = _F_MIN * (_F_MAX / _F_MIN) ** (np.arange(_N_FREQ_STEPS + 1) / _N_FREQ_STEPS)


def _poly3(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


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
        0, _N_FREQ_STEPS - 1,
    )
    return [pos_idx[assign == i] for i in range(_N_FREQ_STEPS)]


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

def _pgd_attack(net, x, y, epsilon, alpha=2.0, n_iter=10):
    """
    Untargeted PGD with L2 ball constraint.
    x: (n, 1, n_ch, n_time) — model input format.
    Returns adversarial examples as CPU tensor, same shape as x.
    """
    device = x.device
    eps    = torch.tensor(epsilon, dtype=torch.float32, device=device)
    x_adv  = x.clone().detach() + torch.zeros_like(x).uniform_(-1e-3, 1e-3)

    for _ in range(n_iter):
        x_adv = x_adv.detach().requires_grad_(True)
        out   = net(x_adv)
        if isinstance(out, tuple):
            out = out[0]
        F.cross_entropy(out, y).backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta  = x_adv - x
            norms  = delta.norm(p=2, dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
            delta  = delta * torch.min(torch.ones_like(norms), eps / norms)
            x_adv  = x + delta

    return x_adv.detach().cpu()


# ─────────────────────────────────────────────────────────────
# ROAD replacement — spatial
# ─────────────────────────────────────────────────────────────

def _road_spatial_replace(samples, channels_to_mask, corr_matrix=None):
    """
    Replace masked channels with correlation-weighted average of remaining
    channels + IQR-scaled noise.
    samples: (n_trials, n_channels, n_time)
    corr_matrix: optional precomputed (n_ch, n_ch) — pass to avoid recomputation.
    """
    channels_to_mask = list(channels_to_mask)
    n_trials, n_ch, n_time = samples.shape
    keep_idx = [c for c in range(n_ch) if c not in channels_to_mask]
    if not keep_idx:
        return samples.copy()

    if corr_matrix is None:
        flat   = samples.transpose(1, 0, 2).reshape(n_ch, -1)
        flat_z = flat - flat.mean(axis=1, keepdims=True)
        norms  = np.linalg.norm(flat_z, axis=1, keepdims=True).clip(min=1e-8)
        corr_matrix = (flat_z / norms) @ (flat_z / norms).T

    result = samples.copy()
    for c in channels_to_mask:
        w = np.abs(corr_matrix[c, keep_idx])
        w /= w.sum() + 1e-8
        result[:, c, :] = (
            (samples[:, keep_idx, :] * w[None, :, None]).sum(axis=1)
            + np.random.randn(n_trials, n_time) * samples[:, c, :].std() * 0.01
        )
    return result


# ─────────────────────────────────────────────────────────────
# Masking application — spatial AR
# ─────────────────────────────────────────────────────────────

def _apply_spatial_ar(samples, adv_samples, channels):
    result = samples.copy()
    result[:, channels, :] = adv_samples[:, channels, :]
    return result


# ─────────────────────────────────────────────────────────────
# Frequency masking — shared neighbour-finding helpers
# ─────────────────────────────────────────────────────────────

def _freq_neighbor_region(specs_t, centroid_fq, ratio, mode):
    """
    For a single trial's spectrum `specs_t` (n_ch, n_freqs) and the
    cluster-centroid gradient spectrum `centroid_fq` (n_ch, n_freqs),
    call _find_neighbors and return the absolute (lo, hi) indices
    into the full rfft array.

    uq = n_freqs - 1 excludes the Nyquist bin, matching XAI_tools_auto's
    convention (uq_freqs = index of -sfreq/2 in full FFT = n//2).
    """
    uq = specs_t.shape[-1] - 1          # exclude Nyquist (matches XAI_tools_auto)
    _, (lo, hi), _ = _find_neighbors(
        specs_t[:, 1:uq],
        np.abs(centroid_fq[:, 1:uq]),
        ratio, mode,
    )
    return lo, hi


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
# Frequency ROAD — exact XAI_tools_auto polynomial approach
# ─────────────────────────────────────────────────────────────

def _road_frequency_replace(samples, centroid_fq, ratio, mode, sfreq,
                             specs=None, cache_path=None, device=None):
    """
    XAI_tools_auto ROAD for frequency domain.

    Polynomial fitting is batched across all trials in one GPU lstsq call
    (replaces per-trial curve_fit — ~1000× faster on GPU).  Boundary
    conditions are dropped since they affect only 2 of 399 fit points and
    have negligible effect on the polynomial shape.  CPU fallback is used
    when device is None or CUDA is unavailable.

    cache_path: optional .npy — loaded directly if it exists, skipping all
                computation (matches XAI_tools_auto's .mat caching convention).
    centroid_fq: (n_ch, n_freqs) — rfft of cluster centroid gradient
    """
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)

    n_trials, _, n_time = samples.shape
    f     = rfftfreq(n_time, d=1.0 / sfreq)
    lowfq = 1
    uq    = len(f) - 1   # exclude Nyquist

    out     = rfft(samples, axis=-1).copy() if specs is None else specs.copy()
    grad_fq = np.abs(centroid_fq)
    x_full  = 1.0 / f[lowfq:uq]   # (n_fit,) — 1/f axis

    # ── Single find_neighbors on cluster-mean spectrum ────────────────────
    # Using the cluster-mean spectrum (instead of per-trial) gives one
    # shared (lo, hi) for all trials — consistent with cluster-level
    # evaluation and eliminates the per-trial find_neighbors loop.
    mean_spec = out.mean(axis=0)   # (n_ch, n_freqs) — mean over trials
    lo, hi = _freq_neighbor_region(mean_spec, centroid_fq, ratio, mode)
    x_mask = 1.0 / f[lo:hi + 1].clip(1e-8)   # (n_mask,)

    # ── GPU batch lstsq + vectorised polynomial evaluation ────────────────
    use_gpu = (device is not None and torch.cuda.is_available())
    if use_gpu:
        xt = torch.tensor(x_full, dtype=torch.float32, device=device)
        A  = torch.stack([xt**3, xt**2, xt, torch.ones_like(xt)], dim=1)  # (n_fit, 4)
        Y  = torch.tensor(
            np.abs(out[:, :, lowfq:uq]).mean(axis=1) / _SQRT2,
            dtype=torch.float32, device=device,
        )  # (n_trials, n_fit)
        driver   = 'gels' if device.type == 'cuda' else 'gelsd'
        popt_all = torch.linalg.lstsq(A, Y.T, driver=driver).solution  # (4, n_trials)

        xmt = torch.tensor(x_mask, dtype=torch.float32, device=device)
        Am  = torch.stack([xmt**3, xmt**2, xmt, torch.ones_like(xmt)], dim=1)  # (n_mask, 4)
        impt_all = (Am @ popt_all).cpu().numpy()  # (n_mask, n_trials)

        # Fully vectorised sign preservation and replacement
        real_sign = np.sign(out[:, :, lo:hi + 1].real); real_sign[real_sign == 0] = 1
        imag_sign = np.sign(out[:, :, lo:hi + 1].imag); imag_sign[imag_sign == 0] = 1
        # impt_all: (n_mask, n_trials) → transpose to (n_trials, n_mask)
        imp = impt_all.T[:, np.newaxis, :]  # (n_trials, 1, n_mask) — broadcast over channels
        out[:, :, lo:hi + 1] = (imp * real_sign) + 1j * (imp * imag_sign)
    else:
        # CPU fallback — original curve_fit per trial with boundary conditions
        lo_positive = f[lo] > 0
        bc_lo_x = 1.0 / (f[lo] if lo_positive else f[lowfq])
        bc_lo_v = grad_fq[:, lo if lo_positive else lowfq].mean() / _SQRT2
        bc_hi_x = 1.0 / f[hi].clip(1e-8)
        bc_hi_v = grad_fq[:, hi].mean() / _SQRT2

        for t in range(n_trials):
            y_full = np.abs(out[t, :, lowfq:uq]).mean(axis=0) / _SQRT2

            def overf(x, a, b, c, d):
                yfit = _poly3(x, a, b, c, d)
                if lo_positive:
                    yfit[x == bc_lo_x] = bc_lo_v
                else:
                    yfit[x > 1.0 / f[lowfq]] = bc_lo_v
                yfit[x == bc_hi_x] = bc_hi_v
                return yfit

            try:
                popt, _ = curve_fit(overf, x_full, y_full, maxfev=5000)
                impt = _poly3(x_mask, *popt)
            except RuntimeError:
                impt = np.full(len(x_mask), y_full.mean())

            real_sign = np.sign(out[t, :, lo:hi + 1].real); real_sign[real_sign == 0] = 1
            imag_sign = np.sign(out[t, :, lo:hi + 1].imag); imag_sign[imag_sign == 0] = 1
            out[t, :, lo:hi + 1] = (impt * real_sign) + 1j * (impt * imag_sign)

    result = irfft(out, n=n_time, axis=-1).astype(np.float32)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, result)
    return result


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


def _trial_accuracy(net, samples_np, targets_np, device, batch_size=64):
    """Trial-level accuracy for fold-level baseline."""
    net.eval()
    x = torch.FloatTensor(samples_np).unsqueeze(1)
    y = torch.LongTensor(targets_np)
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in DataLoader(TensorDataset(x, y), batch_size=batch_size):
            out = net(bx.to(device))
            if isinstance(out, tuple):
                out = out[0]
            correct += (out.argmax(dim=1).cpu() == by).sum().item()
            total   += len(by)
    return correct / total if total > 0 else 0.0


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
      • spatial_ar / spatial_road   — mask channels, k = 1 … n_channels
      • frequency_ar / frequency_road — mask frequency region, k = 1 … 20
        (each step k masks the region containing k×5 % of spectral energy)

    Two baselines per fold:
      • baseline_cluster: subject-level accuracy on cluster subjects (= 1.0 by design)
      • baseline_fold:    trial-level accuracy on positive-class fold test subjects
    """

    def __init__(self, ch_names, sfreq=200, pgd_alpha=2.0, pgd_n_iter=10):
        self.ch_names   = ch_names
        self.sfreq      = sfreq
        self.pgd_alpha  = pgd_alpha
        self.pgd_n_iter = pgd_n_iter
        self.n_channels = len(ch_names)

    def _rank_channels(self, centroid_grad):
        """centroid_grad: (n_ch, n_time) → channel indices sorted most-salient first."""
        return np.argsort(np.abs(centroid_grad).mean(axis=-1))[::-1].tolist()

    def _compute_pgd(self, net, X_f, y_f, epsilon, device, batch_size=256):
        chunks = []
        for i in range(0, len(X_f), batch_size):
            xb = torch.FloatTensor(X_f[i:i + batch_size]).unsqueeze(1).to(device)
            yb = torch.LongTensor(y_f[i:i + batch_size]).to(device)
            chunks.append(
                _pgd_attack(net, xb, yb, epsilon=epsilon,
                            alpha=self.pgd_alpha, n_iter=self.pgd_n_iter)
                .squeeze(1).numpy()
            )
        return np.concatenate(chunks, axis=0)

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

        groups            = encode_groups(dataset_name, groups)
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

        fold_test_indices = get_fold_splits(samples, targets, groups)

        # Cluster centroids in RAW gradient space (n_ch, n_time)
        unique_clusters  = np.unique(cluster_labels)
        subj_to_cluster  = dict(zip(subject_ids_list, cluster_labels))
        raw_centroids    = {
            c: gradients_list[cluster_labels == c].mean(axis=0)
            for c in unique_clusters
        }  # each centroid: (n_ch, n_time)
        # FFT of centroids for frequency masking
        centroid_fqs = {
            c: rfft(raw_centroids[c], axis=-1)   # (n_ch, n_freqs)
            for c in unique_clusters
        }

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
            ranked_ch      = self._rank_channels(raw_centroids[c])
            centroid_fq    = centroid_fqs[c]
            bin_ranking_mo = _rank_logbins(centroid_fq, sub_bin_indices, 'mo')
            bin_ranking_le = _rank_logbins(centroid_fq, sub_bin_indices, 'le')

            fold_to_subjects = defaultdict(list)
            for sid in cluster_subjects:
                fold_to_subjects[sfm.get(int(sid), 0)].append(int(sid))
            print(f"Cluster {c}: {len(cluster_subjects)} subjects "
                  f"across {len(fold_to_subjects)} fold(s)")

            fold_data = {k: dict(morf=[], lerf=[], weights=[], bc=[], bf=[])
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
                        fd['bf'].append(cached[key]['bf'])
                    print(f"  Loaded curves from cache: cluster {c}, fold {fold}")
                    continue
                # ────────────────────────────────────────────────────────────

                net_f = fold_nets[fold]
                f_mask         = np.isin(groups_c, fold_subs)
                X_f, y_f, g_f = samples_c[f_mask], targets_c[f_mask], groups_c[f_mask]

                bc = _subject_accuracy(net_f, X_f, y_f, g_f, fold_subs, device)
                target_class = 0 if task == 'MCI vs Dementia' else 1
                test_idx     = fold_test_indices[fold]
                pos_mask     = targets[test_idx] == target_class
                bf = _trial_accuracy(net_f, samples[test_idx][pos_mask],
                                     targets[test_idx][pos_mask], device)

                epsilon = float(np.abs(X_f).max())
                adv_f   = self._compute_pgd(net_f, X_f, y_f, epsilon, device)

                # Precompute once per fold
                flat   = X_f.transpose(1, 0, 2).reshape(self.n_channels, -1)
                flat_z = flat - flat.mean(axis=1, keepdims=True)
                norms  = np.linalg.norm(flat_z, axis=1, keepdims=True).clip(min=1e-8)
                corr_m = (flat_z / norms) @ (flat_z / norms).T
                f_sp   = rfft(X_f, axis=-1)

                road_cache_dir    = os.path.join(output_dir, dataset_name, 'road_freq_cache')
                road_cache_prefix = (f'{dataset_name}_{task}_{model["name"]}'
                                     f'_iter_{best_iteration}_{gradient_method}'
                                     f'_cluster{c}_fold{fold}')

                maskers = [
                    ('spatial_ar',    self.n_channels,
                     lambda k: _apply_spatial_ar(X_f, adv_f, ranked_ch[:k]),
                     lambda k: _apply_spatial_ar(X_f, adv_f, ranked_ch[-k:])),
                    ('spatial_road',  self.n_channels,
                     lambda k: _road_spatial_replace(X_f, ranked_ch[:k],  corr_m),
                     lambda k: _road_spatial_replace(X_f, ranked_ch[-k:], corr_m)),
                    ('frequency_ar',  _N_FREQ_STEPS,
                     lambda k: _apply_frequency_ar(X_f, adv_f,
                                                   sub_bin_indices, bin_ranking_mo, k),
                     lambda k: _apply_frequency_ar(X_f, adv_f,
                                                   sub_bin_indices, bin_ranking_le, k)),
                    ('frequency_road', _N_FREQ_STEPS,
                     lambda k: _road_frequency_replace(
                         X_f, centroid_fq, k / _N_FREQ_STEPS, 'mo', self.sfreq, f_sp,
                         os.path.join(road_cache_dir,
                                      f'{road_cache_prefix}_k{k:02d}_mo.npy'),
                         device=device),
                     lambda k: _road_frequency_replace(
                         X_f, centroid_fq, k / _N_FREQ_STEPS, 'le', self.sfreq, f_sp,
                         os.path.join(road_cache_dir,
                                      f'{road_cache_prefix}_k{k:02d}_le.npy'),
                         device=device)),
                ]

                fold_curves = {}
                for key, n_steps, morf_fn, lerf_fn in maskers:
                    morf_list, lerf_list = [], []
                    for k in range(1, n_steps + 1):
                        morf_list.append(_subject_accuracy(net_f, morf_fn(k),
                                                           y_f, g_f, fold_subs, device))
                        lerf_list.append(_subject_accuracy(net_f, lerf_fn(k),
                                                           y_f, g_f, fold_subs, device))
                    fold_curves[key] = {'morf': morf_list, 'lerf': lerf_list,
                                        'bc': bc, 'bf': bf}
                    fd = fold_data[key]
                    fd['morf'].append(np.array(morf_list))
                    fd['lerf'].append(np.array(lerf_list))
                    fd['weights'].append(len(fold_subs))
                    fd['bc'].append(bc)
                    fd['bf'].append(bf)

                # Save curve cache for this fold
                with open(curve_cache, 'w') as fh:
                    json.dump(fold_curves, fh)
                print(f"  Saved curves cache: cluster {c}, fold {fold}")

            # Aggregate folds → cluster result
            for key in EXP_KEYS:
                fd  = fold_data[key]
                ws  = fd['weights']
                morf_curve       = _weighted_avg(fd['morf'], ws).tolist()
                lerf_curve       = _weighted_avg(fd['lerf'], ws).tolist()
                baseline_cluster = float(_weighted_avg(fd['bc'], ws))
                baseline_fold    = float(_weighted_avg(fd['bf'], ws))

                aoc_c, auc_c, abc_c = self._compute_metrics(
                    morf_curve, lerf_curve, baseline_cluster, n_classes)
                aoc_f, auc_f, abc_f = self._compute_metrics(
                    morf_curve, lerf_curve, baseline_fold, n_classes)

                all_results[key][int(c)] = {
                    'AOC_cluster_baseline': aoc_c, 'AUC_cluster_baseline': auc_c,
                    'ABC_cluster_baseline': abc_c, 'AOC_fold_baseline':    aoc_f,
                    'AUC_fold_baseline':    auc_f, 'ABC_fold_baseline':    abc_f,
                    'baseline_cluster': baseline_cluster, 'baseline_fold': baseline_fold,
                    'morf_curve': [float(v) for v in morf_curve],
                    'lerf_curve': [float(v) for v in lerf_curve],
                    'n_subjects': len(cluster_subjects),
                }
                print(f"  [{key}] Cluster {c}: "
                      f"AOC(cluster)={aoc_c:.3f}  AOC(fold)={aoc_f:.3f}  "
                      f"AUC(cluster)={auc_c:.3f}  ABC(cluster)={abc_c:.3f}  "
                      f"(n={len(cluster_subjects)}, "
                      f"baseline_cluster={baseline_cluster:.3f}, "
                      f"baseline_fold={baseline_fold:.3f})")

        # Overall weighted average
        for key in EXP_KEYS:
            total = sum(all_results[key][int(c)]['n_subjects'] for c in unique_clusters)
            all_results[key]['overall'] = {
                metric: float(sum(
                    all_results[key][int(c)][metric]
                    * all_results[key][int(c)]['n_subjects'] / total
                    for c in unique_clusters
                ))
                for metric in ('AOC_cluster_baseline', 'AUC_cluster_baseline',
                               'ABC_cluster_baseline', 'AOC_fold_baseline',
                               'AUC_fold_baseline',    'ABC_fold_baseline')
            }
            ov = all_results[key]['overall']
            print(f"  [{key}] Overall: "
                  f"AOC(cluster)={ov['AOC_cluster_baseline']:.3f}  "
                  f"AOC(fold)={ov['AOC_fold_baseline']:.3f}  "
                  f"AUC(cluster)={ov['AUC_cluster_baseline']:.3f}  "
                  f"ABC(cluster)={ov['ABC_cluster_baseline']:.3f}")

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
