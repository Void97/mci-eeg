import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from numpy.fft import rfft, irfft, rfftfreq
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold

from clustering_utils.clustering_20260424 import PSDConverter


# ─────────────────────────────────────────────────────────────
# AR replacement — PGD adversarial
# ─────────────────────────────────────────────────────────────

def _pgd_attack(net, x, y, epsilon, alpha=2.0, n_iter=10):
    """
    Untargeted PGD attack with L2 ball constraint.
    x: (n, 1, n_channels, n_time) — model input format
    y: (n,) long tensor
    Returns adversarial examples as CPU tensor, same shape as x.
    """
    device = x.device
    eps = torch.tensor(epsilon, dtype=torch.float32, device=device)
    x_adv = x.clone().detach() + torch.zeros_like(x).uniform_(-1e-3, 1e-3)

    for _ in range(n_iter):
        x_adv = x_adv.detach().requires_grad_(True)
        out = net(x_adv)
        if isinstance(out, tuple):
            out = out[0]
        loss = F.cross_entropy(out, y)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = x_adv - x
            norms = delta.norm(p=2, dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
            delta = delta * torch.min(torch.ones_like(norms), eps / norms)
            x_adv = x + delta

    return x_adv.detach().cpu()


# ─────────────────────────────────────────────────────────────
# ROAD replacement
# ─────────────────────────────────────────────────────────────

def _road_spatial_replace(samples, channels_to_mask):
    """
    ROAD spatial imputer: replace each masked channel with a
    correlation-weighted average of the remaining channels + IQR-scaled noise.
    samples: (n_trials, n_channels, n_time)
    """
    result = samples.copy()
    n_trials, n_ch, n_time = samples.shape
    channels_to_mask = list(channels_to_mask)
    keep_idx = [c for c in range(n_ch) if c not in channels_to_mask]

    if not keep_idx:
        return result

    # Batch-level correlation matrix (faster than per-trial)
    flat = samples.reshape(n_ch, -1)
    flat_z = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flat_z, axis=1, keepdims=True).clip(min=1e-8)
    corr_matrix = (flat_z / norms) @ (flat_z / norms).T  # (n_ch, n_ch)

    for c in channels_to_mask:
        w = np.abs(corr_matrix[c, keep_idx])
        w = w / (w.sum() + 1e-8)
        # Vectorised over trials: (n_trials, n_time)
        imputed = (samples[:, keep_idx, :] * w[None, :, None]).sum(axis=1)
        noise_std = samples[:, c, :].std() * 0.01
        result[:, c, :] = imputed + np.random.randn(n_trials, n_time) * noise_std

    return result


def _road_frequency_replace(samples, bands_to_mask, bands, sfreq):
    """
    ROAD frequency imputer: replace masked frequency bands with amplitude
    interpolated from adjacent kept bands (log-frequency) + small noise.
    samples: (n_trials, n_channels, n_time)
    """
    n_time = samples.shape[2]
    freqs = rfftfreq(n_time, d=1.0 / sfreq)
    all_band_names = list(bands.keys())
    keep_bands = [b for b in all_band_names if b not in bands_to_mask]

    band_freq_masks = {
        name: (freqs >= fmin) & (freqs < fmax)
        for name, (fmin, fmax) in bands.items()
    }
    band_centers = {name: (fmin + fmax) / 2.0 for name, (fmin, fmax) in bands.items()}

    # Vectorised FFT: (n_trials, n_channels, n_freqs)
    specs = rfft(samples, axis=-1).copy()

    for band_name in bands_to_mask:
        fmask = band_freq_masks[band_name]
        fc = band_centers[band_name]

        if len(keep_bands) >= 2:
            kept_centers = np.array([band_centers[b] for b in keep_bands])
            kept_amps = np.array([
                np.abs(specs[:, :, band_freq_masks[b]]).mean()
                for b in keep_bands
            ])
            interp_amp = np.interp(
                np.log(fc + 1e-8),
                np.log(kept_centers + 1e-8),
                kept_amps
            )
        elif len(keep_bands) == 1:
            interp_amp = np.abs(specs[:, :, band_freq_masks[keep_bands[0]]]).mean()
        else:
            interp_amp = np.abs(specs).mean()

        local_amp = np.abs(specs[:, :, fmask])
        noise_std = local_amp.std() * 0.1
        orig_phase = np.angle(specs[:, :, fmask])
        new_amp = np.maximum(
            interp_amp + np.random.randn(*local_amp.shape) * noise_std, 0
        )
        specs[:, :, fmask] = new_amp * np.exp(1j * orig_phase)

    return irfft(specs, n=n_time, axis=-1).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Masking application
# ─────────────────────────────────────────────────────────────

def _apply_spatial_ar(samples, adv_samples, channels):
    """Replace channels with adversarial counterparts."""
    result = samples.copy()
    result[:, channels, :] = adv_samples[:, channels, :]
    return result


def _apply_frequency_ar(samples, adv_samples, band_names, bands, sfreq):
    """Replace frequency bands with adversarial counterparts."""
    n_time = samples.shape[2]
    freqs = rfftfreq(n_time, d=1.0 / sfreq)
    result = samples.copy()

    for ch in range(samples.shape[1]):
        spec = rfft(result[:, ch, :], axis=-1)
        adv_spec = rfft(adv_samples[:, ch, :], axis=-1)
        for band_name in band_names:
            fmin, fmax = bands[band_name]
            fmask = (freqs >= fmin) & (freqs < fmax)
            spec[:, fmask] = adv_spec[:, fmask]
        result[:, ch, :] = irfft(spec, n=n_time, axis=-1)

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
            total += len(by)
    return correct / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────
# Spearman consistency
# ─────────────────────────────────────────────────────────────

def _spearman_diff(morf, lerf):
    """
    At each masking step k, rank gradient methods by MoRF (ascending) and
    LeRF (descending), compute Spearman rho between the two rankings.
    morf / lerf: (n_methods, n_steps)
    """
    from scipy.stats import spearmanr
    n_steps = morf.shape[1]
    rhos = []
    for k in range(n_steps):
        morf_k = morf[:, k]
        lerf_k = lerf[:, k]
        rho, _ = spearmanr(morf_k, lerf_k)
        rhos.append(float(rho) if not np.isnan(rho) else 0.0)
    return rhos


def compute_spearman_consistency(results_path, dataset_name, task, model_name,
                                 best_iteration, gradient_methods,
                                 exp_key='spatial_ar'):
    """
    Load per-method faithfulness JSON files and compute Spearman consistency
    across gradient methods at each masking step.
    exp_key: which experiment type to use ('spatial_ar', 'spatial_road',
             'frequency_ar', 'frequency_road')
    """
    faith_dir = os.path.join(results_path, dataset_name, 'faithfulness')
    prefix = f'{dataset_name}_{task}_{model_name}_iteration_{best_iteration}'

    morfs, lerfs = [], []
    available = []
    for gm in gradient_methods:
        path = os.path.join(faith_dir, f'{prefix}_{gm}_{exp_key}_faithfulness.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)

        cluster_keys = [k for k in data if k != 'overall']
        if not cluster_keys:
            continue

        total_subj = sum(data[k]['n_subjects'] for k in cluster_keys)
        morf_avg = np.zeros(len(data[cluster_keys[0]]['morf_curve']))
        lerf_avg = np.zeros_like(morf_avg)
        for k in cluster_keys:
            w = data[k]['n_subjects'] / total_subj
            morf_avg += w * np.array(data[k]['morf_curve'])
            lerf_avg += w * np.array(data[k]['lerf_curve'])

        morfs.append(morf_avg)
        lerfs.append(lerf_avg)
        available.append(gm)

    if len(available) < 2:
        print(f"Spearman [{exp_key}]: fewer than 2 methods available, skipping.")
        return

    morf_mat = np.stack(morfs)
    lerf_mat = np.stack(lerfs)
    rhos = _spearman_diff(morf_mat, lerf_mat)
    mean_rho = float(np.mean(rhos))
    std_rho = float(np.std(rhos))
    print(f"Spearman [{exp_key}] ({len(available)} methods): "
          f"rho = {mean_rho:.3f} ± {std_rho:.3f}")

    out = {
        'exp_key': exp_key,
        'methods': available,
        'spearman_mean': mean_rho,
        'spearman_std': std_rho,
        'spearman_per_k': rhos,
    }
    path = os.path.join(faith_dir,
                        f'{prefix}_{exp_key}_spearman.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=4)
    print(f"Saved Spearman results to {path}")


# ─────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────

EXP_KEYS = ['spatial_ar', 'spatial_road', 'frequency_ar', 'frequency_road']


class FaithfulnessEvaluator:
    """
    Faithfulness evaluation for EEG subgroup clusters.

    Runs four independent masking experiments per cluster:
      • spatial_ar   — mask channels, replace with PGD adversarial
      • spatial_road — mask channels, replace with ROAD imputer
      • frequency_ar   — mask frequency bands, replace with PGD adversarial
      • frequency_road — mask frequency bands, replace with ROAD imputer

    Masking steps: k = 1 … n_channels (spatial) or k = 1 … n_bands (frequency),
    one feature at a time — matching XAI_tools_auto convention.

    Two baselines are computed per cluster per fold:
      • baseline_cluster: subject-level accuracy on cluster subjects (= 1.0 by design)
      • baseline_fold:    trial-level accuracy on the full fold test set
    """

    def __init__(self, bands, ch_names, sfreq=200, pgd_alpha=2.0, pgd_n_iter=10):
        self.bands = bands
        self.ch_names = ch_names
        self.sfreq = sfreq
        self.pgd_alpha = pgd_alpha
        self.pgd_n_iter = pgd_n_iter
        self.band_names = list(bands.keys())
        self.n_channels = len(ch_names)
        self.n_bands = len(bands)

    def _rank_channels(self, centroid):
        """centroid: (n_ch, n_bands) → channel indices sorted most-salient first"""
        return np.argsort(centroid.mean(axis=1))[::-1].tolist()

    def _rank_bands(self, centroid):
        """centroid: (n_ch, n_bands) → band indices sorted most-salient first"""
        return np.argsort(centroid.mean(axis=0))[::-1].tolist()

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

        # Encode groups to match inference_saliency encoding
        if dataset_name == 'ADvsFTDvsHC':
            enc = LabelEncoder()
            groups = enc.fit_transform(groups).astype(np.int64) + 1
        else:
            groups = groups.astype(np.int64)
        subject_ids_list = np.array(subject_ids_list, dtype=np.int64)
        cluster_labels = np.array(cluster_labels)

        sfm = ({int(k): int(v) for k, v in subject_fold_map.items()}
               if subject_fold_map is not None
               else {int(sid): 0 for sid in subject_ids_list})

        # Load one model per fold
        folds_needed = sorted({sfm[int(sid)] for sid in subject_ids_list
                                if int(sid) in sfm})
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

        # Reconstruct fold test sets for fold-level baseline
        kf = StratifiedGroupKFold(n_splits=5, shuffle=False)
        fold_test_indices = {
            fold: test_idx
            for fold, (_, test_idx) in enumerate(
                kf.split(samples, targets, groups=groups)
            )
        }

        # Cluster centroids
        psd_converter = PSDConverter(sfreq=self.sfreq)
        psd_features = psd_converter.convert(gradients_list, self.bands)
        unique_clusters = np.unique(cluster_labels)
        subj_to_cluster = dict(zip(subject_ids_list, cluster_labels))
        centroids = {
            c: psd_features[cluster_labels == c].mean(axis=0)
            for c in unique_clusters
        }

        trial_mask = np.isin(groups, subject_ids_list)
        samples_c = samples[trial_mask]
        targets_c = targets[trial_mask]
        groups_c = groups[trial_mask]

        n_classes = len(np.unique(targets))
        all_results = {k: {} for k in EXP_KEYS}

        for c in unique_clusters:
            cluster_subjects = [sid for sid in subject_ids_list
                                 if subj_to_cluster[sid] == c]
            ranked_ch = self._rank_channels(centroids[c])
            ranked_bd = self._rank_bands(centroids[c])

            fold_to_subjects = defaultdict(list)
            for sid in cluster_subjects:
                fold_to_subjects[sfm.get(int(sid), 0)].append(int(sid))

            print(f"Cluster {c}: {len(cluster_subjects)} subjects "
                  f"across {len(fold_to_subjects)} fold(s)")

            fold_data = {k: dict(morf=[], lerf=[], weights=[],
                                 bc=[], bf=[])
                         for k in EXP_KEYS}

            for fold in sorted(fold_to_subjects):
                net_f = fold_nets[fold]
                fold_subs = fold_to_subjects[fold]

                f_mask = np.isin(groups_c, fold_subs)
                X_f = samples_c[f_mask]
                y_f = targets_c[f_mask]
                g_f = groups_c[f_mask]

                # Two baselines
                bc = _subject_accuracy(net_f, X_f, y_f, g_f, fold_subs, device)
                bf = _trial_accuracy(
                    net_f,
                    samples[fold_test_indices[fold]],
                    targets[fold_test_indices[fold]],
                    device
                )

                # PGD adversarial examples (shared by AR variants)
                epsilon = float(np.abs(X_f).max())
                adv_f = self._compute_pgd(net_f, X_f, y_f, epsilon, device)

                # ── Spatial AR ───────────────────────────────────
                s_ar_m, s_ar_l = [], []
                for k in range(1, self.n_channels + 1):
                    s_ar_m.append(_subject_accuracy(
                        net_f, _apply_spatial_ar(X_f, adv_f, ranked_ch[:k]),
                        y_f, g_f, fold_subs, device))
                    s_ar_l.append(_subject_accuracy(
                        net_f, _apply_spatial_ar(X_f, adv_f, ranked_ch[-k:]),
                        y_f, g_f, fold_subs, device))

                # ── Spatial ROAD ─────────────────────────────────
                s_rd_m, s_rd_l = [], []
                for k in range(1, self.n_channels + 1):
                    s_rd_m.append(_subject_accuracy(
                        net_f, _road_spatial_replace(X_f, ranked_ch[:k]),
                        y_f, g_f, fold_subs, device))
                    s_rd_l.append(_subject_accuracy(
                        net_f, _road_spatial_replace(X_f, ranked_ch[-k:]),
                        y_f, g_f, fold_subs, device))

                # ── Frequency AR ──────────────────────────────────
                f_ar_m, f_ar_l = [], []
                for k in range(1, self.n_bands + 1):
                    bm = [self.band_names[i] for i in ranked_bd[:k]]
                    bl = [self.band_names[i] for i in ranked_bd[-k:]]
                    f_ar_m.append(_subject_accuracy(
                        net_f, _apply_frequency_ar(X_f, adv_f, bm, self.bands, self.sfreq),
                        y_f, g_f, fold_subs, device))
                    f_ar_l.append(_subject_accuracy(
                        net_f, _apply_frequency_ar(X_f, adv_f, bl, self.bands, self.sfreq),
                        y_f, g_f, fold_subs, device))

                # ── Frequency ROAD ────────────────────────────────
                f_rd_m, f_rd_l = [], []
                for k in range(1, self.n_bands + 1):
                    bm = [self.band_names[i] for i in ranked_bd[:k]]
                    bl = [self.band_names[i] for i in ranked_bd[-k:]]
                    f_rd_m.append(_subject_accuracy(
                        net_f, _road_frequency_replace(X_f, bm, self.bands, self.sfreq),
                        y_f, g_f, fold_subs, device))
                    f_rd_l.append(_subject_accuracy(
                        net_f, _road_frequency_replace(X_f, bl, self.bands, self.sfreq),
                        y_f, g_f, fold_subs, device))

                w = len(fold_subs)
                for key, mf, lf in [
                    ('spatial_ar',    s_ar_m, s_ar_l),
                    ('spatial_road',  s_rd_m, s_rd_l),
                    ('frequency_ar',  f_ar_m, f_ar_l),
                    ('frequency_road',f_rd_m, f_rd_l),
                ]:
                    fd = fold_data[key]
                    fd['morf'].append(np.array(mf))
                    fd['lerf'].append(np.array(lf))
                    fd['weights'].append(w)
                    fd['bc'].append(bc)
                    fd['bf'].append(bf)

            # Weighted average across folds
            for key in EXP_KEYS:
                fd = fold_data[key]
                total_w = sum(fd['weights'])
                ws = fd['weights']
                morf_curve = sum(m * (w / total_w)
                                 for m, w in zip(fd['morf'], ws)).tolist()
                lerf_curve = sum(l * (w / total_w)
                                 for l, w in zip(fd['lerf'], ws)).tolist()
                baseline_cluster = float(sum(b * w / total_w
                                             for b, w in zip(fd['bc'], ws)))
                baseline_fold = float(sum(b * w / total_w
                                          for b, w in zip(fd['bf'], ws)))

                aoc_c, auc_c, abc_c = self._compute_metrics(
                    morf_curve, lerf_curve, baseline_cluster, n_classes)
                aoc_f, auc_f, abc_f = self._compute_metrics(
                    morf_curve, lerf_curve, baseline_fold, n_classes)

                all_results[key][int(c)] = {
                    'AOC_cluster_baseline': aoc_c,
                    'AUC_cluster_baseline': auc_c,
                    'ABC_cluster_baseline': abc_c,
                    'AOC_fold_baseline':    aoc_f,
                    'AUC_fold_baseline':    auc_f,
                    'ABC_fold_baseline':    abc_f,
                    'baseline_cluster': baseline_cluster,
                    'baseline_fold':    baseline_fold,
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

        # Overall weighted average per experiment type
        for key in EXP_KEYS:
            total = sum(all_results[key][int(c)]['n_subjects']
                        for c in unique_clusters)
            all_results[key]['overall'] = {
                metric: float(sum(
                    all_results[key][int(c)][metric]
                    * all_results[key][int(c)]['n_subjects'] / total
                    for c in unique_clusters
                ))
                for metric in (
                    'AOC_cluster_baseline', 'AUC_cluster_baseline', 'ABC_cluster_baseline',
                    'AOC_fold_baseline',    'AUC_fold_baseline',    'ABC_fold_baseline',
                )
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
            with open(path, 'w') as f:
                json.dump(all_results[key], f, indent=4)
        print(f"Saved faithfulness results to {save_dir}/")

        return all_results
