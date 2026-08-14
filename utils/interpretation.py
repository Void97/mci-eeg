import os
import time
import mne
import numpy as np
import scipy.signal
import torch
from captum.attr import Saliency
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler


def _class_labels(dataset_name, task):
    """Human-readable class names for a dataset/task, shared by every plot
    in this module (PSD, topomap, t-SNE) instead of each re-deriving it."""
    if dataset_name in ('GENEEG', 'MCIvsHC'):
        return ('Healthy Controls', 'MCI')
    if dataset_name == 'ADvsFTDvsHC':
        return {
            'AD vs HC': ('Healthy Controls', 'AD'),
            'FTD vs HC': ('Healthy Controls', 'FTD'),
            'FTD vs AD': ('FTD', 'AD'),
        }[task]
    if dataset_name == 'CAUEEG':
        return {
            'Dementia vs Normal': ('Normal', 'Dementia'),
            'MCI vs Normal': ('Normal', 'MCI'),
            'MCI vs Dementia': ('MCI', 'Dementia'),
        }[task]
    raise ValueError(f"No class labels defined for dataset {dataset_name!r}")


def _logits_from_output(out):
    """Every model except MSVTNet returns a plain logits tensor; MSVTNet
    (b_preds=True by default) returns (main_logits, branch_logits). Kept as
    a private copy of models_train.training_loop.logits_from_output to avoid
    a cross-package import cycle (models_train already imports this module)."""
    return out[0] if isinstance(out, tuple) else out


def _captum_forward_fn(model):
    def forward(x):
        return _logits_from_output(model(x))
    return forward


def compute_saliency(model, device, test_loader, test_preds_fold, test_labels_fold):
    """Gradient-based saliency maps for correctly-classified test samples,
    grouped by class. Needs its own pass over test_loader with
    `xb.requires_grad = True`, separate from any no_grad inference pass."""
    saliency_inst = Saliency(_captum_forward_fn(model))
    gradient_batches = []
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        xb.requires_grad = True
        target = yb.detach().cpu().long().tolist()
        gradient_batches.append(
            saliency_inst.attribute(xb, target=target, abs=False).detach().cpu().numpy()
        )

    gradient_list = np.concatenate(gradient_batches)
    if gradient_list.shape[1] == 1:
        gradient_list = np.squeeze(gradient_list, axis=1)

    saliency_maps = {class_id: [] for class_id in np.unique(test_labels_fold)}
    for class_id in np.unique(test_labels_fold):
        correct_indices = np.where(
            (np.array(test_labels_fold) == class_id) & (np.array(test_preds_fold) == class_id)
        )[0]
        if correct_indices.size > 0:
            saliency_maps[class_id].append(gradient_list[correct_indices])
    for class_id in saliency_maps:
        saliency_maps[class_id] = (
            np.concatenate(saliency_maps[class_id]) if saliency_maps[class_id] else np.array([])
        )
    return saliency_maps


def save_tsne_plot(model_spec, best_model_state, test_loader, tsne_savedir,
                    dataset_name, task, iter_, fold, device, n_iter, random_state):
    class1, class2 = _class_labels(dataset_name, task)

    model = model_spec.cls(**model_spec.kwargs, tsne=True)
    model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()

    features, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            feats = model(xb)
            features.extend(feats.cpu())
            labels.extend(np.array(yb.cpu()))
    features = np.array(features)
    labels = np.array(labels).reshape(-1, 1)

    tsne = TSNE(n_components=2, n_iter=n_iter, random_state=random_state)
    features_2d = tsne.fit_transform(features)

    scaler = MinMaxScaler(feature_range=(0, 1))
    features_norm = scaler.fit_transform(features_2d)

    os.makedirs(tsne_savedir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    colors = ['blue' if label == 0 else 'red' for label in labels]
    plt.scatter(
        [point[0] for point in features_norm],
        [point[1] for point in features_norm],
        c=colors,
        alpha=0.5,
    )
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label=class1),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label=class2),
    ]
    plt.legend(handles=legend_elements, fontsize=14, loc='upper right')
    plt.savefig(os.path.join(tsne_savedir, f'{task}_{model_spec.name}_iteration_{iter_}_fold_{fold}_tsne.png'))
    plt.close()
    print(f"t-SNE plot saved for fold {fold + 1}")


class Interpretation:
    def __init__(self, savedir, sfreq):
        self.savedir = savedir
        self.sfreq = sfreq
        os.makedirs(self.savedir, exist_ok=True)

    # ---------------------------
    # PSD Plot
    # ---------------------------
    def plot_psd(self, dataset_name, model_name, task, gradient, iter, limit=40):

        classes = _class_labels(dataset_name, task)
        limit_freq = min(int(self.sfreq // 2), limit)

        fig, ax = plt.subplots(figsize=(18, 5))
        global_y_min, global_y_max = float('inf'), float('-inf')

        for idx, class_name in enumerate(classes):
            if len(gradient[idx]) == 0:
                continue

            freqs, psds = [], []
            for subject_gradient in gradient[idx]:
                f, p = scipy.signal.welch(
                    subject_gradient,
                    self.sfreq,
                    nperseg=int(self.sfreq),
                    noverlap=int(self.sfreq // 2)
                )
                freqs.append(f)
                # Average across channels but keep frequency resolution
                if p.ndim > 1:
                    psds.append(np.mean(np.abs(p), axis=0))
                else:
                    psds.append(np.abs(p))

            # Normalize PSDs subject-wise
            psd_norm = []
            for psd in psds:
                tmp_data = psd[:limit_freq]
                if tmp_data.max() - tmp_data.min() > 0:
                    psd_norm.append((tmp_data - tmp_data.min()) /
                                    (tmp_data.max() - tmp_data.min()))

            if not psd_norm:
                continue

            psd_norm = np.mean(np.array(psd_norm), axis=0)
            f = freqs[0][:limit_freq]
            ax.semilogy(f, psd_norm, linewidth=3, label=class_name)

            y_min, y_max = ax.get_ylim()
            global_y_min = min(global_y_min, y_min)
            global_y_max = max(global_y_max, y_max)

        ax.set_ylim([global_y_min, global_y_max])
        ax.set_xlim([0, limit_freq])
        ax.set_xticks(np.arange(0, limit_freq + 1, 5))
        ax.set_xlabel('Frequency (Hz)', fontsize=20)
        ax.set_ylabel('Normalized PSD (a.u.)', fontsize=20)
        ax.set_title(f'PSD Comparison ({task}) ({model_name})', fontsize=24)

        # EEG frequency band markers
        for x_tick in [4, 8, 12, 30]:
            ax.axvline(x=x_tick, color='gray', linestyle='--', linewidth=1.7)

        # Frequency band labels
        band_labels = [('δ', 2), ('θ', 6), ('α', 10), ('β', 20), ('γ', 35)]
        for label, xpos in band_labels:
            ax.text(xpos, global_y_min * 1.2, label, ha='center',
                    va='center', fontsize=22)

        ax.tick_params(axis='both', labelsize=18)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.legend(fontsize=16, loc='upper right')
        plt.tight_layout()

        out_path = os.path.join(self.savedir, f'{task}_{model_name}_iteration_{iter}_psd.png')
        plt.savefig(out_path)
        plt.close()
        print(f"PSD plot saved to {out_path}")

    # ---------------------------
    # PSD Topomap Plot
    # ---------------------------
    def plot_psd_topo(self, dataset_name, model_name, task, gradient, freq_bands, used_channels, show_channel, iter):
        montage = mne.channels.make_standard_montage('standard_1020')
        ch_pos_dict = montage.get_positions()['ch_pos']

    # Keep only channels present in montage
        valid_channels = [ch for ch in used_channels if ch in ch_pos_dict]
        if len(valid_channels) == 0:
            print("No valid channels found in montage; skipping topomap.")
            return

        xai_ch_pos = np.array([ch_pos_dict[ch] for ch in valid_channels])

    # Filter show_channel consistently and keep the filtered names
        filtered_show_channel = [ch for ch in show_channel if ch in ch_pos_dict]
        show_ch_pos = np.array([ch_pos_dict[ch] for ch in filtered_show_channel])

    # Slight visual lift
        if xai_ch_pos.size:
            xai_ch_pos[:, 1] += 0.01
        if show_ch_pos.size:
            show_ch_pos[:, 1] += 0.01

        classes = _class_labels(dataset_name, task)
        freq_range = {
            'delta': [0, 4], 'theta': [4, 8], 'alpha': [8, 12],
            'beta': [12, 30], 'gamma': [30, 45], 'all': [0, 45]
        }

    # --- Compute PSD per channel per class ---
        classes_psd = {}  # class_name -> (n_valid_channels, n_freqs)
        for idx, class_name in enumerate(classes):
            if idx >= len(gradient) or len(gradient[idx]) == 0:
                continue

            per_subject_psds = []
            for subject_gradient in gradient[idx]:
            # subject_gradient shape: (n_channels, n_time) expected
                f, p = scipy.signal.welch(
                    subject_gradient, self.sfreq,
                    nperseg=int(self.sfreq), noverlap=int(self.sfreq // 2)
                )  # p: (n_channels, n_freqs) if 2D, else (n_freqs,)

                if p.ndim == 2:
                # Map valid channel indices
                    ch_idx = [used_channels.index(ch) for ch in valid_channels]
                    p = np.abs(p[ch_idx, :])  # (n_valid_channels, n_freqs)
                else:
                # If 1D, broadcast to channels (assume same spectrum for all — unlikely, but safe)
                    p = np.abs(p)[None, :] * np.ones((len(valid_channels), 1))

            # Subject-wise normalization with epsilon to avoid /0
                p_min = np.nanmin(p)
                p_max = np.nanmax(p)
                if not np.isfinite(p_min) or not np.isfinite(p_max):
                    continue
                denom = (p_max - p_min)
                if denom <= 0:
                # flat PSD; skip or keep zeros – here we skip to avoid NaNs
                    continue
                p_norm = (p - p_min) / denom
                per_subject_psds.append(p_norm)

            if len(per_subject_psds) == 0:
                continue

            classes_psd[class_name] = np.nanmean(np.stack(per_subject_psds, axis=0), axis=0)

        if len(classes_psd) == 0:
            print("No PSD data available after normalization; skipping topomap.")
            return

    # --- Plot per band ---
        for freq_bd in freq_bands:
            if freq_bd not in freq_range:
                print(f"Unknown band '{freq_bd}', skipping.")
                continue
            band_min, band_max = freq_range[freq_bd]

        # Determine global color limits across classes for this band
            vmins = []
            vmaxs = []
            band_data_by_class = {}

            for class_name in classes:
                if class_name not in classes_psd:
                    continue
                # NOTE: Welch with nperseg=self.sfreq gives ~1 Hz resolution; slicing by Hz is OK.
                subband = np.nanmean(classes_psd[class_name][:, band_min:band_max], axis=1)  # (n_valid_channels,)
                band_data_by_class[class_name] = subband
            # collect finite min/max for vlim
                if np.any(np.isfinite(subband)):
                    vmins.append(np.nanmin(subband))
                    vmaxs.append(np.nanmax(subband))

            if len(band_data_by_class) == 0:
                print(f"No data for band '{freq_bd}', skipping.")
                continue

        # Robust vlim
            if len(vmins) == 0 or len(vmaxs) == 0:
                print(f"No finite values for band '{freq_bd}', skipping.")
                continue
            vmin = float(np.nanmin(vmins))
            vmax = float(np.nanmax(vmaxs))
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                print(f"Non-finite vlim for band '{freq_bd}', skipping.")
                continue
            if vmax <= vmin:
            # widen tiny/flat ranges
                eps = 1e-6 if vmax == vmin else 0.0
                vmin = vmin - eps
                vmax = vmax + eps

        # Create figure
            fig, axs = plt.subplots(1, len(classes), figsize=(12, 6), constrained_layout=True)
            axs = [axs] if not isinstance(axs, np.ndarray) else axs.flatten()
            ims = []

            for idx, class_name in enumerate(classes):
                ax = axs[idx]
                ax.set_axis_off()
                ax.set_title(f"{class_name} - {freq_bd.capitalize()} Band", fontsize=16)

                if class_name not in band_data_by_class:
                # draw blank axes to preserve layout
                    ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                    continue

                subband_psd = band_data_by_class[class_name]  # (n_valid_channels,)
                im, _ = mne.viz.plot_topomap(
                    subband_psd,
                    pos=xai_ch_pos[:, 0:2],
                    axes=ax,
                    show=False,
                    cmap='Reds',
                    vlim=(vmin, vmax),
                    sensors=False,
                    outlines='head',
                    sphere=(0.0, 0.0, 0.0, 0.12),
                    extrapolate='box'
                )
                ims.append(im)

            # Draw labels at filtered positions
                for (xy, name) in zip(show_ch_pos, filtered_show_channel):
                    ax.text(xy[0], xy[1], name, ha='center', va='center', fontsize=12, color='black')

        # Colorbar (only if we actually plotted at least one image)
            if len(ims) > 0:
                #cbar = plt.colorbar(ims[0], ax=axs, orientation='vertical', shrink=0.7)
                cbar = fig.colorbar(ims[0], ax=axs, shrink=0.8, pad=0.02, aspect=30)
                # Let Matplotlib choose ticks; avoid forcing mismatched tick labels
                cbar.set_label('Normalized PSD (a.u.)', fontsize=12)

            out_path = os.path.join(self.savedir, f"{task}_{model_name}_iteration_{iter}_topo_{freq_bd}.png")
            #plt.tight_layout()
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"Topomap for {freq_bd} saved to {out_path}")
