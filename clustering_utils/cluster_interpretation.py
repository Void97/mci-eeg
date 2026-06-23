import os
import json
import mne
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import welch


def define_class_name(dataset_name, task, func):
    if dataset_name == 'GENEEG' or dataset_name == 'MCIvsHC':
        class_name = 'MCI'
        negative_class_name = 'HC'
    elif dataset_name == 'CAUEEG':
        if task == 'MCI vs Normal' or task == 'MCI vs Dementia':
            class_name = 'MCI'
            negative_class_name = 'Normal' if task == 'MCI vs Normal' else 'Dementia'
        elif task == 'Dementia vs Normal':
            class_name = 'Dementia'
            negative_class_name = 'Normal'
    elif dataset_name == 'ADvsFTDvsHC':
        if task == 'AD vs HC' or task == 'FTD vs AD':
            class_name = 'AD'
            negative_class_name = 'HC' if task == 'AD vs HC' else 'FTD'
        elif task == 'FTD vs HC':
            class_name = 'FTD'
            negative_class_name = 'HC'

    if func == 'topo':
        return class_name
    elif func == 'curve':
        return class_name, negative_class_name

def bands_and_range():
    freq_bands = ['all', 'δ', 'θ', 'α', 'β', 'γ']
    freq_range = {
            'δ': [0, 4], 
            'θ': [4, 8], 
            'α': [8, 12],
            'β': [12, 30], 
            'γ': [30, 45], 
            'all': [0, 45]
        }
    return freq_bands, freq_range

def compute_subject_mean_psd(subject_map, sfreq):

    subject_map = np.abs(subject_map)

    freqs, psd = welch(
        subject_map,
        fs=sfreq,
        axis=1,
        nperseg=sfreq,
        noverlap=sfreq // 2
    )

    #psd = psd / (psd.sum() + 1e-8)
    psd = psd[:45]
    psd = (psd - psd.min()) / (psd.max() - psd.min() + 1e-8)
    mean_psd_curve = psd.mean(axis=0)
    return freqs, mean_psd_curve

def compute_cluster_mean_psd(group_maps, sfreq):

    subject_curves = []
    for subject_map in group_maps:
        freqs, curves = compute_subject_mean_psd(subject_map, sfreq)
        subject_curves.append(curves)
    
    subject_curves = np.stack(subject_curves, axis=0)
    mean_curve = subject_curves.mean(axis=0)
    
    return freqs, mean_curve 

def plot_cluster_subject_psd_curves(
        dataset_name,
        task,
        model_name,
        best_iteration,
        clustering_method,
        saliency_maps,
        saliency_maps_negative,
        cluster_labels,
        used_channels,
        subject_ids,
        negative_subject_ids,
        func,
        sfreq):

    saliency_maps = np.asarray(saliency_maps)
    saliency_maps_negative = np.asarray(saliency_maps_negative)
    cluster_labels = np.asarray(cluster_labels)
    
    class_name, negative_class_name = define_class_name(dataset_name, task, func)

    if saliency_maps.ndim != 3:
        raise ValueError(
            f"saliency_maps must have shape (n_subjects, n_channels, n_timepoints), "
            f"but got shape {saliency_maps.shape}"
        )
    if saliency_maps_negative.ndim != 3:
        raise ValueError(
            f"saliency_maps_negative must have shape (n_subjects, n_channels, n_timepoints), "
            f"but got shape {saliency_maps_negative.shape}"
        )

    n_subjects = saliency_maps.shape[0]

    if len(cluster_labels) != n_subjects:
        raise ValueError(
            f"Mismatch: saliency_maps has {n_subjects} subjects, "
            f"but cluster_labels has {len(cluster_labels)} labels."
        )

    if subject_ids is None:
        subject_ids = [f'Subject_{i+1}' for i in range(n_subjects)]
    else:
        subject_ids = list(subject_ids)

    if len(subject_ids) != n_subjects:
        raise ValueError(
            f"Mismatch: subject_ids has {len(subject_ids)} entries, "
            f"but saliency_maps has {n_subjects} subjects."
        )
    
    if negative_subject_ids is None:
        negative_subject_ids = [f'Subject_{i+1}' for i in range(saliency_maps_negative.shape[0])]
    else:
        negative_subject_ids = list(negative_subject_ids)
    if len(negative_subject_ids) != saliency_maps_negative.shape[0]:
        raise ValueError(
            f"Mismatch: negative_subject_ids has {len(negative_subject_ids)} entries, "
            f"but saliency_maps_negative has {saliency_maps_negative.shape[0]} subjects."
        )

    save_dir = f'results/clustering/{dataset_name}/cluster_psd_curves'
    os.makedirs(save_dir, exist_ok=True)

    montage = mne.channels.make_standard_montage('standard_1020')
    ch_pos_dict = montage.get_positions()['ch_pos']

    valid_channels = [ch for ch in used_channels if ch in ch_pos_dict]
    if len(valid_channels) == 0:
        print("No valid channels found in montage. Please check channel names.")
        return

    valid_idx = [used_channels.index(ch) for ch in valid_channels]
    saliency_maps = saliency_maps[:, valid_idx, :]
    saliency_maps_negative = saliency_maps_negative[:, valid_idx, :]

    freqs_negative, mean_curve_negative = compute_cluster_mean_psd(saliency_maps_negative, sfreq)
    unique_clusters = np.unique(cluster_labels)
    for cluster_label in unique_clusters:
        cluster_mask = cluster_labels == cluster_label
        cluster_subject_maps = saliency_maps[cluster_mask]
        
        if cluster_subject_maps.shape[0] == 0:
            print(f"No subjects in cluster {cluster_label}, skipping.")
            continue

        freqs_cluster, cluster_mean_curve = compute_cluster_mean_psd(cluster_subject_maps, sfreq)

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(
            freqs_cluster,
            cluster_mean_curve,
            linewidth=2.5,
            label=f'Cluster {cluster_label+1 if cluster_label != -1 else "Outliers"} - {class_name}',
            color='red'
        )
        ax.plot(
            freqs_negative,
            mean_curve_negative,
            linewidth=2.5,
            label=f'{negative_class_name}',
            color='blue'
        )
        ax.set_title(
            f'{class_name} Cluster {cluster_label + 1} vs {negative_class_name}'
            if cluster_label != -1 else
            f'{class_name} Outliers vs {negative_class_name}',
            fontsize=16
        )
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Normalized PSD')
        ax.set_xlim(0, 45)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)

        band_edges = [4, 8, 12, 30]
        band_centers = [2, 6, 10, 21, 37.5]
        band_names = ['δ', 'θ', 'α', 'β', 'γ']

        # draw vertical dashed lines
        for edge in band_edges:
            ax.axvline(edge, linestyle='--', color='gray', alpha=0.5, linewidth=1)

        # enforce correct x-axis range
        ax.set_xlim(0, 45)

        # add band names on top
        ymax = ax.get_ylim()[1]
        for x, name in zip(band_centers, band_names):
            ax.text(
                x,
                ymax * 0.95,
                name,
                ha='center',
                va='top',
                fontsize=11,
                alpha=0.8
            )    

        plt.tight_layout()

        savename = (
            f'{dataset_name}_{task}_{model_name}_iter_{best_iteration}_'
            f'{clustering_method}_cluster_{cluster_label}_psd_curves.png'
        )
        savepath = os.path.join(save_dir, savename)
        plt.savefig(savepath, bbox_inches='tight')


    # unique_clusters = np.unique(cluster_labels)

    # for cluster_label in unique_clusters:
    #     cluster_mask = cluster_labels == cluster_label
    #     cluster_subject_maps = saliency_maps[cluster_mask]
    #     cluster_subject_ids = [subject_ids[i] for i in range(n_subjects) if cluster_mask[i]]

    #     if cluster_subject_maps.shape[0] == 0:
    #         continue

    #     subject_mean_psd_curves = []

    #     for subject_map in cluster_subject_maps:
    #         subject_map = np.abs(subject_map)

    #         freqs, psd = welch(
    #             subject_map,
    #             fs=sfreq,
    #             axis=1,
    #             nperseg=sfreq,
    #             noverlap=sfreq // 2
    #         )

    #         psd = psd / (psd.sum() + 1e-8)
    #         mean_psd_curve = psd.mean(axis=0)
    #         subject_mean_psd_curves.append(mean_psd_curve)

    #     subject_mean_psd_curves = np.stack(subject_mean_psd_curves, axis=0)

    #     cmap = plt.cm.get_cmap('tab20', len(cluster_subject_ids))

    #     fig, ax = plt.subplots(figsize=(12, 7))

    #     for i, curve in enumerate(subject_mean_psd_curves):
    #         ax.plot(
    #             freqs,
    #             curve,
    #             color=cmap(i),
    #             linewidth=1.5,
    #             alpha=0.9,
    #             label=str(cluster_subject_ids[i])
    #         )

    #     ax.set_title(
    #         f'Class: {class_name}, Cluster: {cluster_label+1 if cluster_label != -1 else "Outliers"}\n'
    #         f"Subjects' PSD curves",
    #         fontsize=16
    #     )
    #     ax.set_xlabel('Frequency (Hz)')
    #     ax.set_ylabel('Normalized PSD')
    #     ax.grid(True, alpha=0.3)

    #     ax.legend(
    #         title='Subject ID',
    #         loc='center left',
    #         bbox_to_anchor=(1.02, 0.5),
    #         fontsize=9,
    #         frameon=True
    #     )

    #     plt.tight_layout()

    #     savename = (
    #         f'{dataset_name}_{task}_{model_name}_iter_{best_iteration}_'
    #         f'{clustering_method}_cluster_{cluster_label}_subject_psd_curves.png'
    #     )
    #     savepath = os.path.join(save_dir, savename)
    #     plt.savefig(savepath, bbox_inches='tight')
    #     plt.close(fig)

    #     print(f"Saved PSD curves for cluster {cluster_label} at {savepath}")

def plot_cluster_psd_topomaps(
        dataset_name, 
        task, 
        model_name, 
        best_iteration, 
        clustering_method, 
        saliency_maps, 
        cluster_labels,
        used_channels, 
        show_channels,
        subject_ids,
        func,
        sfreq):

    if func == 'topo':
        class_name = define_class_name(dataset_name, task, func)
    freq_bands, freq_range = bands_and_range()
    
    saliency_maps = np.asarray(saliency_maps)
    cluster_labels = np.asarray(cluster_labels)

    save_dir = f'results/clustering/{dataset_name}/cluster_psd_topomaps'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_pos_dict = montage.get_positions()['ch_pos']

    # Keep only channels present in montage
    valid_channels = [ch for ch in used_channels if ch in ch_pos_dict]
    if len(valid_channels) == 0:
        print("No valid channels found in montage. Please check channel names.")
        return

    filtered_show_channels = [ch for ch in show_channels if ch in valid_channels]
    valid_idx = [used_channels.index(ch) for ch in valid_channels]

    # MNE info for valid channels
    info = mne.create_info(ch_names=valid_channels, sfreq=sfreq, ch_types='eeg')
    info.set_montage(montage)

    # Check alignment
    if saliency_maps.shape[0] != len(cluster_labels):
        raise ValueError(
            f"Mismatch: saliency_maps has {saliency_maps.shape[0]} subjects, "
            f"but cluster_labels has {len(cluster_labels)} labels."
        )
    
    if subject_ids is None:
        subject_ids = [f'Subject_{i+1}' for i in range(saliency_maps.shape[0])]
    else:
        subject_ids = list(subject_ids)
    
    if len(subject_ids) != saliency_maps.shape[0]:
        raise ValueError(
            f"Mismatch: subject_ids has {len(subject_ids)} entries, "
            f"but saliency_maps has {saliency_maps.shape[0]} subjects."
        )


    for cluster_label in np.unique(cluster_labels):

        cluster_mask = cluster_labels == cluster_label
        cluster_subject_maps = saliency_maps[cluster_mask]        
        print(f"Processing cluster {cluster_label} with {cluster_subject_maps.shape[0]} subjects")

        if cluster_subject_maps.shape[0] == 0:
            print(f"No subjects in cluster {cluster_label}, skipping.")
            continue

        # Keep only valid channels
        # shape -> (n_subjects_in_cluster, n_valid_channels, timepoints)
        cluster_subject_maps = cluster_subject_maps[:, valid_idx, :]

        subject_psd_list = []

        for subject_map in cluster_subject_maps:
            # subject_map shape: (channels, timepoints)
            subject_map = np.abs(subject_map)

            freqs, psd = welch(
                subject_map,
                fs=sfreq,
                axis=1,
                nperseg=sfreq,
                noverlap=sfreq // 2,
            )
            # psd shape: (channels, freqs)

            # Subject-wise normalization
            #psd = psd / (psd.sum() + 1e-8)
            psd = (psd - psd.min()) / (psd.max() - psd.min() + 1e-8)

            subject_psd_list.append(psd)

        subject_psd_list = np.stack(subject_psd_list, axis=0)
        # shape -> (n_subjects_in_cluster, channels, freqs)

        # Average normalized PSDs across subjects in the cluster
        mean_cluster_psd = subject_psd_list.mean(axis=0)
        # shape -> (channels, freqs)

        band_psd_list = []
        valid_psd_names = []
        for band in freq_bands:
            fmin, fmax = freq_range[band]

            if band == 'all':
                band_mask = (freqs >= fmin) & (freqs <= fmax)
            else:
                band_mask = (freqs >= fmin) & (freqs < fmax)
            
            if not np.any(band_mask):
                print(f"No frequencies found in the {band} band for cluster {cluster_label}. Skipping.")
                continue

            band_psd = mean_cluster_psd[:, band_mask].mean(axis=1)
            band_psd_list.append(band_psd)
            valid_psd_names.append(band)
            # shape -> (channels,)

            # fig, ax = plt.subplots(figsize=(6, 5))
            # im, _ = mne.viz.plot_topomap(
            #     band_psd,
            #     info,
            #     axes=ax,
            #     show=False,
            #     sensors=True,
            #     names=valid_channels if len(filtered_show_channels) > 0 else None,
            # )

            # if len(filtered_show_channels) > 0 and ax.texts:
            #     for txt in ax.texts:
            #         if txt.get_text() not in filtered_show_channels:
            #             txt.set_visible(False)
            
            # plt.colorbar(im, ax=ax)
            # ax.set_title(
            #     f'{model_name}, iter {best_iteration}, {clustering_method}\n'
            #     f'{dataset_name}, {task}, Class: {class_name}\n' 
            #     f'Cluster {cluster_label}, {band} band'
            # )

        # create the horizontal subplot for the current band
        n_bands = len(band_psd_list)
        fig, axes = plt.subplots(1, n_bands, figsize=(6*n_bands, 5))
        if n_bands == 1:
            axes = [axes]
        for ax, band_psd, band_name in zip(axes, band_psd_list, valid_psd_names):
            im, _ = mne.viz.plot_topomap(
                band_psd,
                info,
                axes=ax,
                show=False,
                sensors=True,
                names=valid_channels if len(filtered_show_channels) > 0 else None,
            )

            if len(filtered_show_channels) > 0 and ax.texts:
                for txt in ax.texts:
                    if txt.get_text() not in filtered_show_channels:
                        txt.set_visible(False)
            
            plt.colorbar(im, ax=ax)
            ax.set_title(f'{band_name} band', fontsize=25)

        fig.suptitle(
            f'Class: {class_name}, Cluster: {cluster_label+1 if cluster_label != -1 else "Outliers"}\n'
            f'{dataset_name}, {task}, Class: {class_name}',
            fontsize=25
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        savename = (
            f'{dataset_name}_{task}_{model_name}_iter_{best_iteration}_'
            f'{clustering_method}_cluster_{cluster_label}_psd_topomap.png'
        )
        savepath = os.path.join(save_dir, savename)
        plt.savefig(savepath, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved topomap for cluster {cluster_label} at {savepath}")


def plot_subject_psd_topomaps(
        dataset_name, 
        task, 
        model_name, 
        best_iteration, 
        clustering_method, 
        saliency_maps, 
        cluster_labels,
        used_channels, 
        show_channels,
        subject_ids,
        func,
        sfreq):

    # This function can be implemented similarly to plot_cluster_psd_topomaps,
    # but instead of averaging PSDs across subjects in a cluster, it would generate
    # topomaps for each individual subject's PSD. The title and save path would also
    # include the subject ID for clarity.

    class_name = define_class_name(dataset_name, task, func)
    freq_bands, freq_range = bands_and_range()

    saliency_maps = np.array(saliency_maps)
    cluster_labels = np.array(cluster_labels)

    save_dir = f'results/clustering/{dataset_name}/subject_psd_topomaps'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    montage = mne.channels.make_standard_montage('standard_1020')
    ch_pos_dict = montage.get_positions()['ch_pos']
    # Keep only channels present in montage
    valid_channels = [ch for ch in used_channels if ch in ch_pos_dict]
    if len(valid_channels) == 0:
        print("No valid channels found in montage. Please check channel names.")
        return
    
    filtered_show_channels = [ch for ch in show_channels if ch in valid_channels]
    valid_idx = [used_channels.index(ch) for ch in valid_channels]

    # MNE info for valid channels
    info = mne.create_info(ch_names=valid_channels, sfreq=sfreq, ch_types='eeg')
    info.set_montage(montage)

    if saliency_maps.shape[0] != len(cluster_labels):
        raise ValueError(
            f"Mismatch: saliency_maps has {saliency_maps.shape[0]} subjects, "
            f"but cluster_labels has {len(cluster_labels)} labels."
        )

    for cluster_label in np.unique(cluster_labels):

        cluster_mask = cluster_labels == cluster_label
        cluster_subject_maps = saliency_maps[cluster_mask]
        print(f'Processing cluster {cluster_label} with {cluster_subject_maps.shape[0]} subjects')
        
        if cluster_subject_maps.shape[0] == 0:
            print(f' No subjects in cluster {cluster_label}, skipping.')
            continue
        
        cluster_subject_ids = [subject_ids[i] for i in range(saliency_maps.shape[0]) if cluster_mask[i]]

        # Keep only valid channels
        # shape -> (n_subjects_in_cluster, n_valid_channels, timepoints)
        cluster_subject_maps = cluster_subject_maps[:, valid_idx, :]
        for i, subject_maps in enumerate(cluster_subject_maps):
            subject_maps = np.abs(subject_maps)

            freqs, psd = welch(
                subject_maps,
                fs=sfreq,
                axis=1,
                nperseg=sfreq,
                noverlap=sfreq // 2
            )
            
            #psd = psd / (psd.sum() + 1e-8)
            psd = (psd - psd.min()) / (psd.max() - psd.min() + 1e-8)

            band_psd_list = []
            valid_band_names = []

            for band in freq_bands:
                fmin, fmax = freq_range[band]

                if band == 'all':
                    band_match = (freqs >= fmin) & (freqs <= fmax)
                else:
                    band_match = (freqs >= fmin) & (freqs < fmax)
                if not np.any(band_match):
                    print(f"No frequencies found in the {band} band for cluster {cluster_label}, subject {cluster_subject_ids[i]}. Skipping.")
                    continue

                band_psd = psd[:, band_match].mean(axis=1)
                band_psd_list.append(band_psd)
                valid_band_names.append(band)
            
            if len(band_psd_list) == 0:
                print(f"No valid frequency bands found for cluster {cluster_label}, subject {cluster_subject_ids[i]}. Skipping.")
                continue

            #Create one horizontal subplot for each valid band
            n_bands = len(band_psd_list)
            fig, axes = plt.subplots(1, n_bands, figsize=(6*n_bands, 5))
            if n_bands == 1:
                axes = [axes]
            for ax, band_psd, band_name in zip(axes, band_psd_list, valid_band_names):
                im, _ = mne.viz.plot_topomap(
                    band_psd,
                    info,
                    axes=ax,
                    show=False,
                    sensors=True,
                    names=valid_channels if len(filtered_show_channels) > 0 else None,
                )

                if len(filtered_show_channels) > 0 and ax.texts:
                    for txt in ax.texts:
                        if txt.get_text() not in filtered_show_channels:
                            txt.set_visible(False)
                
                plt.colorbar(im, ax=ax)
                ax.set_title(f'{band_name} band', fontsize=25)
            

            fig.suptitle(
                f'Class: {class_name}, Cluster: {cluster_label+1 if cluster_label != -1 else "Outliers"}, Subject ID: {cluster_subject_ids[i]}',
                fontsize=25
            )

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            savename = (
                f'{dataset_name}_{task}_{model_name}_iter_{best_iteration}_'
                f'{clustering_method}_cluster_{cluster_label}_subject_{i}_psd_topomap.png'
            )
            savepath = os.path.join(save_dir, savename)
            plt.savefig(savepath, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved topomap for cluster {cluster_label}, subject {i} at {savepath}")

# def plot_subject_psd_topomaps(
#         dataset_name, 
#         task, 
#         model_name, 
#         best_iteration, 
#         clustering_method, 
#         saliency_maps, 
#         cluster_labels,
#         used_channels, 
#         show_channels,
#         sfreq):

#     # This function can be implemented similarly to plot_cluster_psd_topomaps,
#     # but instead of averaging PSDs across subjects in a cluster, it would generate
#     # topomaps for each individual subject's PSD. The title and save path would also
#     # include the subject ID for clarity.

#     class_name = define_class_name(dataset_name, task)
#     freq_bands, freq_range = bands_and_range()
    
#     saliency_maps = np.asarray(saliency_maps)
#     cluster_labels = np.asarray(cluster_labels)

#     save_dir = f'results/clustering/{dataset_name}/cluster_psd_topomaps'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
    
#     montage = mne.channels.make_standard_montage('standard_1020')
#     ch_pos_dict = montage.get_positions()['ch_pos']

#     # Keep only channels present in montage
#     valid_channels = [ch for ch in used_channels if ch in ch_pos_dict]
#     if len(valid_channels) == 0:
#         print("No valid channels found in montage. Please check channel names.")
#         return

#     filtered_show_channels = [ch for ch in show_channels if ch in valid_channels]
#     valid_idx = [used_channels.index(ch) for ch in valid_channels]

#     # MNE info for valid channels
#     info = mne.create_info(ch_names=valid_channels, sfreq=sfreq, ch_types='eeg')
#     info.set_montage(montage)

#     # Check alignment
#     if saliency_maps.shape[0] != len(cluster_labels):
#         raise ValueError(
#             f"Mismatch: saliency_maps has {saliency_maps.shape[0]} subjects, "
#             f"but cluster_labels has {len(cluster_labels)} labels."
#         )
    
#     for cluster_label in np.unique(cluster_labels):

#         cluster_subject_maps = saliency_maps[cluster_labels == cluster_label]
#         print(f"Processing cluster {cluster_label} with {cluster_subject_maps.shape[0]} subjects")

#         if cluster_subject_maps.shape[0] == 0:
#             print(f' No subjects in cluster {cluster_label}, skipping.')
#             continue
        
#         # Keep only valid channels
#         # shape -> (n_subjects_in_cluster, n_valid_channels, timepoints)
#         cluster_subject_maps = cluster_subject_maps[:, valid_idx, :]

#         for i, subject_map in enumerate(cluster_subject_maps):
#             # shape: (channels, timepoints)
#             subject_map = np.abs(subject_map)
#             freqs, psd = welch(
#                 subject_map,
#                 fs=sfreq,
#                 axis=1,
#                 nperseg=sfreq,
#                 noverlap=sfreq // 2
#             )

#             # Subject-wise normalization
#             psd = psd / (psd.sum() + 1e-8)

#             for band in freq_bands:
#                 fmin, fmax = freq_range[band]

#                 if band == 'all':
#                     band_mask = (freqs >= fmin) & (freqs <= fmax)
#                 else:
#                     band_mask = (freqs >= fmin) & (freqs < fmax)

#                 if not np.any(band_mask):
#                     print(f"No frequencies found in the {band} band for cluster {cluster_label}, subject {i}. Skipping.")
#                     continue

#                 band_psd = psd[:, band_mask].mean(axis=1)
#                 fig, ax = plt.subplots(figsize=(6, 5))
#                 im, _ = mne.viz.plot_topomap(
#                     band_psd,
#                     info,
#                     axes=ax,
#                     show=False,
#                     sensors=True,
#                     names=valid_channels if len(filtered_show_channels) > 0 else None,
#                 )
#                 if len(filtered_show_channels) > 0 and ax.texts:
#                     for txt in ax.texts:
#                         if txt.get_text() not in filtered_show_channels:
#                             txt.set_visible(False)
#                 plt.colorbar(im, ax=ax)
#                 ax.set_title(
#                     f'{model_name}, iter {best_iteration}, {clustering_method}\n'
#                     f'{dataset_name}, {task}, Class: {class_name}\n' 
#                     f'Cluster {cluster_label}, Subject {i}, {band} band'
#                 )
#                 savename = (
#                     f'{dataset_name}_{task}_{model_name}_iter_{best_iteration}_'
#                     f'{clustering_method}_cluster_{cluster_label}_subject_{i}_{band}_psd_topomap.png'
#                 )
#                 savepath = os.path.join(save_dir, savename)
#                 plt.savefig(savepath, bbox_inches='tight')
#                 plt.close(fig)
#                 print(f"Saved topomap for cluster {cluster_label}, subject {i}, {band} band at {savepath}")


def plot_masking_curves(dataset_name, task, model_name, best_iteration,
                        gradient_methods, exp_key,
                        output_dir='results/clustering'):
    """
    For each cluster, plot MoRF (left) and LeRF (right) accuracy curves for all
    available gradient methods overlaid.  Horizontal lines mark baseline_cluster
    and baseline_fold.  One PNG per cluster.

    Reads faithfulness JSONs from:
        {output_dir}/{dataset_name}/faithfulness/{prefix}_{gm}_{exp_key}_faithfulness.json
    Saves PNGs to:
        {output_dir}/{dataset_name}/masking_curves/{prefix}_{exp_key}_cluster{c}.png
    """
    faith_dir = os.path.join(output_dir, dataset_name, 'faithfulness')
    save_dir  = os.path.join(output_dir, dataset_name, 'masking_curves')
    os.makedirs(save_dir, exist_ok=True)

    prefix = f'{dataset_name}_{task}_{model_name}_iteration_{best_iteration}'

    method_data = {}
    for gm in gradient_methods:
        path = os.path.join(faith_dir, f'{prefix}_{gm}_{exp_key}_faithfulness.json')
        if os.path.exists(path):
            with open(path) as f:
                method_data[gm] = json.load(f)

    if not method_data:
        print(f"No faithfulness data found for {dataset_name} | {task} | {exp_key} — skipping.")
        return

    first      = next(iter(method_data.values()))
    cluster_ids = sorted([k for k in first if k != 'overall'], key=int)
    colors     = plt.cm.tab10(np.linspace(0, 1, len(gradient_methods)))
    color_map  = {gm: colors[i] for i, gm in enumerate(gradient_methods)}

    for c in cluster_ids:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f'{dataset_name}  |  {task}  |  {exp_key}  |  Cluster {c}',
            fontsize=11
        )

        for ax, curve_key, title in [
            (axes[0], 'morf_curve', 'MoRF  (most relevant first)'),
            (axes[1], 'lerf_curve', 'LeRF  (least relevant first)'),
        ]:
            for gm, data in method_data.items():
                if str(c) not in data and c not in data:
                    continue
                entry  = data.get(str(c), data.get(c, {}))
                curve  = entry.get(curve_key, [])
                if not curve:
                    continue
                ks = list(range(1, len(curve) + 1))
                ax.plot(ks, curve, label=gm, color=color_map[gm], linewidth=1.5)

            # Baseline lines (from first available method — same model, same fold)
            first_entry = first.get(str(c), first.get(c, {}))
            bc = first_entry.get('baseline_cluster')
            bf = first_entry.get('baseline_fold')
            if bc is not None:
                ax.axhline(bc, color='black',  linestyle='--', linewidth=1.0,
                           label=f'baseline cluster ({bc:.2f})')
            if bf is not None:
                ax.axhline(bf, color='dimgray', linestyle=':',  linewidth=1.0,
                           label=f'baseline fold ({bf:.2f})')
            ax.axhline(0.5, color='red', linestyle=':', linewidth=0.8, alpha=0.5,
                       label='chance (0.5)')

            ax.set_xlabel('Masking step k', fontsize=10)
            ax.set_ylabel('Accuracy',        fontsize=10)
            ax.set_title(title,              fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7, loc='upper right')

        plt.tight_layout()
        savename  = f'{prefix}_{exp_key}_cluster{c}_masking_curves.png'
        savepath  = os.path.join(save_dir, savename)
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved masking curves [{exp_key}] cluster {c} -> {savepath}")
                    