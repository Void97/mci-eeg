import os
import mne
import numpy as np
import asrpy
import torch
import json
from mne import Annotations

def preprocess(metadata, dataset_path, preprocessed_dir,
               dataset_name, sfreq=None, ch_names=None, ch_types=None,
               new_sfreq=None):
    
    for i, row in metadata.iterrows():
        #subject_id = str(row['id'])
        #label = row['status']
            
            ###READ THE FILE
        if dataset_name == 'GENEEG':
            subject_id = str(row['id'])
            eeg_path = os.path.join(dataset_path, f'{subject_id}.eeg')
            if os.path.exists(eeg_path):    
                eeg = np.loadtxt(eeg_path)
                print(f'{subject_id}.eeg has been loaded')
                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
                raw = mne.io.RawArray(eeg.T, info, verbose=False)
        elif dataset_name == 'MCIvsHC':  
            subject_id = str(row['file number']) 
            eeg_path = os.path.join(dataset_path, f'{subject_id}.edf')
            if os.path.exists(eeg_path):
                raw = mne.io.read_raw_edf(eeg_path, preload=True)
        elif dataset_name == 'ADvsFTDvsHC':
            subject_id = row['participant_id']
            eeg_path = os.path.join(dataset_path, subject_id, 'eeg', f'{subject_id}_task-eyesclosed_eeg.set')
            #eeg_path = os.path.join(dataset_path, 'filtered', f'{subject_id}_preprocessed.set')            
            if os.path.exists(eeg_path):
                raw = mne.io.read_raw_eeglab(eeg_path)
                print(raw)
        elif dataset_name == 'CAUEEG':
            channels_to_drop = ["EKG", "Photic"]
            subject_id = str(row['id'])
            eeg_path = os.path.join(dataset_path, f'{subject_id}.edf')
            if os.path.exists(eeg_path):
                raw = mne.io.read_raw_edf(eeg_path, preload=True)
            raw.drop_channels(channels_to_drop)
            
        ###PREPROCESS THE FILE
        if new_sfreq:
            raw.resample(new_sfreq)
        raw.filter(l_freq = 0.5, h_freq = 45, fir_design='firwin')
        raw.set_eeg_reference(ref_channels="average")
          
        raw.apply_function(lambda x: (x - x.mean()) / x.std(), channel_wise=True) #z-score normalization
        
        ##############################
        #REMOVE BAD EVENTS FOR CAUEEG
        ##############################
        if dataset_name == 'CAUEEG':
            event_path = f"./datasets/raw/CAUEEG/event/{subject_id}.json"
            with open(event_path, "r") as f:
                events = json.load(f)
            
            data, times = raw.get_data(return_times=True)
            segments_to_remove = []
            for i, (sample, desc) in enumerate(events):
                if desc == "Eyes Open":
        # find the next "Eyes Closed"
                    for j in range(i+1, len(events)):
                        if events[j][1] == "Eyes Closed":
                            start = sample
                            end = events[j][0]
                            segments_to_remove.append((start, end))
                            break
                elif (
                    desc == "Photic On - 3.0 Hz" or
                    desc == "Photic On - 6.0 Hz" or
                    desc == "Photic On - 9.0 Hz" or
                    desc == "Photic On - 12.0 Hz" or
                    desc == "Photic On - 15.0 Hz" or
                    desc == "Photic On - 18.0 Hz" or
                    desc == "Photic On - 21.0 Hz" or
                    desc == "Photic On - 24.0 Hz" or
                    desc == "Photic On - 27.0 Hz" or 
                    desc == "Photic On - 30.0 Hz" 
                ):
                    for j in range(i+1, len(events)):
                        if events[j][1] == "Photic Off":
                            start = sample
                            end = events[j][0]
                            segments_to_remove.append((start, end))
                            break
            
            n_samples = data.shape[1]
            mask = np.ones(n_samples, dtype=bool)
            for start, end in segments_to_remove:
                mask[start:end] = False
            data_clean = data[:, mask]
            raw = mne.io.RawArray(data_clean, raw.info)
            
            
        epochs = mne.make_fixed_length_epochs(
                raw, duration = 4, overlap = 0.5, preload=True)
        np_epochs = epochs.get_data().astype(np.float32)

        if not os.path.exists(preprocessed_dir):
            os.makedirs(preprocessed_dir)
        filename = os.path.join(preprocessed_dir, f'{subject_id}.npy')
        np.save(filename, np_epochs)

    return

def load_preprocessed(preprocessed_dir, label_map, label_dict):
    samples = []
    targets = []
    subjects = []
    for id, label in label_dict.items():
        file = os.path.join(preprocessed_dir, f'{id}.npy')
        if os.path.exists(file):
            data = np.load(file)
            for epoch in data:
                samples.append(torch.tensor(epoch, dtype=torch.float32).unsqueeze(0))
                targets.append(label_map[label])
                subjects.append(id)

    samples = np.concatenate(samples, axis = 0)
    targets  = np.array(targets, dtype=np.int32)
    groups   = np.array(subjects, dtype=str) 
    print(samples.shape)
    print(groups)
    
    print("Loaded:", samples.shape, targets.shape, groups.shape)
    return samples, targets, groups
