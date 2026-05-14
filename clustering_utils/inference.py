import numpy as np
import torch
import json
import time
import os

from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset, Subset
from utils.tests import test_overlaps
    

def inference(dataset_name, task, model, samples, targets, groups, best_iteration, best_params, random_seeds):
    
    
    unique_subjects = np.unique(groups)
    np.random.shuffle(unique_subjects)

    embeddings_list = []
    labels_list = []
    subject_ids_list = []
    kf = StratifiedGroupKFold(n_splits=5, shuffle=False)
    for fold, (_, test_index) in enumerate(kf.split(samples, targets, groups=groups)):
        
        print(f"Fold {fold + 1} - Test indices: {test_index}")
        test_subjects = np.unique(groups[test_index])
        print(f"Fold {fold + 1} - Test subjects: {test_subjects}")
        print('---------------------------------------------')


        groups = groups.astype(int)
        test_samples = torch.from_numpy(samples[test_index]).unsqueeze(1).float()
        test_targets = torch.from_numpy(targets[test_index]).float()
        test_groups = torch.from_numpy(groups[test_index]).long()
        test_dataset = TensorDataset(test_samples, test_targets, test_groups)
        test_loader = DataLoader(test_dataset, batch_size=128, 
                                shuffle=False, drop_last=False,
                                num_workers=0, pin_memory=True)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('cuda is available: ', torch.cuda.is_available())
        print('device count: ', torch.cuda.device_count())
        if torch.cuda.is_available():
            print("name:", torch.cuda.get_device_name(0)) 
            print("torch:", torch.__version__, "cuda build:", torch.version.cuda)

        
        net = model['class'](**model['kwargs'], tsne=True)
        weights_path = f'results/weights/{dataset_name}/{dataset_name}_{task}_{model["name"]}_iteration_{best_iteration}_fold_{fold}_best_weights.pth'
        if os.path.exists(weights_path):
            print(f"Loading model weights from {weights_path}")
            # model.load_state_dict(torch.load(weights_path, map_location=device))
            net.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            print(f"Warning: Weights file {weights_path} not found. Skipping fold {fold + 1}.")
            raise KeyError(f'Weights file {weights_path} not found.')
        
        net.to(device)
        net.eval()
        fold_embeddings = []
        fold_labels = []
        fold_subject_ids = []
        with torch.no_grad():
            for bx, by, bg in test_loader:
                bx, by, bg = bx.to(device), by.to(device), bg.to(device)
                
                b_embeddings = net(bx)

                if task == 'MCI vs Dementia':
                    mask = (by == 0)
                else:
                    mask = (by == 1)
                if mask.sum() > 0:
                    fold_embeddings.append(b_embeddings[mask].cpu().numpy())
                    fold_labels.append(by[mask].cpu().numpy())
                    fold_subject_ids.append(bg[mask].cpu().numpy())

        fold_embeddings = np.concatenate(fold_embeddings, axis=0)
        fold_labels = np.concatenate(fold_labels, axis=0) # reshape(-1, 1)
        fold_subject_ids = np.concatenate(fold_subject_ids, axis = 0) # reshape(-1, 1)
        
        print(f'Fold {fold + 1} - Embeddings shape: {fold_embeddings.shape}, Labels shape: {fold_labels.shape}')
        print(f'Fold {fold + 1} - Subject IDs shape: {fold_subject_ids.shape}')
        
        #aggregate to subject level
        subject_embeddings = []
        subject_labels = []
        subject_ids = []
        for subject_id in np.unique(fold_subject_ids):
            subject_mask = (fold_subject_ids == subject_id)
            if subject_mask.sum() > 0:
                subject_embeddings.append(fold_embeddings[subject_mask].mean(axis=0))
                subject_labels.append(fold_labels[subject_mask][0]) # assuming all samples from the same subject have the same label
                subject_ids.append(subject_id)
        subject_embeddings = np.array(subject_embeddings)
        subject_labels = np.array(subject_labels).reshape(-1, 1)
        subject_ids = np.array(subject_ids).reshape(-1, 1)

        embeddings_list.append(subject_embeddings)
        labels_list.append(subject_labels)
        subject_ids_list.append(subject_ids)

        save_dir = f'results/embeddings/{dataset_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        np.save(
            os.path.join(save_dir, f'{dataset_name}_{task}_{model["name"]}_iteration_{best_iteration}_fold_{fold}_subject_embeddings.npy'), 
            subject_embeddings
        )
        np.save(
            os.path.join(save_dir, f'{dataset_name}_{task}_{model["name"]}_iteration_{best_iteration}_fold_{fold}_subject_labels.npy'), 
            subject_labels
        )
        np.save(
            os.path.join(save_dir, f'{dataset_name}_{task}_{model["name"]}_iteration_{best_iteration}_fold_{fold}_subject_ids.npy'), 
            subject_ids
        )
        print('---------------------------------------------')

    return embeddings_list, labels_list ,subject_ids_list
        

                            
                
     
