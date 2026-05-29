import numpy as np
import torch
import json
import time
import os
from captum.attr import Saliency, InputXGradient, NoiseTunnel, IntegratedGradients

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset, Subset
from utils.tests import test_overlaps

from collections import Counter

GRADIENT_METHODS = ['vanilla', 'input_x_gradient', 'smoothgrad', 'smoothgrad_sq', 'vargrad', 'integrated_gradients']

def _build_attributor(method_name, forward_func):
    if method_name == 'vanilla':
        return Saliency(forward_func)
    elif method_name == 'input_x_gradient':
        return InputXGradient(forward_func)
    elif method_name in ('smoothgrad', 'smoothgrad_sq', 'vargrad'):
        return NoiseTunnel(Saliency(forward_func))
    elif method_name == 'integrated_gradients':
        return IntegratedGradients(forward_func)
    else:
        raise ValueError(f"Unknown gradient method: {method_name}. Choose from {GRADIENT_METHODS}")

def _attribute(attributor, method_name, bx, target):
    if method_name == 'vanilla':
        return attributor.attribute(bx, target=target, abs=False)
    elif method_name == 'input_x_gradient':
        return attributor.attribute(bx, target=target)
    elif method_name in ('smoothgrad', 'smoothgrad_sq', 'vargrad'):
        return attributor.attribute(bx, nt_type=method_name, nt_samples=50,
                                    stdevs=0.1, target=target)
    elif method_name == 'integrated_gradients':
        return attributor.attribute(bx, target=target, n_steps=50)

def captum_forward(model):
    def forward(x):
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        return out
    return forward
    

def inference(dataset_name, task, model, samples, targets, groups, best_iteration, best_params, random_seeds, gradient_method='vanilla'):
    
    
    unique_subjects = np.unique(groups)
    np.random.shuffle(unique_subjects)

    gradients_list = []
    negative_gradients_list = [] # to store gradients of correctly predicted negative subjects
    subject_fold_map = {}  # subject_id -> fold index (used for per-fold faithfulness evaluation)
    # (HC for every task with the HC group, Dementia for MCI vs Dementia, and FTD for AD vs FTD)
    subject_ids_list = []
    negative_subject_ids_list = [] # to store subject ids of correctly predicted negative subjects
    kf = StratifiedGroupKFold(n_splits=5, shuffle=False)
    for fold, (_, test_index) in enumerate(kf.split(samples, targets, groups=groups)):
        
        print(f"Fold {fold + 1} - Test indices: {test_index}")
        test_subjects = np.unique(groups[test_index])
        print(f"Fold {fold + 1} - Test subjects: {test_subjects}")
        print('---------------------------------------------')

        if dataset_name == 'ADvsFTDvsHC':
            encoder = LabelEncoder()
            groups = encoder.fit_transform(groups) + 1

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

        
        net = model['class'](**model['kwargs'], tsne=False)
        weights_path = f'results/weights/{dataset_name}/{dataset_name}_{task}_{model["name"]}_iteration_{best_iteration}_fold_{fold}_best_weights.pth'
        if os.path.exists(weights_path):
            print(f"Loading model weights from {weights_path}")
            # model.load_state_dict(torch.load(weights_path, map_location=device))
            net.load_state_dict(torch.load(weights_path, map_location=device))
        else:
            print(f"Warning: Weights file {weights_path} not found. Skipping fold {fold + 1}.")
            raise KeyError(f'Weights file {weights_path} not found.')
        

        saliency_inst = _build_attributor(gradient_method, captum_forward(net))
        net.to(device)
        net.eval()
        fold_gradients = []
        fold_preds = []
        fold_labels = []
        fold_subject_ids = []
        
        for bx, by, bg in test_loader:

            bx, by, bg = bx.to(device), by.to(device), bg.to(device)
            
            bx = bx.clone().detach().requires_grad_(True)

            output = net(bx)
            preds = output.argmax(dim=1)

            fold_preds.append(preds.detach().cpu().numpy())
            fold_labels.append(by.detach().cpu().numpy())
            fold_subject_ids.append(bg.detach().cpu().numpy())

            target = by.detach().cpu().long().tolist()
            fold_gradients.append(
                _attribute(saliency_inst, gradient_method, bx, target).detach().cpu().numpy()
            )
        fold_gradients = np.concatenate(fold_gradients, axis = 0)

        print(f'Fold {fold + 1} gradients shape: {fold_gradients.shape}')
        if fold_gradients.shape[1] == 1:
            fold_gradients = np.squeeze(fold_gradients, axis=1)
        
        print(f'Fold {fold + 1} gradients shape after squeezing: {fold_gradients.shape}')
        #exit()
        fold_preds = np.concatenate(fold_preds, axis = 0)
        fold_labels = np.concatenate(fold_labels, axis = 0)
        fold_subject_ids = np.concatenate(fold_subject_ids)

        #majority_voting (find the correctly predicted subjects)
        if task == 'MCI vs Dementia':
            target_class = 0 # in this task MCI is labeled as 0
            negative_class = 1 # Dementia is labeled as 1
        else:
            target_class = 1 # in other tasks positives are labeled as 1
            negative_class = 0 # negatives are labeled as 0

        correct_subjects = []
        correct_negative = []
        av_gradients_of_correct_subjects = []
        av_gradients_of_correct_negative_subjects = []
        for id_ in np.unique(fold_subject_ids):

            mask = (fold_subject_ids == id_)
            
            true_label = fold_labels[mask][0]
            # print(f'ID {id_}, predictions: {fold_preds[mask]}')
            #exit()
            pred_label = Counter(fold_preds[mask]).most_common(1)[0][0]
            if pred_label == target_class and true_label == target_class:
                av_gradients_of_correct_subjects.append(fold_gradients[mask].mean(axis=0))
                # av_gradients_of_correct_subjects.append(scaled_fold_gradients[mask].mean(axis=0))
                correct_subjects.append(id_)
                subject_fold_map[int(id_)] = fold
            elif pred_label == negative_class and true_label == negative_class:
                av_gradients_of_correct_negative_subjects.append(fold_gradients[mask].mean(axis=0))
                correct_negative.append(id_)

        print(f'Fold {fold + 1} number of the correct subjects: {len(correct_subjects)}')
        print(f'Fold {fold + 1} number of the correct negative subjects: {len(correct_negative)}')

        gradients_list.append(np.array(av_gradients_of_correct_subjects))
        negative_gradients_list.append(np.array(av_gradients_of_correct_negative_subjects))
        subject_ids_list.append(np.array(correct_subjects))
        negative_subject_ids_list.append(np.array(correct_negative))

        save_dir = f'results/saliency_maps/{dataset_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # np.save(
        #     os.path.join(save_dir, f'{dataset_name}_{task}_{model["name"]}_iteration_{best_iteration}_fold_{fold}_saliency_maps.npy'), 
        #     av_gradients_of_correct_subjects
        # )
        # np.save(
        #     os.path.join(save_dir, f'{dataset_name}_{task}_{model["name"]}_iteration_{best_iteration}_fold_{fold}_subject_ids.npy'), 
        #     correct_subjects
        # )
        # print('---------------------------------------------')

    gradients_list = np.concatenate(gradients_list)
    subject_ids_list = np.concatenate(subject_ids_list)
    negative_gradients_list = np.concatenate(negative_gradients_list)
    negative_subject_ids_list = np.concatenate(negative_subject_ids_list)
    print(f'The saliency maps: {gradients_list.shape}\nThe correctly predicted subjects shape: {subject_ids_list.shape}')
    # print(f'Subject IDs: {subject_ids_list}')

    np.save(
        os.path.join(save_dir, f'{dataset_name}_{task}_{model["name"]}_iteration_{best_iteration}_{gradient_method}_saliency_maps.npy'),
        gradients_list
    )

    return gradients_list, negative_gradients_list, subject_ids_list, negative_subject_ids_list, subject_fold_map
        









            
        

                            
                
     
