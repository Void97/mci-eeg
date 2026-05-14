import numpy as np
import torch
import json
import time
import os

from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import TensorDataset, DataLoader, Subset
from utils.tests import test_overlaps
from torch.optim import AdamW, SGD
from torch.nn import CrossEntropyLoss
from models.models_base.MSVTNet import JointCrossEntoryLoss
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from models.models_train.k_fold_logging import count_fold_subjects
from main_func_111125 import subject_wise_metrics

from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from main_func import set_seed
from captum.attr import Saliency

def shuffle(samples, targets, groups, dataset):
            
        ### define the unique subjects and shuffle them
        unique_subjects = np.unique(groups)
        np.random.shuffle(unique_subjects)
        new_indices = []
        for sub in unique_subjects:
            sub_i = np.where(groups == sub)[0]
            new_indices.extend(sub_i)
        shuffled_samples = samples[new_indices]
        shuffled_targets = targets[new_indices]
        shuffled_groups = groups[new_indices]
        shuffled_dataset = torch.utils.data.TensorDataset(*[tensor[new_indices] for tensor in dataset.tensors])
        
        return shuffled_samples, shuffled_targets, shuffled_groups, shuffled_dataset

def captum_forward_fn(model):
    def forward(x):
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        return out
    return forward

def subsamples(seed, idx, groups, fraction=0.1, min_samples=1):
    
    random_gen = np.random.RandomState(seed)
    idx = np.asarray(idx)
    groups = groups[idx]
    
    sub_idx = []
    for subject in np.unique(groups):
        mask = (groups == subject)
        subject_idx = idx[mask]
        total_l = len(subject_idx)
        
        kept_n = max(min_samples, int(np.ceil(total_l * fraction)))
        if kept_n >= total_l:
            chosen_idx = subject_idx
        else:    
            chosen_idx = random_gen.choice(subject_idx, size=kept_n, replace=False)
        sub_idx.extend(chosen_idx)
    return np.array(sub_idx)

def train_10_K_fold(seed, dirs, dataset_name, task, model_class, model_name, model_kwargs, best_params, dataset, samples, targets, groups, k, iter):
    
    subjects_to_label = {}
    k_fold_subjects_logs = []
    
    def tsne():
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if dataset_name == 'GENEEG' or dataset_name == 'MCIvsHC':
            class1 = 'Healthy Controls'
            class2 = 'MCI'  # or ['Control', 'MCI'] but be consistent across your code
        elif dataset_name == 'ADvsFTDvsHC':
            if task == 'AD vs HC':
                class1 = 'Healthy Controls'
                class2 = 'AD'
            elif task == 'FTD vs HC':
                class1 = 'Healthy Controls'
                class2 = 'FTD'
            elif task == 'FTD vs AD':
                class1 = 'FTD'
                class2 = 'AD'
        elif dataset_name == 'CAUEEG':
            if task == 'Dementia vs Normal':
                class1 = 'Normal'
                class2 ='Dementia'
            elif task == 'MCI vs Normal':
                class1 = 'Normal' 
                class2 = 'MCI'
            elif task == 'Dementia vs MCI':
                class1 = 'MCI' 
                class2 = 'Dementia'
        
        model = model_class(**model_kwargs, tsne = True)
        model.load_state_dict(best_model_state)
        model.to(device)
        model.eval()
        
        features = []
        labels = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                feats = model(xb)
                features.extend(feats.cpu())
                labels.extend(np.array(yb.cpu()))
        features = np.array(features)
        print(f'Features shape: {features.shape}')
        labels = np.array(labels).reshape(-1, 1)
        print(f'Labels shape: {labels.shape}')
        
        #TSNE
        tsne = TSNE(n_components=2, n_iter = 3000, random_state = 42)
        features_2d = tsne.fit_transform(features)
        
        scaler = MinMaxScaler(feature_range=(0,1))
        features_norm = scaler.fit_transform(features_2d)
        print(f'Features norm shape: {features_norm.shape}')
        
        
        tsne_save_dir = dirs['tsne_dir']
        if not os.path.exists(tsne_save_dir):
            os.makedirs(tsne_save_dir)
            
        plt.figure(figsize=(10, 8))
        colors = ['blue' if label == 0 else 'red' for label in labels]
        plt.scatter(
            [point[0] for point in features_norm],  # X axis
            [point[1] for point in features_norm],  # Y axis
            c=colors,                             # point color
            alpha=0.5,                            # point transparency
        )
            
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label=class1),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label=class2)
        ]

        plt.legend(handles=legend_elements, fontsize=14, loc='upper right')  # set legend
        
        plt.savefig(os.path.join(tsne_save_dir, f'{task}_{model_name}_iteration_{iter}_fold_{fold}_tsne.png'))
        plt.close()
        print("t-SNE plot is saved!")
        
    def timing_logs():
        
        timing_log = {
            "model_name": model_name,
            "task": task,
            "num_folds": k,
            "num_trainable_parameters": num_params,
            "training_time_(sec/epoch)": round(overall_avg_train_time, 2),
            "inference_time_(sec/segment)": round(overall_avg_inference_time, 2),
            "average_peak_gpu_memory_MB": round(average_peak_gpu_memory, 2)
        }
        
        timing_log_path = os.path.join(
            dirs['training_inference_time_logs_dir'], 
            f'{model_name}_{task}_timing_log.json'
        )
        
        if os.path.exists(timing_log_path):
            print("Timing log already exists at: {timing_log_path}.\n"
                  "Skipping writing new training/inference times.")
        else:
            with open(timing_log_path, 'w') as f:
                json.dump(timing_log, f, indent=2)
            print(f"n\Timing log saved to: {timing_log_path}")
            
    set_seed(seed)
    for subject,label in zip(groups, targets):
        subjects_to_label[subject] = label
    
    all_avg_train_times = []
    all_avg_inference_times = []
    all_peak_memory = []
    
    all_seg_acc = [] 
    all_sub_acc = [] 
    all_sub_prec = [] 
    all_sub_sens = [] 
    all_sub_spec = [] 
    all_sub_f1 = []
    
    unique_subjects = np.unique(groups) 
    np.random.shuffle(unique_subjects) # shuffle subjects randomly 
    #outer_kf = StratifiedGroupKFold(n_splits=int(k), shuffle=True, random_state=seed) 
    outer_kf = StratifiedGroupKFold(n_splits=int(k), shuffle=False)
    for fold, (train_val_idx, test_idx) in enumerate(outer_kf.split(samples, targets, groups=groups)):
        
        print(f'Fold {fold+1}: {len(train_val_idx)} train / {len(test_idx)} test samples')
        train_val_groups = groups[train_val_idx] 
        test_groups = groups[test_idx].tolist()
        
        # === check overlaps on subject IDs ===
        print('Global K-fold overlaps check') 
        test_overlaps(fold, train_val_groups, test_groups)
        
        inner_k = int(k) - 1
        inner_kf = StratifiedGroupKFold(n_splits=inner_k, shuffle=False)
        target_inner_fold = fold % inner_k 
        train_idx, val_idx = None, None
        
        for inner_fold, (tr_i, val_i) in enumerate(
            inner_kf.split(samples[train_val_idx], targets[train_val_idx], 
                           groups=groups[train_val_idx])):
            
            if inner_fold == target_inner_fold: 
                train_idx, val_idx = tr_i, val_i
                break    
        
        if train_idx is None or val_idx is None: 
            raise RuntimeError("Inner K-fold splitting failed to produce train/val indeces.")
        print('Inner K-fold overlaps check')
    
        test_overlaps(fold,
                    groups[train_val_idx][train_idx],
                    groups[train_val_idx][val_idx])

        one_fold_subject_logs = count_fold_subjects(groups, train_val_idx, train_idx, val_idx, test_idx, subjects_to_label, fold)
        k_fold_subjects_logs.append(one_fold_subject_logs)
        
        train_subset = Subset(dataset, train_val_idx[train_idx]) 
        val_subset = Subset(dataset, train_val_idx[val_idx]) 
        test_subset = Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_subset, batch_size=best_params['batch size'],
                                    shuffle=True, drop_last=True) 
        val_loader = DataLoader(val_subset, batch_size=64,
                                shuffle=False, drop_last=False,
                                num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_subset, batch_size=128,
                                    shuffle=False, drop_last=False,
                                    num_workers=0, pin_memory=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("cuda available:", torch.cuda.is_available()) 
        print("device count:", torch.cuda.device_count()) 
        if torch.cuda.is_available(): 
            print("name:", torch.cuda.get_device_name(0)) 
            print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
            
        best_model_dir = dirs['weights_dir']
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        best_model_file_path = best_model_file_path = os.path.join( best_model_dir, 
                                                                    f'{dataset_name}_{task}_{model_name}_iteration_{iter}_fold_{str(fold)}_best_weights.pth')
        
        model = model_class(**model_kwargs, tsne = False).to(device)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
        print(f'Model has {num_params:,} trainabe parameters') 
        if torch.cuda.is_available(): 
            torch.cuda.reset_peak_memory_stats(device)
        
        if os.path.exists(best_model_file_path): 
            print(f"Found existing weights for {dataset_name} {task} {model_name} iteration {iter} fold {fold+1}.\n" "Skipping the training and validation parts") 
            best_model_state = torch.load(best_model_file_path, map_location=device) 
            #model.load_state_dict(best_model_state) 
            av_epoch_train_time = 0.0 
            all_avg_train_times.append(av_epoch_train_time)
        else:
            if model_name == "MBSzEEGNet":
                optimizer = SGD(model.parameters(),
                                lr=best_params['learning rate'],
                                weight_decay=best_params['L2 weight decay']
                                )
            else:
                optimizer = AdamW(model.parameters(),
                                    lr=best_params['learning rate'],
                                    weight_decay=best_params['L2 weight decay']
                                    )
            if model_name == "MSVTNet":
                criterion = JointCrossEntoryLoss()
            else:
                criterion = CrossEntropyLoss()
            
            best_val_acc = 0
            counter = 0
            patience = 30
            epoch_training_times = []
            for epoch in range(100):
                
                epoch_start = time.perf_counter()
                
                model.train()
                train_loss = 0.0
                all_tr_p, all_tr_l = [], []
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device) 
                    optimizer.zero_grad()
                    out = model(xb)
                    if model_name == 'MBSzEEGNet':
                        loss = criterion(out, yb.long())
                    else:
                        loss = criterion(out, yb.long())
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()*yb.size(0)
                    if model_name == 'MSVTNet':
                        preds = out[0].argmax(dim=1)
                    else:
                        preds = out.argmax(dim=1)
                    all_tr_p.extend(preds.cpu().numpy())
                    all_tr_l.extend(yb.cpu().numpy())
                train_loss /= len(train_subset)
                train_acc = accuracy_score(all_tr_l, all_tr_p)
                
                #Validation
                model.eval()
                val_loss, all_v_p, all_v_l = 0.0, [], []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        out = model(xb)
                        if model_name == 'MBSzEEGNet': 
                            loss = criterion(out, yb.long()) 
                        else: 
                            loss = criterion(out, yb.long())
                        val_loss += loss.item()*yb.size(0)
                        if model_name == 'MSVTNet':
                            preds = out[0].argmax(dim=1)
                        else:
                            preds = out.argmax(dim=1)
                        all_v_p.extend(preds.cpu().numpy()) 
                        all_v_l.extend(yb.cpu().numpy())
                    val_loss /= len(val_subset) 
                val_acc = accuracy_score(all_v_l, all_v_p) 
                val_f1 = f1_score(all_v_l, all_v_p) if model_kwargs['num_classes'] == 2 else f1_score(all_v_l, all_v_p, average='weighted')
                
                epoch_end = time.perf_counter()
                epoch_training_times.append(epoch_end-epoch_start)
                
                if (epoch+1)%10 == 0:
                    print(f"Epoch {epoch + 1}: " f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, " f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}, " f"Validation F1-score: {val_f1:.4f}")
                    
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict()
                    counter = 0
                else:
                    counter += 1
                if counter > patience:
                    print(f"Epoch {epoch + 1}: " f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, " f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}, " f"Validation F1-score: {val_f1:.4f}")
                    break
            
            torch.save(best_model_state, best_model_file_path)
            print("The model is saved!")
            
            av_epoch_train_time = sum(epoch_training_times) / len(epoch_training_times)
            print(f"Average training time per epoch for Fold {fold+1}: {av_epoch_train_time:.2f} seconds") 
            all_avg_train_times.append(av_epoch_train_time)
            
        tsne()
            
            #Test with the best weights
        gradient_list = [] 
        gradient_batches = []
        saliency_inst = Saliency(captum_forward_fn(model)) 
        model.load_state_dict(best_model_state) 
            
        metrics_device = device 
        if model_name == 'SzHNN': 
            metrics_device = torch.device('cpu') 
        model.to(metrics_device) 
        
        test_preds_fold, test_labels_fold, test_roc = [], [], [] 
        inference_times = []
        with torch.no_grad():
            start_time = time.perf_counter()
            for xb, yb in test_loader:
                
                xb.requires_grad = False
                xb, yb = xb.to(metrics_device), yb.to(metrics_device) 
                out = model(xb)                     
                if model_name == 'MSVTNet':
                    test_preds = out[0].argmax(dim=1)
                else:    
                    test_preds = out.argmax(dim=1)
                test_preds_fold.extend(test_preds.cpu().numpy())
                test_labels_fold.extend(yb.cpu().numpy())
                if model_name == 'MSVTNet':
                    test_roc.extend(torch.sigmoid(out[0]).cpu().numpy())
                else:
                    test_roc.extend(torch.sigmoid(out).cpu().numpy())
                
                xb.requires_grad = True
                target = yb.detach().cpu().long().tolist()
                gradient_batches.append(
                    saliency_inst.attribute(
                        xb,
                        target=target,
                        abs=False,
                    ).detach().cpu().numpy())
            
            end_time = time.perf_counter() 
            inference_times.append((end_time - start_time) / xb.size(0))
            #print(f"Average inference time per segment for Fold {fold+1}: {avg_inference_time:.5f} seconds")
            
            gradient_list = np.concatenate(gradient_batches)
            if gradient_list.shape[1] == 1:
                gradient_list = np.squeeze(gradient_list, axis=1)
            
            saliency_maps = {class_id: [] for class_id in np.unique(test_labels_fold)}
            for class_id in np.unique(test_labels_fold):
                correct_indices = np.where((np.array(test_labels_fold) == class_id) &
                                        (np.array(test_preds_fold) == class_id))[0]
                if correct_indices.size > 0:
                    saliency_maps[class_id].append(gradient_list[correct_indices])
            for i in saliency_maps:
                saliency_maps[i] = np.concatenate(saliency_maps[i]) if saliency_maps[i] else np.array([])
        
        
        avg_inference_time = sum(inference_times) / len(inference_times)
        all_avg_inference_times.append(avg_inference_time)
        print(f"Average inference time per segment for Fold {fold+1}: {avg_inference_time:.5f} seconds")

        seg_acc = accuracy_score(test_labels_fold, test_preds_fold) 
        sub_acc, sub_sens, sub_spec, sub_prec, sub_f1 = subject_wise_metrics(test_labels_fold, test_preds_fold, test_groups) 
        all_seg_acc.append(seg_acc) 
        all_sub_acc.append(sub_acc) 
        all_sub_sens.append(sub_sens) 
        all_sub_spec.append(sub_spec) 
        all_sub_prec.append(sub_prec) 
        all_sub_f1.append(sub_f1)
        
        print(f"Fold {fold+1}\n" 
                f"Test Accuracy (seg.): {seg_acc:.4f}\n" 
                f"Test Accuracy (sub.): {sub_acc:.4f}\n" 
                f"Test Sensitivity: {sub_sens:.4f}\n" 
                f"Test Specificity: {sub_spec:.4f}\n" 
                f"Test Precision: {sub_prec:.4f}\n" 
                f"Test F1: {f1_score(test_labels_fold, test_preds_fold):.4f}")            
        
        # GPU Memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_gpu_memory_bytes = torch.cuda.max_memory_allocated(device)
            peak_gpu_memory_mb = peak_gpu_memory_bytes / (1024 ** 2)
        else:
            peak_gpu_memory_mb = None
    
        if peak_gpu_memory_mb is not None:
            print(f"GPU memory usage for Fold {fold + 1}: {peak_gpu_memory_mb:.2f} MB")
        else:
            print("No GPU used.")
        all_peak_memory.append(peak_gpu_memory_mb if peak_gpu_memory_mb is not None else 0)

    overall_avg_train_time = sum(all_avg_train_times) / len(all_avg_train_times)
    overall_avg_inference_time = sum(all_avg_inference_times) / len(all_avg_inference_times)
    average_peak_gpu_memory = sum(all_peak_memory) / len(all_peak_memory)
    print(f"\n=== Overall Averages Across {k} Folds ===")
    print(f"Average training time per epoch: {overall_avg_train_time:.2f} seconds")
    print(f"Average inference time per segment: {overall_avg_inference_time:.5f} seconds")
    print(f"Average peak GPU memory: {average_peak_gpu_memory} MB")
    timing_logs()
    
    av_fold_seg_acc = np.round(np.mean(all_seg_acc), 4) 
    av_fold_sub_acc = np.round(np.mean(all_sub_acc), 4) 
    av_fold_sub_sens = np.round(np.mean(all_sub_sens), 4) 
    av_fold_sub_spec = np.round(np.mean(all_sub_spec), 4) 
    av_fold_sub_prec = np.round(np.mean(all_sub_prec), 4) 
    av_fold_sub_f1 = np.round(np.mean(all_sub_f1), 4)
    
    print(f"=== Overall Metrics for {dataset_name} {task} {model_name} iteration {iter} across {k} Folds ===\n" 
          f"Accuracy (seg.): {av_fold_seg_acc}\n" 
          f"Accuracy (sub.): {av_fold_sub_acc}\n" 
          f"Sensitivity: {av_fold_sub_sens}\n" 
          f"Specificity: {av_fold_sub_spec}\n" 
          f"Precision: {av_fold_sub_prec}\n" 
          f"F1: {av_fold_sub_f1}")

    log_dir = os.path.join(dirs['k_fold_dist_logs_dir'], f'{dataset_name}_{task}_{model_name}_iteration_{iter}_kfold_subjects_log.json') 
    with open(log_dir, 'w') as f: 
        json.dump(k_fold_subjects_logs, f, indent=2)
        
    return av_fold_seg_acc, av_fold_sub_acc, av_fold_sub_sens, av_fold_sub_spec, av_fold_sub_prec, av_fold_sub_f1, saliency_maps




### Pilot training function
def train_10_K_fold_pilot(seed, task, model_class, model_name, model_kwargs,
                          batch_sizes, lrs, L2s, 
                          dataset, samples, targets, 
                          groups, k):
    
    #best_val_loss = float('inf')
    best_params = {'val_loss': float('inf'),
                   'val_acc': 0}
    set_seed(seed)

############################################################################
    #samples, targets, groups, dataset = shuffle(samples, targets, groups, dataset)
############################################################################
    outer_kf = StratifiedGroupKFold(n_splits=k, shuffle=False)
    print('Pilot training begins')
    for batch_size in batch_sizes:
        for lr in lrs:
            for l2 in L2s:
                print(f'Batch size: {batch_size}, Learning rate: {lr}, Weight decay: {l2}')
                
                v_p_all, v_l_all = [], []
                for fold, (train_val_idx, test_idx) in enumerate(
                        outer_kf.split(samples, targets, groups=groups)
                ):
                    
                    test_overlaps(fold,
                                groups[train_val_idx],
                                groups[test_idx])

                    # === check overlaps on subject IDs ===
                    inner_kf = StratifiedGroupKFold(n_splits=int(k)-1, shuffle=False)
                    train_idx, val_idx = next(
                    inner_kf.split(samples[train_val_idx],
                                    targets[train_val_idx],
                                    groups=groups[train_val_idx])
                    )
                    
                    train_idx, val_idx = train_val_idx[train_idx], train_val_idx[val_idx]
                    
                    print(f'Fold {fold+1}: {len(train_idx)} train / {len(val_idx)} validation samples')
                    print('K-fold overlaps check')
                    test_overlaps(fold,
                                groups[train_idx],
                                groups[val_idx])

                    # === Unique subjects per split ===   
                    train_subidx = subsamples(seed, train_idx, groups)
                    #val_subidx = subsamples(seed, val_idx, groups)
                    # === build Subsets (no new copies!) ===
                    # if task == 'FTD vs AD':
                    #     train_subset = Subset(dataset, train_idx)
                    # else:
                    train_subset = Subset(dataset, train_subidx)
                    val_subset   = Subset(dataset, val_idx)
                    train_loader = DataLoader(train_subset,
                                            batch_size=batch_size,
                                            shuffle=True,  drop_last=True)
                    val_loader   = DataLoader(val_subset,
                                            batch_size=64,
                                            shuffle=True, drop_last=False)


                    # === model, optimizer, loss ===
                    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    if fold == 0:
                        print("cuda available:", torch.cuda.is_available())
                        print("device count:", torch.cuda.device_count())
                        if torch.cuda.is_available():
                            print("name:", torch.cuda.get_device_name(0))
                        print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
                    
                    model    = model_class(**model_kwargs).to(device)
                    
                    if model_name == "MBSzEEGNet":
                        optimizer= SGD(model.parameters(),
                                      lr=lr,
                                      weight_decay=l2
                        )
                    else:
                        optimizer= AdamW(model.parameters(),
                                        lr=lr,
                                        betas=(0.5, 0.999),
                                        weight_decay=l2
                        ) 
                    
                    if model_name == 'MSVTNet':
                        criterion = JointCrossEntoryLoss()
                    else:
                        criterion = CrossEntropyLoss()

                    val_loss_folds = []
                    val_acc_fold = []
                    # --- training epochs ---
                    for epoch in range(30):
                        model.train()
                        train_loss = 0.0
                        all_tr_p, all_tr_l = [], []
                        for xb, yb in train_loader:
                            xb, yb = xb.to(device), yb.to(device)
                            optimizer.zero_grad()
                            out = model(xb)
                            if model_name == 'MBSzEEGNet':
                                loss = criterion(out, yb.long())
                            else:
                                loss = criterion(out, yb.long())
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item() * yb.size(0)
                            if model_name == 'MSVTNet':
                                preds = out[0].argmax(dim=1)
                            else:    
                                preds = out.argmax(dim=1)
                            all_tr_p.extend(preds.cpu().numpy())
                            all_tr_l.extend(yb.cpu().numpy())

                        train_loss /= len(train_subset)
                        train_acc  = accuracy_score(all_tr_l, all_tr_p)
                        
                        if (epoch + 1) % 10 == 0:
                                # Print epoch metrics
                            print(f"Epoch {epoch + 1}: "
                                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                                )

                        # --- validation ---
                    model.eval()
                    val_loss, v_p, v_l = 0.0, [], []
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            xb, yb = xb.to(device), yb.to(device)
                            out = model(xb)
                            if model_name == 'MBSzEEGNet':
                                loss = criterion(out, yb.long())
                            else:
                                loss = criterion(out, yb.long())
                            val_loss += loss.item() * yb.size(0)
                            if model_name == 'MSVTNet':
                                preds = out[0].argmax(dim=1)    
                            else:
                                preds = out.argmax(dim=1)
                            v_p.extend(preds.cpu().numpy())
                            v_l.extend(yb.cpu().numpy())
                            v_p_all.extend(preds.cpu().numpy())
                            v_l_all.extend(yb.cpu().numpy())
                            
                            

                    val_loss /= len(val_subset)
                    val_loss_folds.append(val_loss)
                    val_acc = accuracy_score(v_l, v_p)
                    #val_acc_fold.append(val_acc)
                    print(f'Validation Loss: {val_loss}, Validation Accuracy {val_acc}')
                
                val_loss_av = np.mean(val_loss_folds)
                #val_acc_final = np.mean(val_acc_fold)
                val_acc_seg = accuracy_score(v_l_all, v_p_all)
                
                print(f"Total Validation Accuracy (seg.): {val_acc_seg}")

                     
                #if val_loss_final < best_params['val_loss']:
                if val_acc_seg > best_params['val_acc']:
                    best_params['model'] = model_name
                    best_params['val_loss'] = val_loss_av
                    best_params['val_acc'] = val_acc_seg
                    best_params['batch size'] = batch_size 
                    best_params['learning rate'] = lr
                    best_params['L2 weight decay'] = l2
    
    print('The best parameters have been found!')
    print(f"Model: {model_name} \nLoss: {best_params['val_loss']} \nBatch size: {best_params['batch size']} \nLearning rate: {best_params['learning rate']} \nL2: {best_params['L2 weight decay']}")
    return best_params
            
        
        
        
        
        
        
        
        
        
    