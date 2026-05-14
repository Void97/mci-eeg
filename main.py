import json
import os
import captum
import numpy as np
import pandas as pd
import torch
import random

from utils.EEG_preprocess import preprocess, load_preprocessed
from models.models_base.models_list import models_list
#from models.models_train.train_func import train_10_K_fold, train_10_K_fold_pilot
from models.models_train.train_func import train_10_K_fold, train_10_K_fold_pilot
from captum.attr import Saliency
from collections import Counter
from torch.utils.data import TensorDataset


from main_func import *
from utils.interpretation import *


def main():
    
    def collect_metrics(labels, preds, subjects, dataset, task, model_name, iter, dir):
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        acc_all.append(acc)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        prec_all.append(prec)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        sens_all.append(sens)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        spec_all.append(spec)
        f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0
        f1_all.append(f1)  
        
        subject_acc = subject_wise_acc(labels, preds, subjects)
        subject_acc_all.append(subject_acc)
        
        subject_sens, subject_spec, subject_prec, subject_f1 = subject_wise_metrics(labels, preds, subjects)
        subject_acc_all.append(subject_acc)
        subject_sens_all.append(subject_sens)
        subject_spec_all.append(subject_spec)
        subject_prec_all.append(subject_prec)
        subject_f1_all.append(subject_f1)
        
        metrics_logs = {
            "Task": task,
            "Model": model['name'],
            "Accuracy (seg.)": round(acc, 4),
            "Accuracy (subj.)": round(subject_acc, 4),
            "Precision (seg.)": round(prec, 4),
            "Precision (subj.)": round(subject_prec, 4),
            "Sensitivity (seg.)": round(sens, 4),
            "Sensitivity (subj.)": round(subject_sens, 4),
            "Specificity (seg.)": round(spec, 4),
            "Specificity (subj.)": round(subject_spec, 4),
            "F1 Score (seg.)": round(f1, 4),
            "F1 Score (subj.)": round(subject_f1, 4),
        } 
        
        if not os.path.exists(dir):
            os.makedirs(dir)
        metrics_logs_file = os.path.join(dir, f"{dataset}_{task}_{model_name}_iteration_{iter}_metrics.json")
        with open(metrics_logs_file, 'w') as f:
            json.dump(metrics_logs, f, indent=2)
        print(f"Metrics of {model_name} of the iteration {iter} were saved!")
        
        return acc_all, subject_acc_all, prec_all, subject_prec_all, sens_all, subject_sens_all, spec_all, subject_spec_all, f1_all, subject_f1_all

    
    sfreq = 200
    dataset_name = 'ADvsFTDvsHC'
    #dataset_name = 'GENEEG'
    need_preprocessing = False
    
    results_seg = []
    results_subj = []
    freq_bands = ['all', 'delta', 'theta', 'alpha', 'beta', 'gamma']
    k = 5
    n_repeats = 15
    
    batch_sizes = [16, 32, 64]
    learning_rates = [0.001, 0.005, 0.0001, 0.0005, 0.00001]
    L2_weight_decays = [0, 0.01, 0.001, 0.0001]
    
    #random_seeds = generate_seeds(n_repeats, dataset_name, task)
    dataset_path, preprocessed_dir, ch_num, ch_names, show_channel, tasks, metadata, subjects_list, labels_list = build(dataset_name)
    dirs = makedirs(dataset_name)
    
    if need_preprocessing:
        if dataset_name == 'GENEEG':
            #sfreq = 200
            ch_types = ['eeg'] * len(ch_names)
            ch_num = len(ch_names)
            
            preprocess(metadata, dataset_path, preprocessed_dir,
                    dataset_name, sfreq, 
                    ch_names, ch_types)
        elif dataset_name == 'MCIvsHC' or dataset_name == 'ADvsFTDvsHC':
            preprocess(metadata, dataset_path, preprocessed_dir,
                    dataset_name, new_sfreq=200)
        else:
            preprocess(metadata, dataset_path, preprocessed_dir,
                    dataset_name)
    
    for task in tasks:
    
        random_seeds = generate_seeds(n_repeats, dataset_name, task)
        
        labels_map = labels_mapping(dataset_name, task)
        filtered_subjects, filtered_labels = filter_data(subjects_list, labels_list, labels_map)
        #print(labels_map)
        
        label_dict = dict(zip(filtered_subjects, filtered_labels))
        samples, targets, groups = load_preprocessed(preprocessed_dir, labels_map, label_dict)
        models = models_list(num_classes=2, num_channels=ch_num, time_points=samples.shape[2])
    #for i in range(n_repeats):
        
        for model in models:
        #for task in tasks:    
            acc_all = []
            subject_acc_all = []
            prec_all = []
            sens_all = []
            spec_all = []
            f1_all = []
            subject_prec_all = []
            subject_sens_all = []
            subject_spec_all = []
            subject_f1_all = []
            #all_gradients = {}  
            
            
            for i in range(n_repeats):
            #for model in models: 
                
                seed = random_seeds[i]
                metrics_exist_file = os.path.join(dirs['metrics_logs_dir'], f"{dataset_name}_{task}_{model['name']}_iteration_{i}_metrics.json")
                if os.path.exists(metrics_exist_file):
                    
                    print(f"{dataset_name}_{task}_{model['name']}_iteration_{i} metrics exist! No need to train.")
                    with open(metrics_exist_file, 'r') as f:
                        metrics = json.load(f)
                    
                    acc, subject_acc, prec, subject_prec, sens, subject_sens, spec, subject_spec, f1, subject_f1 = round(metrics['Accuracy (seg.)'], 4), round(metrics['Accuracy (subj.)'], 4), round(metrics['Precision (seg.)'], 4), round(metrics['Precision (subj.)'], 4), round(metrics['Sensitivity (seg.)'], 4), round(metrics['Sensitivity (subj.)'], 4), round(metrics['Specificity (seg.)'], 4), round(metrics['Specificity (subj.)'], 4), round(metrics['F1 Score (seg.)'], 4), round(metrics['F1 Score (subj.)'], 4)
                    acc_all.append(acc)
                    subject_acc_all.append(subject_acc)
                    prec_all.append(prec)
                    subject_prec_all.append(subject_prec)
                    sens_all.append(sens)
                    subject_sens_all.append(subject_sens)
                    spec_all.append(spec)
                    subject_spec_all.append(subject_spec)
                    f1_all.append(f1)   
                    subject_f1_all.append(subject_f1)                                              
                                    
                else:
                    print(f"{dataset_name}_{task}_{model['name']}_iteration_{i+1} metrics doesn't exist. Let's train!")

                    if model['name'] == 'Oh_CNN' or model['name'] == 'SzHNN' or model['name'] == 'EEG_Deformer':
                        full_samples = torch.from_numpy(samples).float()
                    else:
                        full_samples = torch.from_numpy(samples).unsqueeze(1).float()   # one big tensor
                        
                    kwargs = model['kwargs'].copy()
                    kwargs['num_classes'] = 3 if task == '3-class' else 2
                    
                    full_targets = torch.from_numpy(targets).float()    # one big tensor
                    full_dataset = TensorDataset(full_samples, full_targets)
                
                    ###############################################################################################################################
                
                    params_file_path = os.path.join(dirs['best_params_logs_dir'], f"{dataset_name}_{task}_{model['name']}_iteration_{i}_best_params_file.json")
                    
                    if os.path.exists(params_file_path):
                        print(f"Best parameters file for the iteration {i+1} of {task} and {model['name']} exists! No need the pilot training.")
                        best_params_file = os.path.join(dirs['best_params_logs_dir'], f"{dataset_name}_{task}_{model['name']}_iteration_{i}_best_params_file.json")
                        with open(best_params_file, 'r') as f:
                            best_params = json.load(f)
                    
                        print(f"Task: {task}. Model: {model['name']}. Iteration: {i + 1}")
                        labels, preds, preds_roc_auc, subjects, gradient = train_10_K_fold(
                                seed,
                                dirs,
                                dataset_name,
                                task,
                                model['class'],
                                model['name'],
                                kwargs,
                                best_params,
                                full_dataset,
                                full_samples.numpy(),
                                full_targets.numpy(),
                                groups,
                                k,
                                i
                            )

                        save_labels_and_predictions(dirs['predictions_dir'], dataset_name, i, task, model['name'], preds, labels)
                        acc_all, subject_acc_all, prec_all, subject_prec_all, sens_all, subject_sens_all, spec_all, subject_spec_all, f1_all, subject_f1_all = collect_metrics(labels, preds, groups, dataset_name, task, model['name'], i, dirs['metrics_logs_dir'])
                        
                        # for class_id in gradient:
                        #     if gradient[class_id].size > 0:
                        #         all_gradients.setdefault(class_id, []).append(gradient[class_id])
                    
                        xai = Interpretation(dirs['topomaps_dir'], sfreq)
                        xai.plot_psd(dataset_name, model['name'], task, gradient, i, 40)
                        xai.plot_psd_topo(dataset_name, model['name'], task, gradient, freq_bands, ch_names, show_channel, i)
                        
                    
                    else:
                        print(f"Best parameters file for the iteration {i+1} of {task} and {model['name']} doesn't exists! Needs pilot training.")
                        print(f"The pilot training begins. \nTask: {task}. Model: {model['name']}. Iteration: {i+1}.")
                        best_params = train_10_K_fold_pilot(
                            seed,                            
                            model['class'],
                            model['name'],
                            kwargs,
                            batch_sizes,
                            learning_rates,
                            L2_weight_decays,
                            full_dataset,
                            full_samples.numpy(),
                            full_targets.numpy(),
                            groups,
                            k
                        )
                        
                        best_params_file = os.path.join(dirs['best_params_logs_dir'], f"{dataset_name}_{task}_{model['name']}_iteration_{i}_best_params_file.json")
                        with open(best_params_file, 'w') as f:
                            json.dump(best_params, f, indent=2)  
                        print(f'The best parameters have been saved!')
                        

                        print(f"Task: {task}. Model: {model['name']}. Iteration: {i + 1}")
                        labels, preds, preds_roc_auc, subjects, gradient = train_10_K_fold(
                                seed,
                                dirs,
                                dataset_name,
                                task,
                                model['class'],
                                model['name'],
                                kwargs,
                                best_params,
                                full_dataset,
                                full_samples.numpy(),
                                full_targets.numpy(),
                                groups,
                                k,
                                i
                            )

                        save_labels_and_predictions(dirs['predictions_dir'], dataset_name, task, i, model['name'], preds, labels)
                        acc_all, subject_acc_all, prec_all, subject_prec_all, sens_all, subject_sens_all, spec_all, subject_spec_all, f1_all, subject_f1_all = collect_metrics(labels, preds, groups, dataset_name, task, model['name'], i, dirs['metrics_logs_dir'])
                        
                        # for class_id in gradient:
                        #     if gradient[class_id].size > 0:
                        #         all_gradients.setdefault(class_id, []).append(gradient[class_id])
                        
                        xai = Interpretation(dirs['topomaps_dir'], sfreq)
                        xai.plot_psd(dataset_name, model['name'], task, gradient, i, 40)
                        xai.plot_psd_topo(dataset_name, model['name'], task, gradient, freq_bands, ch_names, show_channel, i)
                
                
                        
            acc_av, subject_acc_av, prec_av, subject_prec_av, sens_av, subject_sens_av, spec_av, subject_spec_av, f1_av, subject_f1_av = av_metrics(acc_all, subject_acc_all, prec_all, subject_prec_all, sens_all, subject_sens_all, spec_all, subject_spec_all, f1_all, subject_f1_all)
            acc_std, subject_acc_std, prec_std, subject_prec_std, sens_std, subject_sens_std, spec_std, subject_spec_std, f1_std, subject_f1_std = std_metrics(acc_all, subject_acc_all, prec_all, subject_prec_all, sens_all, subject_sens_all, spec_all, subject_spec_all, f1_all, subject_f1_all)
            print(f'Overall accuracy: {acc_av:.4f} \nOverall precision: {prec_av:.4f} \nOverall sensitivity: {sens_av:.4f} \nOverall specificity: {spec_av:.4f} \nOverall f1-score: {f1_av:.4f}')
            save_metrics_logs(dirs['metrics_logs_dir'], dataset_name, task, model['name'],
                    acc_av, acc_std, subject_acc_av, subject_acc_std,
                    prec_av, prec_std, subject_prec_av, subject_prec_std,
                    sens_av, sens_std, subject_sens_av, subject_sens_std,
                    spec_av, f1_av, subject_spec_av, subject_spec_std, 
                    spec_std, f1_std, subject_f1_av, subject_f1_std)
            
            result_seg = {
                    "Task": task,
                    "Model": model['name'],
                    "Accuracy (seg.)": f'{round(acc_av, 4)} ± {round(acc_std, 4)}',
                    "Precision (seg.)": f'{round(prec_av, 4)} ± {round(prec_std, 4)}',
                    "Sensitivity (seg.)": f'{round(sens_av, 4)} ± {round(sens_std, 4)}',
                    "Specificity (seg.)": f'{round(spec_av, 4)} ± {round(spec_std, 4)}',
                    "F1 Score (seg.)": f'{round(f1_av, 4)} ± {round(f1_std, 4)}',
                    }
            
            result_subj = {
                    "Task": task,
                    "Model": model['name'],
                    "Accuracy (subj.)": f'{round(subject_acc_av, 4)} ± {round(subject_acc_std, 4)}',
                    "Precision (subj.)": f'{round(subject_prec_av, 4)} ± {round(subject_prec_std, 4)}',
                    "Sensitivity (subj.)": f'{round(subject_sens_av, 4)} ± {round(subject_sens_std, 4)}',
                    "Specificity (subj.)": f'{round(subject_spec_av, 4)} ± {round(subject_spec_std, 4)}',
                    "F1 Score (subj.)": f'{round(subject_f1_av, 4)} ± {round(subject_f1_std, 4)}',
                    }
                
            results_seg.append(result_seg) 
            results_subj.append(result_subj)
            
    metrics_seg = pd.DataFrame(results_seg)
    metrics_subj = pd.DataFrame(results_subj)
    if not os.path.exists(dirs['metrics_dir']):
        os.makedirs(dirs['metrics_dir'])
    metrics_seg_filename = f'{dataset_name}_metrics_seg_overall_(4s, 0.5 overlaps).xlsx'
    metrics_subj_filename = f'{dataset_name}_metrics_subj_overall_(4s, 0.5 overlaps).xlsx'
    metrics_seg_path = os.path.join(dirs['metrics_dir'], metrics_seg_filename)
    metrics_subj_path = os.path.join(dirs['metrics_dir'], metrics_subj_filename)
    metrics_seg.to_excel(metrics_seg_path)
    print(f"✅ All models trained and results saved to {metrics_seg_filename}") 
    metrics_subj.to_excel(metrics_subj_path)
    print(f"✅ All models trained and results saved to {metrics_subj_filename}") 
    
    

if __name__ == '__main__':
    main()    
    
    