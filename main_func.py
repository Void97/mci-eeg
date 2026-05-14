import numpy as np
import pandas as pd
import random
import torch 
import json
import os

from collections import Counter
from sklearn.metrics import confusion_matrix

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def generate_seeds(n_repeats, dataset_name, task):
    
    seeds_file = f"./results/seeds_{dataset_name}_{task}.txt"
    if os.path.exists(seeds_file):
        print("The random seeds already exist!\n"
              f"{seeds_file} is used")
        with open(seeds_file, "r") as f:
            random_seeds = [int(line.strip()) for line in f.readlines()]
    else:
        random_seeds = []
        for _ in range(n_repeats):
            seed = random.randint(0, 2**32-1)
            random_seeds.append(seed)
        
        os.makedirs(os.path.dirname(seeds_file), exist_ok=True)
        with open(seeds_file, "w") as f:
            for seed in random_seeds:
                f.write(str(seed) + "\n")
        print(f"Seeds are generated and saved to {seeds_file}")
        
    return random_seeds

def labels_mapping(dataset_name, task):
    
    if dataset_name == 'GENEEG':
        labels_map = {'Normal': 0, 'MCI': 1}
    elif dataset_name == 'MCIvsHC':
        labels_map = {'NORMAL': 0, 'MCI': 1}
    
    if dataset_name == 'CAUEEG':
        if task == 'Dementia vs Normal':
            labels_map = {'Normal': 0, 'Dementia': 1}
        elif task == 'MCI vs Normal':
            labels_map = {'Normal': 0, 'MCI': 1}
        elif task == 'MCI vs Dementia':
            labels_map = {'MCI': 0, 'Dementia': 1}
        elif task == '3-class':
            labels_map = {'Normal': 0, 'MCI': 1, 'Dementia': 2}
    
    elif dataset_name == 'ADvsFTDvsHC':
        if task == 'AD vs HC':
            labels_map = {'C': 0, 'A': 1}
        elif task == 'FTD vs HC':
            labels_map = {'C': 0, 'F': 1}
        elif task == 'FTD vs AD':
            labels_map = {'F': 0, 'A': 1}
        elif task == '3-class':
            labels_map = {'C': 0, 'F': 1, 'A': 2}
    
    return labels_map    

def filter_data(subjects, labels, labels_map):
    
    filtered_subjects, filtered_labels = [], []
    for sub, lab in zip(subjects, labels):
        if lab in labels_map:
            filtered_subjects.append(sub)
            filtered_labels.append(lab)
    
    return filtered_subjects, filtered_labels

def save_labels_and_predictions(dir, dataset, task, iter, model_name, preds, labels):
    if not os.path.exists(dir):
        os.makedirs(dir)
    predictions_file = os.path.join(dir, f"{dataset}_{task}_{model_name}_iteration_{iter}_predictions.txt")
    with open(predictions_file, 'w') as p:
        for pred in preds:
            p.write(f"{pred}\n")
    labels_file = os.path.join(dir, f"{dataset}_{task}_{model_name}_iteration_{iter}_labels.txt")
    with open(labels_file, 'w') as l:
        for label in labels:
            l.write(f"{label}\n") 
            
def subject_wise_acc(labels, preds, subjects):
    
    labels = np.array(labels)
    preds = np.array(preds)
    subjects = np.array(subjects)
    
    subject_ids = np.unique(subjects)
    subject_accs = {}
    #subject_majority_preds = {}
     
    for id in subject_ids:
        mask = subjects == id
        subj_labels = labels[mask]
        subj_preds = preds[mask]
         
        majority_pred = Counter(subj_preds).most_common(1)[0][0]
        true_label = Counter(subj_labels).most_common(1)[0][0]
         
        correct = 1 if majority_pred == true_label else 0
        subject_accs[str(id)] = correct
        #subject_majority_preds[str(id)] = int(majority_pred)
    subject_acc = np.mean(list(subject_accs.values()))
    return subject_acc

def subject_wise_metrics(labels, preds, subjects):
    """
    Compute subject-level precision, sensitivity, specificity, and F1-score
    based on majority-vote predictions for each subject.
    """
    labels = np.array(labels)
    preds = np.array(preds)
    subjects = np.array(subjects)

    subject_ids = np.unique(subjects)
    subj_true = []
    subj_pred = []

    for sid in subject_ids:
        mask = subjects == sid
        subj_labels = labels[mask]
        subj_preds = preds[mask]

        # Majority vote per subject
        majority_pred = Counter(subj_preds).most_common(1)[0][0]
        true_label = Counter(subj_labels).most_common(1)[0][0]

        subj_true.append(true_label)
        subj_pred.append(majority_pred)

    # Confusion matrix at subject level
    tn, fp, fn, tp = confusion_matrix(subj_true, subj_pred).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0

    return sens, spec, prec, f1    

def av_metrics(acc_all, subject_acc_all, prec_all, subject_prec_all, sens_all, subject_sens_all, spec_all, subject_spec_all, f1_all, subject_f1_all):
    
    acc_av = np.mean(acc_all)
    subject_acc_av = np.mean(subject_acc_all)
    prec_av = np.mean(prec_all)
    sens_av = np.mean(sens_all)
    spec_av = np.mean(spec_all)
    f1_av = np.mean(f1_all)
    
    subject_prec_av = np.mean(subject_prec_all)
    subject_sens_av = np.mean(subject_sens_all)
    subject_spec_av = np.mean(subject_spec_all)
    subject_f1_av = np.mean(subject_f1_all)
    
    return acc_av, subject_acc_av, prec_av, subject_prec_av, sens_av, subject_sens_av, spec_av, subject_spec_av, f1_av, subject_f1_av

def std_metrics(acc_all, subject_acc_all, prec_all, subject_prec_all, sens_all, subject_sens_all, spec_all, subject_spec_all, f1_all, subject_f1_all):
    acc_std = np.std(acc_all)
    subject_acc_std = np.std(subject_acc_all)
    prec_std = np.std(prec_all)
    sens_std = np.std(sens_all)
    spec_std = np.std(spec_all)
    f1_std = np.std(f1_all)
    
    subject_prec_std = np.std(subject_prec_all)
    subject_sens_std = np.std(subject_sens_all)
    subject_spec_std = np.std(subject_spec_all)
    subject_f1_std = np.std(subject_f1_all)
    
    return acc_std, subject_acc_std, prec_std, subject_prec_std, sens_std, subject_sens_std, spec_std, subject_spec_std, f1_std, subject_f1_std

def save_metrics_logs(dir, dataset, task, model_name,
                      acc_av, acc_std, subject_acc_av, subject_acc_std,
                      prec_av, prec_std, subject_prec_av, subject_prec_std,
                      sens_av, sens_std, subject_sens_av, subject_sens_std,
                      spec_av, f1_av, subject_spec_av, subject_spec_std, 
                      spec_std, f1_std, subject_f1_av, subject_f1_std):
    
    metrics_logs = {
    "Overall_accuracy (seg.)": f'{acc_av} ± {acc_std}',
    "Overall_accuracy (subj.)": f'{subject_acc_av} ± {subject_acc_std}',
    "Overall_precision (seg.)": f'{prec_av} ± {prec_std}',
    "Overall_precision (subj.)": f'{subject_prec_av} ± {subject_prec_std}',
    "Overall_sensitivity (seg.)": f'{sens_av} ± {sens_std}',
    "Overall_sensitivity (subj.)": f'{subject_sens_av} ± {subject_sens_std}',
    "Overall_specificity (seg.)": f'{spec_av} ± {spec_std}',
    "Overall_specificity (subj.)": f'{subject_spec_av} ± {subject_spec_std}',
    "Overall_f1 (seg.)": f'{f1_av} ± {f1_std}',
    "Overall_f1 (subj.)": f'{subject_f1_av} ± {subject_f1_std}'
    }
    
    if not os.path.exists(dir):
        os.makedirs(dir)
    metrics_logs_file = os.path.join(dir, f"{dataset}_{task}_{model_name}_overall_metrics.json")
    with open(metrics_logs_file, 'w') as f:
        json.dump(metrics_logs, f, indent=2)
    print(f"Overall metrics of {model_name} were saved!")

def build(dataset_name):
    
    if dataset_name == 'GENEEG':
        dataset_path = './datasets/raw/GENEEG/all_data'
        preprocessed_dir = './datasets/preprocessed/GENEEG_preprocessed'
        metadata = pd.read_excel(os.path.join(dataset_path, 'metadata.xlsx'))
        
        ch_num = 17
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'C3', 'C4', 'P3', 'P4', 
            'O1', 'O2', 'T3', 'T4', 'Fz', 'Cz', 'Pz']
        show_channel = ch_names
        
        tasks = ['MCI vs HC']
        subjects_list, labels_list = metadata['id'], metadata['status']
    
    elif dataset_name == 'MCIvsHC':
        annotation_path = './datasets/raw/MCIvsHC/data_extended/states_2.xlsx'
        dataset_path = './datasets/raw/MCIvsHC/data_extended/'
        preprocessed_dir = './datasets/raw/MCIvsHC_preprocessed'
        metadata = pd.read_excel(os.path.join(annotation_path))
        
        ch_num = 19
        ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
                    'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
        show_channel = ch_names
        
        tasks = ['MCI vs HC']
        subjects_list, labels_list = metadata['file number'], metadata['status']
        
    elif dataset_name == 'ADvsFTDvsHC':
        annotation_path = 'datasets/raw/ADvsFTDvsHC/data/dataset_description.json'
        dataset_path = 'datasets/raw/ADvsFTDvsHC/data/'
        preprocessed_dir = 'datasets/preprocessed/ADvsFTDvsHC_preprocessed/'
        metadata = pd.read_csv(os.path.join(dataset_path, 'participants.tsv') ,sep = '\t')
        
        ch_num = 19
        ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
                    'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
        show_channel = ch_names
        
        #tasks = ['AD vs HC', 'FTD vs HC', 'FTD vs AD']
        tasks = ['AD vs HC', 'FTD vs HC', 'FTD vs AD']
        subjects_list, labels_list = metadata['participant_id'], metadata['Group']
        
    elif dataset_name == 'CAUEEG':
        annotation_path = './datasets/raw/CAUEEG/dementia-no-overlap.json'
        #dataset_path = './datasets/raw/CAUEEG/signal/edf/'
        dataset_path = './datasets/raw/CAUEEG/filtered_signal/'
        preprocessed_dir = './datasets/preprocessed/CAUEEG-preprocessed/'
        
        with open(annotation_path, 'r') as file:
            annotation = json.load(file)
        subjects_list = []
        labels_list = []
        for entry in annotation['train_split']:
            if 'normal' in entry["symptom"]:
                subjects_list.append(entry['serial'])
                labels_list.append('Normal')
            elif 'mci' in entry['symptom']:
                subjects_list.append(entry['serial'])
                labels_list.append('MCI')
            elif 'dementia' in entry['symptom']:
                subjects_list.append(entry['serial'])
                labels_list.append('Dementia')
        for entry in annotation['test_split']:
            if 'normal' in entry["symptom"]:
                subjects_list.append(entry['serial'])
                labels_list.append('Normal')
            elif 'mci' in entry['symptom']:
                subjects_list.append(entry['serial'])
                labels_list.append('MCI')
            elif 'dementia' in entry['symptom']:
                subjects_list.append(entry['serial'])
                labels_list.append('Dementia')

        table = list(zip(subjects_list, labels_list))
        metadata = pd.DataFrame(table, columns=['id', 'label'])
        print(metadata)
        ch_num = 19
        ch_names = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F4', 'C4', 'P4', 'O2', 
                    'F7', 'T3', 'T5', 'F8', 'T4', 'T6', 'Fz', 'Cz', 'Pz']
        show_channel = ch_names
        
        #tasks = ['Dementia vs Normal', 'MCI vs Normal', 'MCI vs Dementia']
        tasks = ['Dementia vs Normal']
        subjects_list, labels_list = metadata['id'], metadata['label']

    return dataset_path, preprocessed_dir, ch_num, ch_names, show_channel, tasks, metadata, subjects_list, labels_list

def makedirs(dataset_name):
    
    dirs = {
    'k_fold_dist_logs_dir': f'./results/k-fold dist logs/{dataset_name}',
    'best_params_logs_dir': f'./results/best params logs/{dataset_name}',
    'weights_dir': f'./results/weights/{dataset_name}',
    'predictions_dir': f'./results/predictions/{dataset_name}',
    'metrics_dir': f'./results/metrics/{dataset_name}',
    'metrics_logs_dir': f'./results/metrics/{dataset_name}/logs',
    'tsne_dir': f'./results/tsne plots/{dataset_name}',
    'topomaps_dir': f'./results/topomaps/{dataset_name}',
    'training_inference_time_logs_dir': f'./results/training and inference time logs/{dataset_name}'
    }

    for dir_name, path in dirs.items():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")
            
    return dirs