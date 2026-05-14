import os
import json
import re

def find_best_iteration(dataset_name, task, model_name):

    print('---------------------------------------------')
    print(f"Finding best iteration for dataset: {dataset_name}, task: {task}, model: {model_name}...")
    
    metrics_dir = f'results/metrics/{dataset_name}/logs/'
    if not os.path.exists(metrics_dir):
        print(f'Metrics directory {metrics_dir} does not exist.')
        return None
    
    key_metric = 'Accuracy (subj.)'

    best_score = -1
    #best_iteration = None

    for file in os.listdir(metrics_dir):

        if not file.endswith(".json"):
            print(f'Skipping non-JSON file: {file}')
            continue
        if task not in file:
            continue
        if model_name not in file:
            continue
        if "_iteration_" not in file:
            print(f'Skipping file without iteration info: {file}')
            continue
        
        with open(os.path.join(metrics_dir, file), 'r') as f:
            data = json.load(f)
        
        score = data[key_metric]
        #print(f'{file}: {score}')

        if score > best_score:
            best_score = score
            best_file = file

    best_iteration = int(re.search(r'iteration_(\d+)', best_file).group(1))
    print(f"Best iteration: {best_iteration} with the best score: {best_score}")
    print('---------------------------------------------')
    return best_iteration


