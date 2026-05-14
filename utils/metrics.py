from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib as plt
import numpy as np
from collections import Counter, defaultdict


def metrics(labels, preds, subjects, model_name, num_classes, task):

    def majority_vote(labels, preds, subjects):
        """
        Perform majority voting over predictions for each subject.

        Args:
            labels   (list/np.array): True label for each sample (epoch).
            preds    (list/np.array): Predicted label for each sample.
            subjects (list/np.array): Subject ID for each sample.

        Returns:
            voted_labels (np.array): True label per subject (assumes all epochs for a subject have the same label).
            voted_preds  (np.array): Majority-voted predicted label per subject.
            subject_ids  (np.array): Subject IDs (order matches voted_labels and voted_preds).
        """
        subject_true = defaultdict(list)
        subject_pred = defaultdict(list)

        # Group all predictions and labels by subject
        for y_true, y_pred, subj in zip(labels, preds, subjects):
            subject_true[subj].append(y_true)
            subject_pred[subj].append(y_pred)

        voted_labels = []
        voted_preds = []
        subject_ids = []

        for subj in subject_true.keys():
            subject_ids.append(subj)
            # All epochs for a subject should have the same ground truth
            voted_labels.append(subject_true[subj][0])
            # Majority vote on predictions
            most_common_pred = Counter(subject_pred[subj]).most_common(1)[0][0]
            voted_preds.append(most_common_pred)

        return np.array(voted_labels), np.array(voted_preds), np.array(subject_ids)

    def metrics_binary(labels, preds):
            
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print(f'Overall accuracy: {accuracy:.4f}')
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f'Overall precision: {precision:.4f}')
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f'Overall sensitivity: {sensitivity:.4f}')
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f'Overall specificity: {specificity:.4f}')
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        print(f'Overall f1-score: {f1:.4f}')
            
        return accuracy, precision, sensitivity, specificity, f1
    
    def metrics_multiclass():
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='weighted', zero_division=0)
        recall = recall_score(labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)

        print(f'Overall accuracy: {accuracy:.4f}')
        print(f'Weighted precision: {precision:.4f}')
        print(f'Weighted recall: {recall:.4f}')
        print(f'Weighted f1-score: {f1:.4f}')

        return accuracy, precision, recall, f1
    
    
    labels_sub_wise, preds_sub_wise, _ = majority_vote(labels, preds, subjects)

    if num_classes == 2:

                accuracy, precision, sensitivity, specificity, f1 = metrics_binary(labels, preds)
                accuracy_sw, precision_sw, sensitivity_sw, specificity_sw, f1_sw = metrics_binary(labels_sub_wise, preds_sub_wise)

                result = {
                "Task": task,
                "Model": model_name,
                "Accuracy": round(accuracy, 4),
                "Accuracy (subject ind)": round(accuracy_sw, 4),
                "Precision": round(precision, 4),
                "Precision (subject ind)": round(precision_sw, 4),
                "Sensitivity": round(sensitivity, 4),
                "Sensitivity (subject ind)": round(sensitivity_sw, 4),
                "Specificity": round(specificity, 4),
                "Specificity (subject ind)": round(specificity_sw, 4),
                "F1 Score": round(f1, 4),
                "F1 Score (subject ind)": round(f1_sw, 4)
                }

    else: 
                accuracy, precision, recall, f1 = metrics_multiclass(labels, preds)

                result = {
                "Task": task,
                "Model": model_name,
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Sensitivity": round(recall, 4),
                "Specificity": 'None',
                "F1 Score": round(f1, 4)
                }

    return result

def timing_logs(model_name, task, 
                k, train_time,
                savedir):
    
    log = {
        'model_name': model_name,
        'tasks': task,
        
    }