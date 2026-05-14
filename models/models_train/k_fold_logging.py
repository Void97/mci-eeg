import numpy as np
from collections import Counter

def count_fold_subjects(groups, train_val_idx, 
                        train_idx, val_idx,
                        test_idx, subject_to_label,
                        fold):
    
    def count_subjects_per_class(subjects, subject_to_label):
        labels = [subject_to_label[s] for s in subjects]
        return dict(Counter(labels))     
        
    def keys_to_int(d):
        return {int(k): v for k,v in d.items()}
    
    train_subjects = np.unique(groups[train_val_idx][train_idx])
    val_subjects = np.unique(groups[train_val_idx][val_idx])
    test_subjects = np.unique(groups[test_idx])

    c_train_s = count_subjects_per_class(train_subjects, subject_to_label)
    c_val_s = count_subjects_per_class(val_subjects, subject_to_label)
    c_test_s = count_subjects_per_class(test_subjects, subject_to_label)

    print(f"  Train: {len(train_subjects)} subjects, per class: {c_train_s}")
    print(f"  Val:   {len(val_subjects)} subjects, per class: {c_val_s}")
    print(f"  Test:  {len(test_subjects)} subjects, per class: {c_test_s}")

    one_fold_subject_log = {
            'fold': fold + 1,
            'n_train_subjects': int(len(train_subjects)),
            'n_val_subjects': int(len(val_subjects)),
            'n_test_subjects': int(len(test_subjects)),
            'train_class_counts': keys_to_int(c_train_s),
            'val_class_counts': keys_to_int(c_val_s),
            'test_class_counts': keys_to_int(c_test_s),
            'train_subjects': train_subjects.tolist(),
            'val_subjects': val_subjects.tolist(),
            'test_subjects': test_subjects.tolist()
        }

    return one_fold_subject_log