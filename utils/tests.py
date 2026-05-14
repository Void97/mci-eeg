def test_overlaps(fold, train_subjects, test_subjects):
    
    overlap = set(train_subjects).intersection(set(test_subjects))
    if overlap:
        print(f'Subject overlap detected in fold {fold + 1}: {overlap}')
    else:
        print(f'No overlaps in fold {fold + 1}')