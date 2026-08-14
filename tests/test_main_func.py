"""Tests for main_func.py's non-dataset-registry helpers: seeding,
filtering, and subject-wise metric computation.

Note: subject_wise_acc() is dead code (zero callers anywhere in the
codebase) and isn't tested here for that reason -- not an oversight.
"""
import numpy as np
import pytest
import torch

from main_func import filter_data, generate_seeds, set_seed, subject_wise_metrics


def test_set_seed_makes_torch_numpy_random_reproducible():
    set_seed(123)
    a = (torch.randn(5), np.random.rand(5))
    set_seed(123)
    b = (torch.randn(5), np.random.rand(5))

    assert torch.equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_filter_data_keeps_only_labels_in_the_map():
    subjects = ['s1', 's2', 's3', 's4']
    labels = ['A', 'B', 'A', 'C']
    labels_map = {'A': 0, 'B': 1}  # 'C' is not in the map

    filtered_subjects, filtered_labels = filter_data(subjects, labels, labels_map)

    assert filtered_subjects == ['s1', 's2', 's3']
    assert filtered_labels == ['A', 'B', 'A']


def test_filter_data_preserves_pairing_order():
    subjects = ['a', 'b', 'c']
    labels = ['X', 'Y', 'X']
    filtered_subjects, filtered_labels = filter_data(subjects, labels, {'X': 0})
    assert list(zip(filtered_subjects, filtered_labels)) == [('a', 'X'), ('c', 'X')]


def test_subject_wise_metrics_majority_vote_and_confusion_matrix():
    # 3 subjects: sub 1 all correct (label 0), sub 2 all correct (label 1),
    # sub 3 majority-vote wrong (true label 0, majority predicted 1).
    labels =   [0, 0, 0,  1, 1, 1,  0, 0, 0]
    preds =    [0, 0, 0,  1, 1, 1,  1, 1, 0]
    subjects = [1, 1, 1,  2, 2, 2,  3, 3, 3]

    acc, sens, spec, prec, f1 = subject_wise_metrics(labels, preds, subjects)

    # subject-level: true=[0,1,0], pred=[0,1,1] -> tp=1, tn=1, fp=1, fn=0
    assert acc == pytest.approx(2 / 3)
    assert sens == pytest.approx(1.0)     # tp / (tp+fn) = 1/1
    assert spec == pytest.approx(0.5)     # tn / (tn+fp) = 1/2
    assert prec == pytest.approx(0.5)     # tp / (tp+fp) = 1/2
    assert f1 == pytest.approx(2 * 0.5 * 1.0 / (0.5 + 1.0))


def test_generate_seeds_clustering_mode_returns_fixed_seeds():
    seeds = generate_seeds('clustering', n_repeats=3, dataset_name='X', task='Y')
    assert seeds == [42, 123, 2024, 5678, 91011]


def test_generate_seeds_benchmark_mode_is_deterministic_and_cached(tmp_path, monkeypatch):
    """benchmark mode writes to ./results/seeds_<dataset>_<task>.txt (a
    relative path) -- monkeypatch cwd to a scratch dir so this never
    touches the real results/ folder."""
    monkeypatch.chdir(tmp_path)

    seeds_first_call = generate_seeds('benchmark', n_repeats=5, dataset_name='TestDS', task='TestTask')
    assert len(seeds_first_call) == 5

    seeds_file = tmp_path / 'results' / 'seeds_TestDS_TestTask.txt'
    assert seeds_file.exists()

    # Second call should reuse the same seeds from disk, not regenerate new ones.
    seeds_second_call = generate_seeds('benchmark', n_repeats=5, dataset_name='TestDS', task='TestTask')
    assert seeds_second_call == seeds_first_call