"""Tests for utils.dataset_registry / main_func.build()+labels_mapping().

build() itself calls each dataset's metadata_loader, which reads real
files off disk -- that belongs in the slow/integration tier (see
tests/integration/test_full_pipeline_smoke.py), not here. What's tested
here is everything about the registry's *structure* that doesn't require
touching disk: it fails before ever reaching a metadata_loader for an
unknown dataset name, since DATASETS[dataset_name] raises first.
"""
import pytest

from utils.dataset_registry import DATASETS
from main_func import build, labels_mapping


def test_every_dataset_ch_num_matches_ch_names_length():
    for name, spec in DATASETS.items():
        assert spec.ch_num == len(spec.ch_names), name


def test_every_dataset_has_tasks_and_label_maps():
    for name, spec in DATASETS.items():
        assert len(spec.tasks) > 0, name
        assert len(spec.label_maps) > 0, name
        # every returned task must have a label map -- otherwise labels_mapping()
        # would raise for a task build() claims this dataset supports
        for task in spec.tasks:
            assert task in spec.label_maps, f"{name}: task {task!r} has no label map"


def test_3class_label_map_exists_but_is_not_in_tasks():
    """Documents a deliberate asymmetry: CAUEEG and ADvsFTDvsHC support a
    '3-class' task via labels_mapping(), but build()'s `tasks` list (what
    main.py actually iterates per dataset) doesn't include it -- so it's
    reachable but not run automatically. This test exists so nobody
    "fixes" this asymmetry by accident without realizing it's on purpose."""
    for name in ('CAUEEG', 'ADvsFTDvsHC'):
        spec = DATASETS[name]
        assert '3-class' in spec.label_maps
        assert '3-class' not in spec.tasks


def test_build_unknown_dataset_raises_before_touching_disk():
    with pytest.raises(KeyError):
        build('NotARealDataset')


def test_labels_mapping_unknown_dataset_raises():
    with pytest.raises(KeyError):
        labels_mapping('NotARealDataset', 'some task')


@pytest.mark.parametrize("dataset_name", list(DATASETS.keys()))
def test_labels_mapping_unknown_task_raises(dataset_name):
    with pytest.raises(KeyError):
        labels_mapping(dataset_name, 'not a real task')


@pytest.mark.parametrize("dataset_name", list(DATASETS.keys()))
def test_labels_mapping_known_tasks_return_binary_or_3class_maps(dataset_name):
    spec = DATASETS[dataset_name]
    for task, expected_map in spec.label_maps.items():
        assert labels_mapping(dataset_name, task) == expected_map