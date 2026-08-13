"""Registry of the datasets the benchmarking pipeline knows how to load.

Each dataset's *static* facts (paths, channel names, tasks, label maps) live
as plain fields on a ``DatasetSpec``. The one thing that's genuinely dynamic
-- actually reading the dataset's metadata file off disk -- is deferred to a
small per-dataset loader function, so importing this module never touches
the filesystem.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd


@dataclass
class DatasetSpec:
    name: str
    dataset_path: str
    preprocessed_dir: str
    ch_names: list[str]
    tasks: list[str]
    label_maps: dict[str, dict[str, int]]
    metadata_loader: Callable[["DatasetSpec"], tuple[pd.DataFrame, Any, Any]]

    @property
    def ch_num(self) -> int:
        return len(self.ch_names)

    def load_metadata(self):
        """Returns (metadata, subjects_list, labels_list)."""
        return self.metadata_loader(self)


def _load_geneeg_metadata(spec: DatasetSpec):
    metadata = pd.read_excel(os.path.join(spec.dataset_path, 'metadata.xlsx'))
    return metadata, metadata['id'], metadata['status']


def _load_mcivshc_metadata(spec: DatasetSpec):
    metadata = pd.read_excel(os.path.join(spec.dataset_path, 'states_2.xlsx'))
    return metadata, metadata['file number'], metadata['status']


def _load_advsftdvshc_metadata(spec: DatasetSpec):
    metadata = pd.read_csv(os.path.join(spec.dataset_path, 'participants.tsv'), sep='\t')
    return metadata, metadata['participant_id'], metadata['Group']


def _load_caueeg_metadata(spec: DatasetSpec):
    # Lives outside dataset_path (which points at the raw signal/edf/ dir), so
    # it's a literal path rather than derived from spec.dataset_path.
    annotation_path = './datasets/raw/CAUEEG/dementia-no-overlap.json'
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)

    symptom_to_label = {'normal': 'Normal', 'mci': 'MCI', 'dementia': 'Dementia'}

    subjects_list, labels_list = [], []
    for split in ('train_split', 'test_split'):
        for entry in annotation[split]:
            for symptom_key, label in symptom_to_label.items():
                if symptom_key in entry['symptom']:
                    subjects_list.append(entry['serial'])
                    labels_list.append(label)
                    break

    metadata = pd.DataFrame(list(zip(subjects_list, labels_list)), columns=['id', 'label'])
    return metadata, metadata['id'], metadata['label']


DATASETS: dict[str, DatasetSpec] = {
    'GENEEG': DatasetSpec(
        name='GENEEG',
        dataset_path='./datasets/raw/GENEEG/all_data',
        preprocessed_dir='./datasets/preprocessed/GENEEG_preprocessed',
        ch_names=['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'C3', 'C4', 'P3', 'P4',
                  'O1', 'O2', 'T3', 'T4', 'Fz', 'Cz', 'Pz'],
        tasks=['MCI vs HC'],
        label_maps={
            'MCI vs HC': {'Normal': 0, 'MCI': 1},
        },
        metadata_loader=_load_geneeg_metadata,
    ),

    'MCIvsHC': DatasetSpec(
        name='MCIvsHC',
        dataset_path='./datasets/raw/MCIvsHC/data_extended/',
        preprocessed_dir='./datasets/raw/MCIvsHC_preprocessed',
        ch_names=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
                  'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'],
        tasks=['MCI vs HC'],
        label_maps={
            'MCI vs HC': {'NORMAL': 0, 'MCI': 1},
        },
        metadata_loader=_load_mcivshc_metadata,
    ),

    'ADvsFTDvsHC': DatasetSpec(
        name='ADvsFTDvsHC',
        dataset_path='datasets/raw/ADvsFTDvsHC/data/',
        preprocessed_dir='datasets/preprocessed/ADvsFTDvsHC_preprocessed/',
        ch_names=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
                  'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'],
        tasks=['AD vs HC', 'FTD vs HC', 'FTD vs AD'],
        label_maps={
            'AD vs HC': {'C': 0, 'A': 1},
            'FTD vs HC': {'C': 0, 'F': 1},
            'FTD vs AD': {'F': 0, 'A': 1},
            '3-class': {'C': 0, 'F': 1, 'A': 2},
        },
        metadata_loader=_load_advsftdvshc_metadata,
    ),

    'CAUEEG': DatasetSpec(
        name='CAUEEG',
        dataset_path='./datasets/raw/CAUEEG/signal/edf/',
        preprocessed_dir='./datasets/preprocessed/CAUEEG-preprocessed/',
        ch_names=['Fp1', 'F3', 'C3', 'P3', 'O1', 'Fp2', 'F4', 'C4', 'P4', 'O2',
                  'F7', 'T3', 'T5', 'F8', 'T4', 'T6', 'Fz', 'Cz', 'Pz'],
        tasks=['Dementia vs Normal', 'MCI vs Normal', 'MCI vs Dementia'],
        label_maps={
            'Dementia vs Normal': {'Normal': 0, 'Dementia': 1},
            'MCI vs Normal': {'Normal': 0, 'MCI': 1},
            'MCI vs Dementia': {'MCI': 0, 'Dementia': 1},
            '3-class': {'Normal': 0, 'MCI': 1, 'Dementia': 2},
        },
        metadata_loader=_load_caueeg_metadata,
    ),
}
