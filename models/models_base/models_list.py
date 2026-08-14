from dataclasses import dataclass, field
from typing import Any

from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, SGD

from models.models_base.Oh_CNN import Oh_CNN
from models.models_base.EEGNet import EEGnet
from models.models_base.ShallowConvNet import ShawllowConvNet
from models.models_base.SCCNet import SCCNet
from models.models_base.DeepConvNet import DeepConvNet
from models.models_base.SzHNN import SzHNN
from models.models_base.EEG_CNN import EEG_CNN
from models.models_base.eeg_conformer import Conformer
from models.models_base.MultiBranch import MBSzEEGNet
from models.models_base.eeg_deformer import Deformer
from models.models_base.MSVTNet import MSVTNet, JointCrossEntoryLoss

import math

MODEL_NAMES = [
    'EEG_Conformer', 'EEG_Deformer', 'MSVTNet', 'Oh_CNN', 'SzHNN',
    'DeepConvNet', 'EEGNet', 'ShallowConveNet', 'SCCNet', 'MBSzEEGNet',
]


@dataclass
class ModelSpec:
    """One entry in the model registry: a name, the class to instantiate,
    the constructor kwargs to instantiate it with, and the optimizer/loss
    this model is trained with. Defaults (AdamW, CrossEntropyLoss) cover
    every model except the two with a documented reason to differ:
    MBSzEEGNet (SGD) and MSVTNet (JointCrossEntoryLoss, since its forward()
    returns per-branch auxiliary outputs that only this loss consumes)."""
    name: str
    cls: type
    kwargs: dict[str, Any] = field(default_factory=dict)
    optimizer_cls: type = AdamW
    criterion_cls: type = CrossEntropyLoss


def models_list(num_classes, num_channels, time_points) -> list[ModelSpec]:

    return [

        ModelSpec(
            name='EEG_Conformer',
            cls=Conformer,
            kwargs={
                'num_classes': num_classes,
                'num_channels': num_channels,
                'time_points': time_points,
            },
        ),

        ModelSpec(
            name='EEG_Deformer',
            cls=Deformer,
            kwargs={
                'num_chan': num_channels,
                'num_time': time_points,
                'num_classes': num_classes,
            },
        ),

        ModelSpec(
            name='MSVTNet',
            cls=MSVTNet,
            kwargs={
                'num_ch': num_channels,
                'nTime': time_points,
                'num_classes': num_classes,
            },
            criterion_cls=JointCrossEntoryLoss,
        ),

        ModelSpec(
            name='Oh_CNN',
            cls=Oh_CNN,
            kwargs={
                'num_classes': num_classes,
                'num_channels': num_channels,
                'time_points': time_points,
            },
        ),

        ModelSpec(
            name='SzHNN',
            cls=SzHNN,
            kwargs={
                'num_classes': num_classes,
                'num_channels': num_channels,
            },
        ),

        ModelSpec(
            name='DeepConvNet',
            cls=DeepConvNet,
            kwargs={
                'num_classes': num_classes,
                'num_channels': num_channels,
                'time_points': time_points,
            },
        ),

        ModelSpec(
            name='EEGNet',
            cls=EEGnet,
            kwargs={
                'num_classes': num_classes,
                'num_channels': num_channels,
                'time_points': time_points,
                'half_sfreq': math.floor(200 / 2),
            },
        ),

        ModelSpec(
            name='ShallowConveNet',
            cls=ShawllowConvNet,
            kwargs={
                'num_classes': num_classes,
                'C': num_channels,
                'N': time_points,
            },
        ),

        ModelSpec(
            name='SCCNet',
            cls=SCCNet,
            kwargs={
                'num_classes': num_classes,
                'C': num_channels,
                'N': time_points,
                'sfreq': time_points,
            },
        ),

        ModelSpec(
            name='MBSzEEGNet',
            cls=MBSzEEGNet,
            kwargs={
                'channels': num_channels,
                'samples': time_points,
            },
            optimizer_cls=SGD,
        ),
    ]


def single_model(num_classes, num_channels, time_points, model_name='SCCNet') -> ModelSpec:
    available = models_list(num_classes, num_channels, time_points)
    for entry in available:
        if entry.name == model_name:
            return entry
    raise ValueError(
        f"Unknown model_name '{model_name}'. "
        f"Available: {[entry.name for entry in available]}"
    )