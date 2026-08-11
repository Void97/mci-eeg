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
from models.models_base.MSVTNet import MSVTNet

import math

MODEL_NAMES = [
    'EEG_Conformer', 'EEG_Deformer', 'MSVTNet', 'Oh_CNN', 'SzHNN',
    'DeepConvNet', 'EEGNet', 'ShallowConveNet', 'SCCNet', 'MBSzEEGNet',
]


def models_list(num_classes, num_channels, time_points):

    return  [

        {
            'name': 'EEG_Conformer',
            'class': Conformer,
            'kwargs': {
                'num_classes': num_classes,
                'num_channels': num_channels,
                'time_points': time_points
            }
        },
            
        {    
            'name': 'EEG_Deformer',
            'class': Deformer, 
            'kwargs': {
                'num_chan': num_channels,
                'num_time': time_points,
                'num_classes': num_classes
            },
        },
        
        {
            'name': 'MSVTNet',
            'class': MSVTNet,
            'kwargs': {
                'num_ch': num_channels,
                'nTime': time_points,
                'num_classes': num_classes,
            },
        },
        
        {
            'name': 'Oh_CNN',
            'class': Oh_CNN,
            'kwargs': {
                'num_classes': num_classes,
                'num_channels': num_channels,
                'time_points': time_points,
            }
        },

        {
            'name': 'SzHNN',
            'class': SzHNN,
            'kwargs': {
                'num_classes': num_classes,
                'num_channels': num_channels,
            }
        },

        {
            'name': 'DeepConvNet',
            'class': DeepConvNet,
            'kwargs': {
                'num_classes': num_classes,
                'num_channels': num_channels,
                'time_points': time_points,
            }
        },

        {
            'name': 'EEGNet',
            'class': EEGnet,
            'kwargs': {
                'num_classes': num_classes,
                'num_channels': num_channels,
                'time_points': time_points,
                'half_sfreq': math.floor(200/2)
            }
        },

        {
            'name': 'ShallowConveNet',
            'class': ShawllowConvNet,
            'kwargs': {
                'num_classes': num_classes,
                'C': num_channels,
                'N': time_points,
            }
        },

        {
            'name': 'SCCNet',
            'class': SCCNet,
            'kwargs': {
                'num_classes': num_classes,
                'C': num_channels,
                'N': time_points,
                'sfreq': time_points,
            }
        },

        {
            'name': 'MBSzEEGNet',
            'class': MBSzEEGNet,
            'kwargs': {
                'channels': num_channels,
                'samples': time_points,
            }
        },
    ]

def single_model(num_classes, num_channels, time_points, model_name='SCCNet'):
    available = models_list(num_classes, num_channels, time_points)
    for entry in available:
        if entry['name'] == model_name:
            return entry
    raise ValueError(
        f"Unknown model_name '{model_name}'. "
        f"Available: {[entry['name'] for entry in available]}"
    )
