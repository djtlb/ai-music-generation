"""
Critic model package for music quality assessment.

This package implements a neural critic that scores 10-second audio clips
on multiple quality dimensions for training preference-aligned music generation.
"""

from .model import CriticModel, CriticLoss, create_critic_model
from .dataset import PreferenceDataset, PairwisePreferenceDataset, create_preference_dataloader
from .train import CriticTrainer
from .evaluate import ModelEvaluator, create_validation_playlist

__all__ = [
    'CriticModel',
    'CriticLoss', 
    'create_critic_model',
    'PreferenceDataset',
    'PairwisePreferenceDataset',
    'create_preference_dataloader',
    'CriticTrainer',
    'ModelEvaluator',
    'create_validation_playlist'
]