"""
Finetuning package for preference optimization.

Implements Direct Preference Optimization (DPO) for aligning
symbolic music generation with human preferences.
"""

from .dpo import DPOLoss, MusicGeneratorWrapper, DPOTrainer, create_mock_dpo_data

__all__ = [
    'DPOLoss',
    'MusicGeneratorWrapper', 
    'DPOTrainer',
    'create_mock_dpo_data'
]