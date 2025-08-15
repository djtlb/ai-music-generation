"""
Critic package for music generation quality assessment.
Includes adherence classifier, comprehensive critic model, and DPO finetuning.
"""

from .classifier import AdherenceClassifier, AdherenceScore, AdherenceDataset, train_classifier
from .model import ComprehensiveCritic, CriticScore, StyleEmbeddingEncoder, MixQualityAssessor, extract_mix_features
from .dpo_finetune import DPOTrainer, DPODataset, DPOLoss, create_preference_pairs
from .evaluate import AdherenceEvaluator

__all__ = [
    # Classifier
    'AdherenceClassifier',
    'AdherenceScore', 
    'AdherenceDataset',
    'train_classifier',
    
    # Main critic model
    'ComprehensiveCritic',
    'CriticScore',
    'StyleEmbeddingEncoder',
    'MixQualityAssessor',
    'extract_mix_features',
    
    # DPO training
    'DPOTrainer',
    'DPODataset', 
    'DPOLoss',
    'create_preference_pairs',
    
    # Evaluation
    'AdherenceEvaluator'
]