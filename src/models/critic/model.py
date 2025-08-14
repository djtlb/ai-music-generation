"""
Critic model for scoring 10-second audio clips on multiple quality dimensions.

This module implements a neural network that takes audio features as input
and outputs scores for:
- hook_strength: Memorability and catchiness of melodic content
- harmonic_stability: Quality and coherence of chord progressions  
- arrangement_contrast: Dynamic variation and structural interest
- mix_quality: Technical audio quality (LUFS, spectral balance)
- style_match: Consistency with target musical style
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class AudioFeatureExtractor(nn.Module):
    """Extracts audio features from 10-second clips for critic evaluation."""
    
    def __init__(self, sample_rate: int = 44100, n_mels: int = 128):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        
        # Convolutional feature extraction for spectrograms
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Temporal attention for sequence modeling
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=8, batch_first=True
        )
        
        self.feature_norm = nn.LayerNorm(128)
        
    def forward(self, mel_spectrograms: torch.Tensor) -> torch.Tensor:
        """
        Extract features from mel spectrograms.
        
        Args:
            mel_spectrograms: [batch, 1, time, freq] mel spectrogram tensors
            
        Returns:
            features: [batch, feature_dim] extracted audio features
        """
        batch_size = mel_spectrograms.size(0)
        
        # Extract convolutional features
        conv_features = self.conv_layers(mel_spectrograms)  # [batch, 128, 8, 8]
        conv_features = conv_features.view(batch_size, 128, -1).transpose(1, 2)  # [batch, 64, 128]
        
        # Apply temporal attention
        attended_features, _ = self.temporal_attention(
            conv_features, conv_features, conv_features
        )
        
        # Global average pooling
        pooled_features = attended_features.mean(dim=1)  # [batch, 128]
        
        return self.feature_norm(pooled_features)


class CriticModel(nn.Module):
    """
    Critic model that scores audio clips on multiple quality dimensions.
    
    Architecture:
    - Audio feature extractor (CNN + attention)
    - Style embedding lookup
    - Multi-task prediction heads for each quality dimension
    - Auxiliary features (LUFS, spectral centroid, etc.)
    """
    
    def __init__(
        self,
        audio_feature_dim: int = 128,
        style_embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_styles: int = 3,  # rock_punk, rnb_ballad, country_pop
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.audio_feature_extractor = AudioFeatureExtractor()
        
        # Style embeddings
        self.style_embeddings = nn.Embedding(num_styles, style_embedding_dim)
        
        # Auxiliary feature processing (LUFS, spectral features, etc.)
        self.aux_feature_dim = 8  # LUFS, spectral_centroid, etc.
        
        # Combined feature processing
        combined_dim = audio_feature_dim + style_embedding_dim + self.aux_feature_dim
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-task prediction heads
        self.quality_heads = nn.ModuleDict({
            'hook_strength': nn.Linear(hidden_dim, 1),
            'harmonic_stability': nn.Linear(hidden_dim, 1),
            'arrangement_contrast': nn.Linear(hidden_dim, 1),
            'mix_quality': nn.Linear(hidden_dim, 1),
            'style_match': nn.Linear(hidden_dim, 1)
        })
        
        # Overall score head (weighted combination)
        self.overall_head = nn.Linear(hidden_dim, 1)
        
    def forward(
        self, 
        mel_spectrograms: torch.Tensor,
        style_ids: torch.Tensor,
        aux_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of critic model.
        
        Args:
            mel_spectrograms: [batch, 1, time, freq] mel spectrograms
            style_ids: [batch] style indices (0=rock_punk, 1=rnb_ballad, 2=country_pop)
            aux_features: [batch, aux_dim] auxiliary features (LUFS, spectral, etc.)
            
        Returns:
            scores: Dict of quality dimension scores, each [batch, 1]
        """
        # Extract audio features
        audio_features = self.audio_feature_extractor(mel_spectrograms)
        
        # Get style embeddings
        style_features = self.style_embeddings(style_ids)
        
        # Combine all features
        combined_features = torch.cat([
            audio_features, style_features, aux_features
        ], dim=-1)
        
        # Fuse features
        fused_features = self.feature_fusion(combined_features)
        
        # Predict quality scores
        scores = {}
        for quality_name, head in self.quality_heads.items():
            scores[quality_name] = torch.sigmoid(head(fused_features))
            
        # Overall score
        scores['overall'] = torch.sigmoid(self.overall_head(fused_features))
        
        return scores
    
    def predict_scores(
        self,
        mel_spectrograms: torch.Tensor,
        style_ids: torch.Tensor, 
        aux_features: torch.Tensor
    ) -> Dict[str, float]:
        """
        Predict scores for a single batch (inference mode).
        
        Returns:
            scores: Dict of quality scores as Python floats
        """
        self.eval()
        with torch.no_grad():
            scores = self.forward(mel_spectrograms, style_ids, aux_features)
            
        # Convert to Python floats
        return {
            name: score.item() if score.numel() == 1 else score.cpu().numpy()
            for name, score in scores.items()
        }


class CriticLoss(nn.Module):
    """
    Multi-task loss for training the critic model.
    
    Combines:
    - Individual quality dimension losses
    - Overall score loss  
    - Regularization terms
    """
    
    def __init__(
        self,
        quality_weights: Optional[Dict[str, float]] = None,
        overall_weight: float = 2.0,
        consistency_weight: float = 0.1
    ):
        super().__init__()
        
        self.quality_weights = quality_weights or {
            'hook_strength': 1.0,
            'harmonic_stability': 1.0, 
            'arrangement_contrast': 1.0,
            'mix_quality': 1.0,
            'style_match': 1.0
        }
        self.overall_weight = overall_weight
        self.consistency_weight = consistency_weight
        
    def forward(
        self,
        predicted_scores: Dict[str, torch.Tensor],
        target_scores: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task critic loss.
        
        Args:
            predicted_scores: Model predictions for each quality dimension
            target_scores: Ground truth scores
            
        Returns:
            total_loss: Combined loss value
            loss_breakdown: Individual loss components
        """
        losses = {}
        
        # Individual quality dimension losses
        quality_loss = 0.0
        for quality_name in self.quality_weights:
            if quality_name in predicted_scores and quality_name in target_scores:
                loss = F.mse_loss(
                    predicted_scores[quality_name],
                    target_scores[quality_name]
                )
                losses[f'{quality_name}_loss'] = loss
                quality_loss += self.quality_weights[quality_name] * loss
        
        losses['quality_loss'] = quality_loss
        
        # Overall score loss
        if 'overall' in predicted_scores and 'overall' in target_scores:
            overall_loss = F.mse_loss(
                predicted_scores['overall'],
                target_scores['overall']
            )
            losses['overall_loss'] = overall_loss
        else:
            overall_loss = 0.0
            
        # Consistency loss: overall should be consistent with individual scores
        if 'overall' in predicted_scores and len(self.quality_weights) > 0:
            individual_scores = torch.stack([
                predicted_scores[name] for name in self.quality_weights.keys()
                if name in predicted_scores
            ], dim=-1)
            avg_individual = individual_scores.mean(dim=-1, keepdim=True)
            consistency_loss = F.mse_loss(predicted_scores['overall'], avg_individual)
            losses['consistency_loss'] = consistency_loss
        else:
            consistency_loss = 0.0
            
        # Total loss
        total_loss = (
            quality_loss + 
            self.overall_weight * overall_loss +
            self.consistency_weight * consistency_loss
        )
        
        losses['total_loss'] = total_loss
        
        return total_loss, losses


def extract_auxiliary_features(audio: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
    """
    Extract auxiliary audio features for critic model.
    
    Args:
        audio: Audio signal as numpy array
        sample_rate: Sample rate in Hz
        
    Returns:
        features: Auxiliary features array [aux_feature_dim]
    """
    # Placeholder implementation - in practice would use librosa, pyloudnorm, etc.
    features = np.array([
        0.0,  # LUFS loudness
        0.0,  # Spectral centroid  
        0.0,  # Spectral rolloff
        0.0,  # Zero crossing rate
        0.0,  # RMS energy
        0.0,  # Spectral contrast
        0.0,  # Chroma deviation
        0.0   # Tempo stability
    ])
    
    return features


def create_critic_model(
    num_styles: int = 3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> CriticModel:
    """
    Factory function to create and initialize a critic model.
    
    Args:
        num_styles: Number of musical styles to support
        device: Device to place model on
        
    Returns:
        model: Initialized CriticModel
    """
    model = CriticModel(num_styles=num_styles)
    model.to(device)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
    model.apply(init_weights)
    
    return model


if __name__ == "__main__":
    # Test the critic model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = create_critic_model(device=device)
    
    # Test forward pass
    batch_size = 4
    mel_spectrograms = torch.randn(batch_size, 1, 128, 431).to(device)  # ~10s @ 22kHz
    style_ids = torch.randint(0, 3, (batch_size,)).to(device)
    aux_features = torch.randn(batch_size, 8).to(device)
    
    # Forward pass
    scores = model(mel_spectrograms, style_ids, aux_features)
    
    print("Critic model test successful!")
    print(f"Output scores shape: {scores['overall'].shape}")
    print(f"Quality dimensions: {list(scores.keys())}")