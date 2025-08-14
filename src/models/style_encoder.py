"""
Style Audio Encoder

Encodes 10-second audio clips into style vectors using log-mel spectrograms.
Uses a CNN-based architecture similar to MusicNN or short-time CNNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LogMelExtractor(nn.Module):
    """Extract log-mel spectrograms from raw audio"""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: Optional[float] = None
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Create mel filterbank
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max or sample_rate // 2,
            power=2.0,
            normalized=True
        )
        
        # Small epsilon for log computation
        self.eps = 1e-10
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to log-mel spectrogram
        
        Args:
            audio: (batch_size, channels, time) or (batch_size, time)
            
        Returns:
            log_mel: (batch_size, n_mels, time_frames)
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # Add channel dim
        
        # Convert to mono if stereo
        if audio.size(1) > 1:
            audio = audio.mean(dim=1, keepdim=True)
        
        # Extract mel spectrogram
        mel_spec = self.mel_transform(audio.squeeze(1))  # (batch, n_mels, time)
        
        # Convert to log scale
        log_mel = torch.log(mel_spec + self.eps)
        
        return log_mel


class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (1, 1),
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class StyleEncoder(nn.Module):
    """
    CNN-based audio encoder that maps log-mel spectrograms to style vectors
    
    Architecture inspired by MusicNN and similar audio classification models
    """
    
    def __init__(
        self,
        n_mels: int = 128,
        embedding_dim: int = 512,
        n_classes: int = 3,  # rock_punk, rnb_ballad, country_pop
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        
        # Log-mel extractor
        self.mel_extractor = LogMelExtractor(n_mels=n_mels)
        
        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            # First block: capture local patterns
            ConvBlock(1, 32, kernel_size=(3, 3), stride=(1, 1)),
            ConvBlock(32, 32, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d((2, 2)),
            
            # Second block: mid-level features
            ConvBlock(32, 64, kernel_size=(3, 3), stride=(1, 1)),
            ConvBlock(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d((2, 2)),
            
            # Third block: high-level features
            ConvBlock(64, 128, kernel_size=(3, 3), stride=(1, 1)),
            ConvBlock(128, 128, kernel_size=(3, 3), stride=(1, 1)),
            nn.MaxPool2d((2, 2)),
            
            # Fourth block: abstract features
            ConvBlock(128, 256, kernel_size=(3, 3), stride=(1, 1)),
            ConvBlock(256, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Style embedding projection
        self.style_projection = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.Tanh()  # Normalize embeddings to [-1, 1]
        )
        
        # Classification head for training
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
        
    def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract convolutional features from audio"""
        # Convert to log-mel
        log_mel = self.mel_extractor(audio)  # (batch, n_mels, time)
        
        # Add channel dimension for conv2d
        log_mel = log_mel.unsqueeze(1)  # (batch, 1, n_mels, time)
        
        # Extract features
        features = self.conv_layers(log_mel)  # (batch, 256, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (batch, 256)
        
        return features
        
    def encode_style(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to style vector"""
        features = self.extract_features(audio)
        style_vector = self.style_projection(features)
        return style_vector
        
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training
        
        Args:
            audio: (batch_size, time) audio waveform
            
        Returns:
            dict with 'embeddings' and 'logits'
        """
        features = self.extract_features(audio)
        embeddings = self.style_projection(features)
        logits = self.classifier(embeddings)
        
        return {
            'embeddings': embeddings,
            'logits': logits,
            'features': features
        }


class StyleEncoderLoss(nn.Module):
    """Combined loss for style encoder training"""
    
    def __init__(
        self,
        classification_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        margin: float = 0.5,
        temperature: float = 0.1
    ):
        super().__init__()
        
        self.classification_weight = classification_weight
        self.contrastive_weight = contrastive_weight
        self.margin = margin
        self.temperature = temperature
        
        self.ce_loss = nn.CrossEntropyLoss()
        
    def contrastive_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Contrastive loss to cluster same-style embeddings"""
        batch_size = embeddings.size(0)
        
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise similarities
        similarities = torch.matmul(embeddings_norm, embeddings_norm.t()) / self.temperature
        
        # Create mask for positive pairs (same style)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_eq = labels_eq.float()
        
        # Remove diagonal
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        labels_eq.masked_fill_(mask, 0)
        
        # Positive and negative similarities
        pos_similarities = similarities * labels_eq
        neg_similarities = similarities * (1 - labels_eq)
        
        # Contrastive loss
        pos_loss = -torch.log(torch.exp(pos_similarities).sum(dim=1) + 1e-8)
        neg_loss = torch.log(torch.exp(neg_similarities).sum(dim=1) + 1e-8)
        
        contrastive = (pos_loss + neg_loss).mean()
        
        return contrastive
        
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor], 
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            outputs: dict with 'embeddings', 'logits', 'features'
            labels: (batch_size,) style class labels
            
        Returns:
            dict with loss components
        """
        # Classification loss
        ce_loss = self.ce_loss(outputs['logits'], labels)
        
        # Contrastive loss
        contrastive_loss = self.contrastive_loss(outputs['embeddings'], labels)
        
        # Combined loss
        total_loss = (
            self.classification_weight * ce_loss + 
            self.contrastive_weight * contrastive_loss
        )
        
        return {
            'total_loss': total_loss,
            'classification_loss': ce_loss,
            'contrastive_loss': contrastive_loss
        }


def create_style_encoder(config: Dict) -> StyleEncoder:
    """Factory function to create style encoder"""
    return StyleEncoder(
        n_mels=config.get('n_mels', 128),
        embedding_dim=config.get('embedding_dim', 512),
        n_classes=config.get('n_classes', 3),
        dropout=config.get('dropout', 0.2)
    )


# Training utilities
def preprocess_audio(
    audio_path: str, 
    target_length: float = 10.0,
    sample_rate: int = 22050
) -> torch.Tensor:
    """Load and preprocess audio for style encoding"""
    
    # Load audio
    waveform, orig_sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if orig_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Trim or pad to target length
    target_samples = int(target_length * sample_rate)
    current_samples = waveform.size(1)
    
    if current_samples > target_samples:
        # Random crop
        start_idx = torch.randint(0, current_samples - target_samples + 1, (1,)).item()
        waveform = waveform[:, start_idx:start_idx + target_samples]
    elif current_samples < target_samples:
        # Pad with zeros
        padding = target_samples - current_samples
        waveform = F.pad(waveform, (0, padding))
    
    return waveform.squeeze(0)  # Remove channel dim


if __name__ == "__main__":
    # Example usage
    config = {
        'n_mels': 128,
        'embedding_dim': 512,
        'n_classes': 3,
        'dropout': 0.2
    }
    
    # Create model
    model = create_style_encoder(config)
    
    # Example input (10 seconds at 22050 Hz)
    audio = torch.randn(4, 220500)  # Batch of 4 audio clips
    
    # Forward pass
    outputs = model(audio)
    
    print(f"Embeddings shape: {outputs['embeddings'].shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Features shape: {outputs['features'].shape}")