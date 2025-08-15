"""
Main Critic Model: Combines adherence scores, style embeddings, and mix features
to provide comprehensive quality assessment for generated music.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass
import logging

from .classifier import AdherenceClassifier, AdherenceScore

logger = logging.getLogger(__name__)

@dataclass
class CriticScore:
    """Comprehensive critic evaluation"""
    overall: float  # 0-1 combined score
    adherence: float  # From classifier
    style_match: float  # Cosine similarity
    mix_quality: float  # LUFS/spectral/dynamics score
    
    # Detailed breakdown
    adherence_details: AdherenceScore
    style_details: Dict[str, float]
    mix_details: Dict[str, float]
    
    # Meta information
    confidence: float  # Model confidence in prediction
    notes: List[str]  # Qualitative feedback

class StyleEmbeddingEncoder(nn.Module):
    """Encode audio to style embeddings for comparison"""
    def __init__(self, input_dim: int = 128, output_dim: int = 512):
        super().__init__()
        
        # Audio feature encoder (mel-spectrogram -> embedding)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim),
            nn.Tanh()  # Normalize for cosine similarity
        )
        
    def forward(self, mel_spectrograms: torch.Tensor) -> torch.Tensor:
        """
        Encode mel spectrograms to style embeddings
        
        Args:
            mel_spectrograms: [batch_size, 1, time, mel_bins]
            
        Returns:
            style_embeddings: [batch_size, output_dim]
        """
        # Convolutional feature extraction
        conv_out = self.conv_layers(mel_spectrograms)
        
        # Flatten and process through FC layers
        batch_size = conv_out.size(0)
        flattened = conv_out.view(batch_size, -1)
        embeddings = self.fc_layers(flattened)
        
        return embeddings

class MixQualityAssessor(nn.Module):
    """Assess mixing and mastering quality from audio features"""
    def __init__(self, input_dim: int = 32):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0-1 quality score
        )
        
    def forward(self, mix_features: torch.Tensor) -> torch.Tensor:
        """
        Assess mix quality from extracted features
        
        Args:
            mix_features: [batch_size, feature_dim] audio analysis features
            
        Returns:
            quality_scores: [batch_size, 1] quality scores 0-1
        """
        return self.layers(mix_features)

class ComprehensiveCritic(nn.Module):
    """
    Main critic model combining adherence, style matching, and mix quality.
    Provides comprehensive assessment of generated music quality.
    """
    def __init__(
        self,
        vocab_size: int,
        style_embed_dim: int = 512,
        adherence_weight: float = 0.4,
        style_weight: float = 0.3,
        mix_weight: float = 0.3
    ):
        super().__init__()
        
        # Component models
        self.adherence_classifier = AdherenceClassifier(vocab_size)
        self.style_encoder = StyleEmbeddingEncoder(output_dim=style_embed_dim)
        self.mix_assessor = MixQualityAssessor()
        
        # Weighting for final score
        self.adherence_weight = adherence_weight
        self.style_weight = style_weight
        self.mix_weight = mix_weight
        
        # Confidence estimation network
        self.confidence_estimator = nn.Sequential(
            nn.Linear(3, 16),  # 3 component scores -> confidence
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        prompt_texts: List[str],
        control_jsons: List[Dict],
        token_sequences: torch.Tensor,
        generated_mel_specs: torch.Tensor,
        reference_style_embeddings: torch.Tensor,
        mix_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Comprehensive forward pass
        
        Args:
            prompt_texts: Original prompts
            control_jsons: Control JSON specifications
            token_sequences: Generated token sequences
            generated_mel_specs: Mel spectrograms of generated audio
            reference_style_embeddings: Target style embeddings
            mix_features: Audio analysis features (LUFS, spectral, etc.)
            
        Returns:
            overall_scores: [batch_size, 1] combined scores
            component_scores: Dict with detailed breakdowns
        """
        batch_size = len(prompt_texts)
        
        # 1. Adherence scoring
        adherence_scores, adherence_components = self.adherence_classifier(
            prompt_texts, control_jsons, token_sequences
        )
        
        # 2. Style matching
        generated_style_embeddings = self.style_encoder(generated_mel_specs)
        style_similarities = F.cosine_similarity(
            generated_style_embeddings, 
            reference_style_embeddings,
            dim=1
        ).unsqueeze(1)  # [batch_size, 1]
        
        # Convert cosine similarity to 0-1 score
        style_scores = (style_similarities + 1) / 2
        
        # 3. Mix quality assessment
        mix_scores = self.mix_assessor(mix_features)
        
        # 4. Combine scores
        overall_scores = (
            self.adherence_weight * adherence_scores +
            self.style_weight * style_scores +
            self.mix_weight * mix_scores
        )
        
        # 5. Estimate confidence
        component_stack = torch.cat([adherence_scores, style_scores, mix_scores], dim=1)
        confidence_scores = self.confidence_estimator(component_stack)
        
        # Return detailed breakdown
        component_scores = {
            'adherence': adherence_scores,
            'style_match': style_scores,
            'mix_quality': mix_scores,
            'confidence': confidence_scores,
            'adherence_components': adherence_components
        }
        
        return overall_scores, component_scores
    
    def evaluate_comprehensive(
        self,
        prompt_texts: List[str],
        control_jsons: List[Dict],
        token_sequences: torch.Tensor,
        generated_mel_specs: torch.Tensor,
        reference_style_embeddings: torch.Tensor,
        mix_features: torch.Tensor,
        target_mix_specs: Optional[Dict[str, float]] = None
    ) -> List[CriticScore]:
        """
        Evaluate and return structured comprehensive scores
        
        Args:
            target_mix_specs: Optional target mix specifications for detailed analysis
            
        Returns:
            List of CriticScore objects with detailed analysis
        """
        self.eval()
        with torch.no_grad():
            overall_scores, component_scores = self.forward(
                prompt_texts, control_jsons, token_sequences,
                generated_mel_specs, reference_style_embeddings, mix_features
            )
            
            # Get detailed adherence scores
            adherence_details = self.adherence_classifier.predict_adherence(
                prompt_texts, control_jsons, token_sequences
            )
            
            results = []
            for i in range(len(prompt_texts)):
                # Extract mix feature details
                mix_details = self._analyze_mix_features(
                    mix_features[i].cpu().numpy(),
                    target_mix_specs
                )
                
                # Style analysis details
                style_details = {
                    'cosine_similarity': component_scores['style_match'][i].item(),
                    'embedding_norm': torch.norm(
                        self.style_encoder(generated_mel_specs[i:i+1])
                    ).item()
                }
                
                # Generate qualitative notes
                notes = self._generate_feedback_notes(
                    adherence_details[i],
                    component_scores['style_match'][i].item(),
                    mix_details,
                    overall_scores[i].item()
                )
                
                score = CriticScore(
                    overall=overall_scores[i].item(),
                    adherence=component_scores['adherence'][i].item(),
                    style_match=component_scores['style_match'][i].item(),
                    mix_quality=component_scores['mix_quality'][i].item(),
                    adherence_details=adherence_details[i],
                    style_details=style_details,
                    mix_details=mix_details,
                    confidence=component_scores['confidence'][i].item(),
                    notes=notes
                )
                results.append(score)
            
            return results
    
    def _analyze_mix_features(
        self, 
        mix_features: np.ndarray, 
        target_specs: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Analyze mix features and compare to targets"""
        # Assume mix_features contains: [lufs, spectral_centroid, stereo_width, 
        # dynamic_range, rms_energy, ...] - expand based on actual features
        
        feature_names = [
            'lufs', 'spectral_centroid', 'stereo_width', 'dynamic_range',
            'rms_energy', 'spectral_rolloff', 'spectral_flux', 'zero_crossing_rate'
        ]
        
        analysis = {}
        for i, name in enumerate(feature_names[:len(mix_features)]):
            analysis[name] = float(mix_features[i])
            
            # Compare to targets if provided
            if target_specs and name in target_specs:
                target = target_specs[name]
                deviation = abs(analysis[name] - target) / max(abs(target), 1e-6)
                analysis[f'{name}_target_deviation'] = deviation
        
        return analysis
    
    def _generate_feedback_notes(
        self,
        adherence: AdherenceScore,
        style_match: float,
        mix_details: Dict[str, float],
        overall_score: float
    ) -> List[str]:
        """Generate human-readable feedback notes"""
        notes = []
        
        # Overall assessment
        if overall_score >= 0.8:
            notes.append("Excellent overall quality - ready for release")
        elif overall_score >= 0.6:
            notes.append("Good quality with minor areas for improvement")
        elif overall_score >= 0.4:
            notes.append("Moderate quality - some significant issues to address")
        else:
            notes.append("Low quality - major improvements needed")
        
        # Adherence feedback
        if adherence.overall < 0.5:
            notes.append("Generated music doesn't closely match the prompt")
            if adherence.tempo_adherence < 0.5:
                notes.append("Tempo doesn't match specified BPM")
            if adherence.structure_adherence < 0.5:
                notes.append("Song structure deviates from requested arrangement")
        
        # Style feedback
        if style_match < 0.4:
            notes.append("Style doesn't match reference - consider different genre tokens")
        elif style_match > 0.8:
            notes.append("Excellent style matching - captures target genre well")
        
        # Mix feedback
        lufs = mix_details.get('lufs', 0)
        if lufs < -20:
            notes.append("Mix is too quiet - needs more loudness")
        elif lufs > -6:
            notes.append("Mix is too loud - may cause distortion")
        
        spectral_centroid = mix_details.get('spectral_centroid', 0)
        if spectral_centroid < 1000:
            notes.append("Mix sounds muddy - needs more high frequency content")
        elif spectral_centroid > 4000:
            notes.append("Mix sounds harsh - too much high frequency energy")
        
        return notes

def extract_mix_features(audio_data: np.ndarray, sample_rate: int = 44100) -> np.ndarray:
    """
    Extract mix analysis features from audio data
    
    Args:
        audio_data: Audio samples [n_samples] or [n_channels, n_samples]
        sample_rate: Audio sample rate
        
    Returns:
        features: [feature_dim] array of mix features
    """
    import librosa
    
    # Ensure mono
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=0)
    
    # Basic features
    features = []
    
    # 1. RMS Energy (approximate LUFS)
    rms = np.sqrt(np.mean(audio_data**2))
    lufs_approx = 20 * np.log10(rms + 1e-10) + 3.0  # Rough LUFS approximation
    features.append(lufs_approx)
    
    # 2. Spectral features
    stft = librosa.stft(audio_data)
    magnitude = np.abs(stft)
    
    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate)[0]
    features.append(np.mean(spectral_centroid))
    
    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sample_rate)[0]
    features.append(np.mean(spectral_rolloff))
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
    features.append(np.mean(zcr))
    
    # 3. Dynamic range
    dynamic_range = np.max(audio_data) - np.min(audio_data)
    features.append(dynamic_range)
    
    # 4. Stereo width (if stereo input)
    if audio_data.ndim > 1 and audio_data.shape[0] == 2:
        # Mid-side processing
        mid = (audio_data[0] + audio_data[1]) / 2
        side = (audio_data[0] - audio_data[1]) / 2
        stereo_width = np.std(side) / (np.std(mid) + 1e-10)
        features.append(stereo_width)
    else:
        features.append(0.0)  # Mono
    
    # 5. Additional spectral features
    spectral_flux = np.mean(np.diff(magnitude, axis=1)**2)
    features.append(spectral_flux)
    
    # 6. Harmonic-percussive separation features
    harmonic, percussive = librosa.effects.hpss(audio_data)
    harmonic_ratio = np.mean(harmonic**2) / (np.mean(audio_data**2) + 1e-10)
    features.append(harmonic_ratio)
    
    # Pad to fixed size (32 features)
    while len(features) < 32:
        features.append(0.0)
    
    return np.array(features[:32], dtype=np.float32)

if __name__ == "__main__":
    # Example usage
    vocab_size = 1000
    critic = ComprehensiveCritic(vocab_size)
    
    # Example inputs
    batch_size = 2
    prompts = ["upbeat pop song", "dramatic ballad"]
    controls = [
        {'style': 'pop', 'bpm': 120, 'key': 'C'},
        {'style': 'ballad', 'bpm': 80, 'key': 'Am'}
    ]
    
    tokens = torch.randint(0, vocab_size, (batch_size, 64))
    mel_specs = torch.randn(batch_size, 1, 128, 128)  # Mel spectrograms
    ref_embeddings = torch.randn(batch_size, 512)  # Reference style embeddings
    mix_features = torch.randn(batch_size, 32)  # Mix analysis features
    
    # Evaluate
    scores = critic.evaluate_comprehensive(
        prompts, controls, tokens, mel_specs, ref_embeddings, mix_features
    )
    
    for i, score in enumerate(scores):
        print(f"\nSample {i+1}:")
        print(f"Overall: {score.overall:.3f}")
        print(f"Adherence: {score.adherence:.3f}")
        print(f"Style Match: {score.style_match:.3f}")
        print(f"Mix Quality: {score.mix_quality:.3f}")
        print(f"Confidence: {score.confidence:.3f}")
        print(f"Notes: {', '.join(score.notes)}")