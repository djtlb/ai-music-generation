"""
Utility functions for audio analysis and mixing target computation.
"""

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Tuple, List


def compute_lufs(audio: torch.Tensor, sample_rate: int = 48000) -> float:
    """
    Compute LUFS (Loudness Units relative to Full Scale) using ITU-R BS.1770-4.
    
    This is a simplified implementation. For production use, consider using
    a proper LUFS library like python-audio-tools or pyloudnorm.
    
    Args:
        audio: Stereo audio tensor [2, samples]
        sample_rate: Sample rate in Hz
        
    Returns:
        LUFS value in dB
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
        
    # Pre-filter (high-pass at ~38Hz and high-frequency shelving)
    # Simplified version - in practice use proper filter coefficients
    
    # K-weighting filter approximation
    # This should implement the full ITU-R BS.1770-4 filters
    # For now, we use a simplified RMS-based approximation
    
    # Convert to mono using ITU weightings (L and R channels)
    if audio.shape[0] == 2:
        left, right = audio[0], audio[1]
        # ITU weighting: mono = (L + R) / 2
        mono = (left + right) / 2
    else:
        mono = audio[0]
        
    # Apply K-weighting (simplified)
    # Real implementation would use proper biquad filters
    
    # Gate the signal: only consider blocks above -70 LUFS
    block_size = int(0.4 * sample_rate)  # 400ms blocks
    overlap = int(0.1 * sample_rate)  # 100ms overlap
    
    blocks = []
    for i in range(0, len(mono) - block_size, overlap):
        block = mono[i:i + block_size]
        block_power = torch.mean(block ** 2)
        
        # Convert to LUFS (approximate)
        block_lufs = -0.691 + 10 * torch.log10(block_power + 1e-10)
        
        # Gate: only include blocks above -70 LUFS
        if block_lufs > -70:
            blocks.append(block_power)
            
    if len(blocks) == 0:
        return -70.0  # Very quiet signal
        
    # Compute gated loudness
    mean_power = torch.mean(torch.stack(blocks))
    lufs = -0.691 + 10 * torch.log10(mean_power + 1e-10)
    
    return float(lufs)


def compute_spectral_centroid(audio: torch.Tensor, sample_rate: int = 48000) -> float:
    """
    Compute spectral centroid in Hz.
    
    Args:
        audio: Audio tensor [channels, samples]
        sample_rate: Sample rate in Hz
        
    Returns:
        Spectral centroid in Hz
    """
    if audio.dim() == 2:
        audio = audio.mean(dim=0)  # Convert to mono
        
    # Compute STFT
    stft = torch.stft(
        audio,
        n_fft=2048,
        hop_length=512,
        window=torch.hann_window(2048),
        return_complex=True
    )
    
    magnitude = torch.abs(stft)
    
    # Frequency bins
    freqs = torch.linspace(0, sample_rate / 2, magnitude.shape[0])
    
    # Compute weighted average frequency
    total_magnitude = torch.sum(magnitude, dim=0)  # Sum over time
    total_magnitude = torch.sum(total_magnitude)   # Sum over frequency
    
    if total_magnitude == 0:
        return 0.0
        
    weighted_freq = torch.sum(
        freqs.unsqueeze(1) * magnitude, dim=0
    )  # Weight by magnitude
    weighted_freq = torch.sum(weighted_freq)  # Sum over time
    
    centroid = weighted_freq / total_magnitude
    
    return float(centroid)


def compute_stereo_ms_ratio(audio: torch.Tensor) -> float:
    """
    Compute stereo Mid/Side ratio.
    
    Args:
        audio: Stereo audio tensor [2, samples]
        
    Returns:
        M/S ratio (mid energy / side energy)
    """
    if audio.shape[0] != 2:
        return 1.0  # Mono signal
        
    left, right = audio[0], audio[1]
    
    # Convert to M/S
    mid = (left + right) / 2
    side = (left - right) / 2
    
    # Compute RMS energy
    mid_energy = torch.mean(mid ** 2)
    side_energy = torch.mean(side ** 2)
    
    if side_energy == 0:
        return float('inf')  # Pure mono
        
    ms_ratio = mid_energy / side_energy
    return float(ms_ratio)


def load_style_targets(config_path: str = None) -> Dict:
    """
    Load mixing targets for different styles.
    
    Args:
        config_path: Path to style targets YAML file
        
    Returns:
        Dictionary of style targets
    """
    if config_path is None:
        # Try to load from default config path
        default_config = Path(__file__).parent.parent / "configs" / "style_targets.yaml"
        if default_config.exists():
            try:
                with open(default_config, 'r') as f:
                    config_data = yaml.safe_load(f)
                    # Extract style_targets section if it exists
                    if 'style_targets' in config_data:
                        return config_data['style_targets']
                    else:
                        return config_data
            except Exception:
                pass  # Fall back to hardcoded defaults
        
        # Use hardcoded defaults if config not found
        return {
            'rock_punk': {
                'lufs': -9.5,
                'spectral_centroid_hz': 2800,
                'stereo_ms_ratio': 0.6
            },
            'rnb_ballad': {
                'lufs': -12.0,
                'spectral_centroid_hz': 1800,
                'stereo_ms_ratio': 0.8
            },
            'country_pop': {
                'lufs': -10.5,
                'spectral_centroid_hz': 2200,
                'stereo_ms_ratio': 0.7
            }
        }
        
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Style targets config not found: {config_path}")
        
    with open(config_path, 'r') as f:
        targets = yaml.safe_load(f)
        
    # Extract style_targets section if it exists
    if 'style_targets' in targets:
        return targets['style_targets']
    else:
        return targets


def create_white_noise_stems(n_stems: int = 4, duration: float = 10.0, 
                           sample_rate: int = 48000) -> List[torch.Tensor]:
    """
    Create white noise test stems for validation.
    
    Args:
        n_stems: Number of stems to create
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        List of stereo audio tensors
    """
    n_samples = int(duration * sample_rate)
    stems = []
    
    for i in range(n_stems):
        # Create noise with different characteristics
        noise = torch.randn(2, n_samples) * 0.1  # -20dB noise
        
        # Apply different filtering to each stem
        if i % 4 == 0:  # Bass-heavy
            # Low-pass filter approximation
            noise = noise * 0.5  # Quieter
        elif i % 4 == 1:  # Mid-range
            # Band-pass approximation
            noise = noise * 0.7
        elif i % 4 == 2:  # High-frequency
            # High-pass approximation
            noise = noise * 0.3
        else:  # Full-range
            noise = noise * 0.6
            
        stems.append(noise)
        
    return stems


def validate_lufs_accuracy(predicted_lufs: float, target_lufs: float, 
                          tolerance: float = 1.0) -> bool:
    """
    Validate LUFS prediction accuracy.
    
    Args:
        predicted_lufs: Predicted LUFS value
        target_lufs: Target LUFS value
        tolerance: Acceptable tolerance in dB
        
    Returns:
        True if within tolerance
    """
    error = abs(predicted_lufs - target_lufs)
    return error <= tolerance


def validate_spectral_accuracy(predicted_centroid: float, target_centroid: float,
                             tolerance: float = 200.0) -> bool:
    """
    Validate spectral centroid accuracy.
    
    Args:
        predicted_centroid: Predicted centroid in Hz
        target_centroid: Target centroid in Hz
        tolerance: Acceptable tolerance in Hz
        
    Returns:
        True if within tolerance
    """
    error = abs(predicted_centroid - target_centroid)
    return error <= tolerance


def compute_mix_quality_score(analysis: Dict, style: str, targets: Dict) -> float:
    """
    Compute overall mix quality score based on target adherence.
    
    Args:
        analysis: Mix analysis results
        style: Style name
        targets: Style targets dictionary
        
    Returns:
        Quality score (0-1, higher is better)
    """
    if style not in targets:
        return 0.5  # Neutral score
        
    style_targets = targets[style]
    scores = []
    
    # LUFS score
    if 'lufs' in style_targets:
        lufs_error = abs(analysis['lufs'] - style_targets['lufs'])
        lufs_score = max(0, 1 - lufs_error / 5.0)  # 5dB tolerance for full score
        scores.append(lufs_score)
        
    # Spectral centroid score
    if 'spectral_centroid_hz' in style_targets:
        centroid_error = abs(analysis['spectral_centroid'] - style_targets['spectral_centroid_hz'])
        centroid_score = max(0, 1 - centroid_error / 1000.0)  # 1kHz tolerance
        scores.append(centroid_score)
        
    # Stereo ratio score
    if 'stereo_ms_ratio' in style_targets:
        ms_error = abs(analysis['stereo_ms_ratio'] - style_targets['stereo_ms_ratio'])
        ms_score = max(0, 1 - ms_error / 0.5)  # 0.5 ratio tolerance
        scores.append(ms_score)
        
    return sum(scores) / len(scores) if scores else 0.5


def export_mix_analysis(analysis: Dict, output_path: str):
    """
    Export mix analysis to JSON file.
    
    Args:
        analysis: Analysis dictionary
        output_path: Output file path
    """
    import json
    
    # Convert torch tensors to float if present
    exportable_analysis = {}
    for key, value in analysis.items():
        if isinstance(value, torch.Tensor):
            exportable_analysis[key] = float(value)
        else:
            exportable_analysis[key] = value
            
    with open(output_path, 'w') as f:
        json.dump(exportable_analysis, f, indent=2)


def load_reference_mix(reference_path: str) -> torch.Tensor:
    """
    Load reference mix for comparison.
    
    Args:
        reference_path: Path to reference audio file
        
    Returns:
        Reference audio tensor
    """
    try:
        audio, sample_rate = torchaudio.load(reference_path)
        return audio, sample_rate
    except Exception as e:
        raise RuntimeError(f"Failed to load reference mix: {e}")


def compute_mix_similarity(mix1: torch.Tensor, mix2: torch.Tensor, 
                          sample_rate: int = 48000) -> Dict[str, float]:
    """
    Compute similarity metrics between two mixes.
    
    Args:
        mix1: First mix tensor
        mix2: Second mix tensor
        sample_rate: Sample rate
        
    Returns:
        Dictionary of similarity metrics
    """
    # Ensure same length
    min_length = min(mix1.shape[-1], mix2.shape[-1])
    mix1 = mix1[..., :min_length]
    mix2 = mix2[..., :min_length]
    
    similarity = {}
    
    # LUFS similarity
    lufs1 = compute_lufs(mix1, sample_rate)
    lufs2 = compute_lufs(mix2, sample_rate)
    similarity['lufs_difference'] = abs(lufs1 - lufs2)
    
    # Spectral centroid similarity
    centroid1 = compute_spectral_centroid(mix1, sample_rate)
    centroid2 = compute_spectral_centroid(mix2, sample_rate)
    similarity['centroid_difference'] = abs(centroid1 - centroid2)
    
    # Stereo ratio similarity
    ms1 = compute_stereo_ms_ratio(mix1)
    ms2 = compute_stereo_ms_ratio(mix2)
    similarity['ms_ratio_difference'] = abs(ms1 - ms2)
    
    # Overall similarity score (0-1, higher is more similar)
    lufs_sim = max(0, 1 - similarity['lufs_difference'] / 10.0)
    centroid_sim = max(0, 1 - similarity['centroid_difference'] / 2000.0)
    ms_sim = max(0, 1 - similarity['ms_ratio_difference'] / 1.0)
    
    similarity['overall_similarity'] = (lufs_sim + centroid_sim + ms_sim) / 3
    
    return similarity