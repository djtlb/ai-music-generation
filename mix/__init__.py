"""
Differentiable mixing and mastering chain for AI music production.

This module provides PyTorch-based components for automatic mixing
with target LUFS, spectral centroid, and stereo image control.
"""

from .auto_mix import (
    AutoMixChain,
    StemFeatureExtractor,
    MixingParameterPredictor,
    ChannelStrip,
    MasteringChain
)
from .utils import (
    compute_lufs,
    compute_spectral_centroid,
    compute_stereo_ms_ratio,
    load_style_targets
)

__all__ = [
    'AutoMixChain',
    'StemFeatureExtractor', 
    'MixingParameterPredictor',
    'ChannelStrip',
    'MasteringChain',
    'compute_lufs',
    'compute_spectral_centroid',
    'compute_stereo_ms_ratio',
    'load_style_targets'
]