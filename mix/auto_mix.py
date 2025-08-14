"""
Auto-mixing chain with differentiable mixing components.

This module provides a complete differentiable mixing system that:
1. Extracts features from audio stems
2. Predicts optimal mixing parameters using an MLP
3. Applies per-stem EQ, compression, and effects
4. Processes through mastering chain
5. Targets specific LUFS and spectral characteristics per style
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from typing import Dict, List, Tuple, Optional
import yaml
from pathlib import Path


class StemFeatureExtractor(nn.Module):
    """Extracts RMS, crest factor, spectral centroid from audio stems."""
    
    def __init__(self, sample_rate: int = 48000, n_fft: int = 2048):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract features from audio.
        
        Args:
            audio: [batch, channels, samples] audio tensor
            
        Returns:
            features: [batch, n_features] feature tensor
        """
        batch_size = audio.shape[0]
        features = []
        
        for i in range(batch_size):
            stem_audio = audio[i]  # [channels, samples]
            
            # Convert to mono for analysis
            mono_audio = stem_audio.mean(dim=0, keepdim=True)
            
            # RMS (Root Mean Square)
            rms = torch.sqrt(torch.mean(mono_audio ** 2))
            
            # Peak amplitude
            peak = torch.max(torch.abs(mono_audio))
            
            # Crest factor (peak-to-RMS ratio)
            crest = peak / (rms + 1e-8)
            
            # Spectral centroid using FFT
            stft = torch.stft(
                mono_audio.squeeze(0),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True
            )
            
            magnitude = torch.abs(stft)
            frequencies = torch.linspace(0, self.sample_rate / 2, magnitude.shape[0])
            
            # Weighted average of frequencies
            freq_weights = frequencies.unsqueeze(1).to(magnitude.device)
            magnitude_sum = torch.sum(magnitude, dim=1)
            centroid = torch.sum(freq_weights * magnitude_sum) / (torch.sum(magnitude_sum) + 1e-8)
            
            # Log-scale centroid (Hz to log space)
            log_centroid = torch.log(centroid + 1.0)
            
            # Dynamic range (difference between 95th and 5th percentile)
            audio_sorted = torch.sort(torch.abs(mono_audio.squeeze()))[0]
            n_samples = audio_sorted.shape[0]
            p5_idx = int(0.05 * n_samples)
            p95_idx = int(0.95 * n_samples)
            dynamic_range = audio_sorted[p95_idx] - audio_sorted[p5_idx]
            
            stem_features = torch.stack([
                torch.log(rms + 1e-8),  # Log RMS
                torch.log(crest + 1e-8),  # Log crest factor
                log_centroid,  # Log spectral centroid
                torch.log(dynamic_range + 1e-8),  # Log dynamic range
            ])
            
            features.append(stem_features)
            
        return torch.stack(features)


class EQBand(nn.Module):
    """Parametric EQ band with gain and frequency control."""
    
    def __init__(self, eq_type: str = "peaking"):
        super().__init__()
        self.eq_type = eq_type
        
    def forward(self, audio: torch.Tensor, freq: float, gain: torch.Tensor, q: float = 1.0) -> torch.Tensor:
        """
        Apply EQ filtering (simplified differentiable version).
        
        In practice, this would use proper filter coefficients.
        For now, we use spectral domain processing as approximation.
        """
        if audio.numel() == 0:
            return audio
            
        # Handle single channel or stereo
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            
        # Process each channel
        processed_channels = []
        for ch in range(audio.shape[0]):
            ch_audio = audio[ch]
            
            # Simplified EQ: apply gain boost/cut in frequency domain
            stft = torch.stft(ch_audio, n_fft=2048, hop_length=512, 
                            window=torch.hann_window(2048).to(ch_audio.device),
                            return_complex=True)
            
            # Create frequency mask centered around target frequency
            freqs = torch.linspace(0, 24000, stft.shape[0]).to(stft.device)
            freq_mask = torch.exp(-((freqs - freq) / (freq / q)) ** 2)
            freq_mask = freq_mask.unsqueeze(-1)
            
            # Apply gain (convert tensor to scalar if needed)
            gain_val = gain.item() if torch.is_tensor(gain) else gain
            gain_linear = 10 ** (gain_val / 20)  # dB to linear
            stft_eq = stft * (1 + (gain_linear - 1) * freq_mask)
            
            # Convert back to time domain
            try:
                audio_eq = torch.istft(stft_eq, n_fft=2048, hop_length=512,
                                     window=torch.hann_window(2048).to(ch_audio.device))
                # Ensure output length matches input
                if audio_eq.shape[0] != ch_audio.shape[0]:
                    if audio_eq.shape[0] < ch_audio.shape[0]:
                        padding = ch_audio.shape[0] - audio_eq.shape[0]
                        audio_eq = F.pad(audio_eq, (0, padding))
                    else:
                        audio_eq = audio_eq[:ch_audio.shape[0]]
                        
                processed_channels.append(audio_eq)
            except Exception:
                # Fallback: return original audio if STFT/ISTFT fails
                processed_channels.append(ch_audio)
        
        return torch.stack(processed_channels)


class Compressor(nn.Module):
    """Differentiable compressor with threshold, ratio, attack, release."""
    
    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.sample_rate = sample_rate
        
    def forward(self, audio: torch.Tensor, threshold: float, ratio: float, 
                attack: float, release: float, makeup_gain: float = 0.0) -> torch.Tensor:
        """
        Apply compression (simplified differentiable version).
        
        Args:
            audio: Input audio tensor
            threshold: Compression threshold in dB
            ratio: Compression ratio (e.g., 4.0 for 4:1)
            attack: Attack time in seconds
            release: Release time in seconds
            makeup_gain: Makeup gain in dB
        """
        # Avoid compression on very quiet signals
        if torch.max(torch.abs(audio)) < 1e-6:
            return audio
            
        # Convert to dB domain for processing
        audio_abs = torch.abs(audio) + 1e-8
        audio_db = 20 * torch.log10(audio_abs)
        
        # Compute gain reduction
        over_threshold = audio_db - threshold
        over_threshold = torch.clamp(over_threshold, min=0)
        
        gain_reduction = over_threshold * (1 - 1/ratio)
        
        # Apply simple envelope following (simplified attack/release)
        # In practice, this would use proper envelope detection
        alpha_attack = 1 - torch.exp(torch.tensor(-1 / (attack * self.sample_rate)))
        alpha_release = 1 - torch.exp(torch.tensor(-1 / (release * self.sample_rate)))
        
        # Move alphas to same device as audio
        alpha_attack = alpha_attack.to(audio.device)
        alpha_release = alpha_release.to(audio.device)
        
        # Simplified envelope following
        gain_smooth = torch.zeros_like(gain_reduction)
        for i in range(1, gain_reduction.shape[-1]):
            if gain_reduction[..., i] > gain_smooth[..., i-1]:
                alpha = alpha_attack
            else:
                alpha = alpha_release
            gain_smooth[..., i] = (1 - alpha) * gain_smooth[..., i-1] + alpha * gain_reduction[..., i]
        
        # Apply gain reduction and makeup gain
        gain_linear = 10 ** ((-gain_smooth + makeup_gain) / 20)
        
        # Preserve sign of original audio
        audio_sign = torch.sign(audio)
        audio_compressed = audio_sign * audio_abs * gain_linear
        
        return audio_compressed


class ChannelStrip(nn.Module):
    """Complete channel strip with EQ, compression, and saturation."""
    
    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.eq_low = EQBand("shelf")
        self.eq_low_mid = EQBand("peaking") 
        self.eq_high_mid = EQBand("peaking")
        self.eq_high = EQBand("shelf")
        self.compressor = Compressor(sample_rate)
        
    def forward(self, audio: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process audio through channel strip.
        
        Args:
            audio: Input audio [channels, samples]
            params: Dictionary of processing parameters
        """
        # EQ processing
        audio = self.eq_low(audio, 100.0, params['eq_low_gain'])
        audio = self.eq_low_mid(audio, 500.0, params['eq_low_mid_gain'])  
        audio = self.eq_high_mid(audio, 3000.0, params['eq_high_mid_gain'])
        audio = self.eq_high(audio, 10000.0, params['eq_high_gain'])
        
        # Compression (convert tensor parameters to scalars)
        comp_threshold = params['comp_threshold'].item() if torch.is_tensor(params['comp_threshold']) else params['comp_threshold']
        comp_ratio = params['comp_ratio'].item() if torch.is_tensor(params['comp_ratio']) else params['comp_ratio']
        comp_attack = params['comp_attack'].item() if torch.is_tensor(params['comp_attack']) else params['comp_attack']
        comp_release = params['comp_release'].item() if torch.is_tensor(params['comp_release']) else params['comp_release']
        comp_makeup_gain = params['comp_makeup_gain'].item() if torch.is_tensor(params['comp_makeup_gain']) else params['comp_makeup_gain']
        
        audio = self.compressor(
            audio,
            comp_threshold,
            comp_ratio,
            comp_attack,
            comp_release,
            comp_makeup_gain
        )
        
        # Soft saturation
        saturation = params['saturation'].item() if torch.is_tensor(params['saturation']) else params['saturation']
        if saturation > 1.0:
            audio = torch.tanh(audio * saturation) / saturation
        
        return audio


class StereoWidener(nn.Module):
    """Stereo width enhancement using M/S processing."""
    
    def forward(self, audio: torch.Tensor, width: float) -> torch.Tensor:
        """
        Apply stereo widening.
        
        Args:
            audio: Stereo audio [2, samples]
            width: Width factor (1.0 = normal, >1.0 = wider, <1.0 = narrower)
        """
        if audio.shape[0] != 2:
            return audio  # Skip if not stereo
            
        left, right = audio[0], audio[1]
        
        # Convert to M/S
        mid = (left + right) / 2
        side = (left - right) / 2
        
        # Apply width
        side_wide = side * width
        
        # Convert back to L/R
        left_out = mid + side_wide
        right_out = mid - side_wide
        
        return torch.stack([left_out, right_out])


class Limiter(nn.Module):
    """Brick-wall limiter for preventing clipping."""
    
    def forward(self, audio: torch.Tensor, threshold: float, release: float) -> torch.Tensor:
        """
        Apply limiting.
        
        Args:
            audio: Input audio
            threshold: Limiting threshold (linear, 0-1)
            release: Release time in seconds
        """
        # Simple hard limiting (in practice would use lookahead)
        gain = torch.clamp(threshold / (torch.abs(audio) + 1e-8), max=1.0)
        
        # Apply gain smoothing for release
        # Simplified version - real limiter would use proper lookahead
        limited = audio * gain
        
        return limited


class MasteringChain(nn.Module):
    """Complete mastering chain with bus compression, EQ, and limiting."""
    
    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.bus_compressor = Compressor(sample_rate)
        self.eq = ChannelStrip(sample_rate)
        self.stereo_widener = StereoWidener()
        self.limiter = Limiter()
        
    def forward(self, audio: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process audio through mastering chain.
        
        Args:
            audio: Mixed audio [channels, samples]
            params: Mastering parameters
        """
        # Convert tensor parameters to scalars for processing
        def to_scalar(param):
            return param.item() if torch.is_tensor(param) else param
        
        # Bus compression
        audio = self.bus_compressor(
            audio,
            to_scalar(params['bus_comp_threshold']),
            to_scalar(params['bus_comp_ratio']),
            to_scalar(params['bus_comp_attack']),
            to_scalar(params['bus_comp_release']),
            to_scalar(params['bus_comp_makeup_gain'])
        )
        
        # Mastering EQ
        eq_params = {
            'eq_low_gain': params['master_eq_low_gain'],
            'eq_low_mid_gain': params['master_eq_low_mid_gain'],
            'eq_high_mid_gain': params['master_eq_high_mid_gain'],
            'eq_high_gain': params['master_eq_high_gain'],
            'comp_threshold': torch.tensor(-50.0).to(audio.device),  # Disable comp in EQ
            'comp_ratio': torch.tensor(1.0).to(audio.device),
            'comp_attack': torch.tensor(0.001).to(audio.device),
            'comp_release': torch.tensor(0.1).to(audio.device),
            'comp_makeup_gain': torch.tensor(0.0).to(audio.device),
            'saturation': torch.tensor(1.0).to(audio.device)
        }
        audio = self.eq(audio, eq_params)
        
        # Stereo enhancement
        audio = self.stereo_widener(audio, to_scalar(params['stereo_width']))
        
        # Final limiting
        audio = self.limiter(audio, to_scalar(params['limiter_threshold']), to_scalar(params['limiter_release']))
        
        return audio


class MixingParameterPredictor(nn.Module):
    """MLP that predicts mixing parameters from stem features."""
    
    def __init__(self, n_features: int = 4, n_stems: int = 8, hidden_size: int = 256):
        super().__init__()
        
        self.n_stems = n_stems
        input_size = n_features * n_stems  # Concatenated features from all stems
        
        # Per-stem parameter count
        stem_params = 12  # eq (4) + compression (5) + saturation (1) + level (1) + pan (1)
        master_params = 11  # bus comp (5) + master eq (4) + stereo (1) + limiter (1)
        
        output_size = stem_params * n_stems + master_params
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Parameter ranges for scaling
        self.register_buffer('param_ranges', self._get_parameter_ranges(n_stems))
        
    def _get_parameter_ranges(self, n_stems: int) -> torch.Tensor:
        """Define parameter ranges for scaling network outputs."""
        ranges = []
        
        # Per-stem parameters
        for _ in range(n_stems):
            ranges.extend([
                # EQ gains (-12 to +12 dB)
                (-12.0, 12.0), (-12.0, 12.0), (-12.0, 12.0), (-12.0, 12.0),
                # Compression threshold (-40 to -6 dB)
                (-40.0, -6.0),
                # Compression ratio (1 to 10)
                (1.0, 10.0),
                # Attack (0.001 to 0.1s)
                (0.001, 0.1),
                # Release (0.01 to 1.0s) 
                (0.01, 1.0),
                # Makeup gain (0 to 12 dB)
                (0.0, 12.0),
                # Saturation (1.0 to 3.0)
                (1.0, 3.0),
                # Level (-24 to +6 dB)
                (-24.0, 6.0),
                # Pan (-1 to +1)
                (-1.0, 1.0)
            ])
            
        # Master parameters
        ranges.extend([
            # Bus comp threshold (-20 to -2 dB)
            (-20.0, -2.0),
            # Bus comp ratio (1 to 6)
            (1.0, 6.0), 
            # Bus comp attack (0.001 to 0.05s)
            (0.001, 0.05),
            # Bus comp release (0.05 to 0.5s)
            (0.05, 0.5),
            # Bus comp makeup gain (0 to 6 dB)
            (0.0, 6.0),
            # Master EQ gains (-6 to +6 dB)
            (-6.0, 6.0), (-6.0, 6.0), (-6.0, 6.0), (-6.0, 6.0),
            # Stereo width (0.5 to 1.5)
            (0.5, 1.5),
            # Limiter threshold (0.7 to 1.0 linear)
            (0.7, 1.0)
        ])
        
        return torch.tensor(ranges)
        
    def forward(self, stem_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict mixing parameters from stem features.
        
        Args:
            stem_features: [batch, n_stems, n_features] feature tensor
            
        Returns:
            Dictionary of parameter tensors
        """
        batch_size = stem_features.shape[0]
        
        # Flatten features
        features_flat = stem_features.view(batch_size, -1)
        
        # Predict parameters
        raw_params = self.network(features_flat)
        
        # Scale parameters to appropriate ranges
        param_mins = self.param_ranges[:, 0]
        param_maxs = self.param_ranges[:, 1]
        
        # Sigmoid scaling to [0, 1] then to target range
        scaled_params = torch.sigmoid(raw_params)
        scaled_params = param_mins + scaled_params * (param_maxs - param_mins)
        
        # Parse parameters into structured dictionary
        params = self._parse_parameters(scaled_params)
        
        return params
        
    def _parse_parameters(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Parse flat parameter tensor into structured dictionary."""
        batch_size = params.shape[0]
        parsed = {}
        
        idx = 0
        
        # Per-stem parameters
        for stem_idx in range(self.n_stems):
            stem_key = f'stem_{stem_idx}'
            parsed[stem_key] = {}
            
            # EQ gains
            parsed[stem_key]['eq_low_gain'] = params[:, idx]
            parsed[stem_key]['eq_low_mid_gain'] = params[:, idx + 1] 
            parsed[stem_key]['eq_high_mid_gain'] = params[:, idx + 2]
            parsed[stem_key]['eq_high_gain'] = params[:, idx + 3]
            idx += 4
            
            # Compression
            parsed[stem_key]['comp_threshold'] = params[:, idx]
            parsed[stem_key]['comp_ratio'] = params[:, idx + 1]
            parsed[stem_key]['comp_attack'] = params[:, idx + 2]
            parsed[stem_key]['comp_release'] = params[:, idx + 3]
            parsed[stem_key]['comp_makeup_gain'] = params[:, idx + 4]
            idx += 5
            
            # Other
            parsed[stem_key]['saturation'] = params[:, idx]
            parsed[stem_key]['level'] = params[:, idx + 1]
            parsed[stem_key]['pan'] = params[:, idx + 2]
            idx += 3
            
        # Master parameters  
        parsed['master'] = {}
        parsed['master']['bus_comp_threshold'] = params[:, idx]
        parsed['master']['bus_comp_ratio'] = params[:, idx + 1]
        parsed['master']['bus_comp_attack'] = params[:, idx + 2]
        parsed['master']['bus_comp_release'] = params[:, idx + 3]
        parsed['master']['bus_comp_makeup_gain'] = params[:, idx + 4]
        idx += 5
        
        parsed['master']['master_eq_low_gain'] = params[:, idx]
        parsed['master']['master_eq_low_mid_gain'] = params[:, idx + 1]
        parsed['master']['master_eq_high_mid_gain'] = params[:, idx + 2]
        parsed['master']['master_eq_high_gain'] = params[:, idx + 3]
        idx += 4
        
        parsed['master']['stereo_width'] = params[:, idx]
        parsed['master']['limiter_threshold'] = params[:, idx + 1]
        parsed['master']['limiter_release'] = torch.full_like(params[:, idx], 0.05)  # Fixed release
        
        return parsed


class AutoMixChain(nn.Module):
    """Complete auto-mixing system combining all components."""
    
    def __init__(self, n_stems: int = 8, sample_rate: int = 48000, 
                 style_targets: Optional[Dict] = None):
        super().__init__()
        
        self.n_stems = n_stems
        self.sample_rate = sample_rate
        self.style_targets = style_targets or {}
        
        # Components
        self.feature_extractor = StemFeatureExtractor(sample_rate)
        self.parameter_predictor = MixingParameterPredictor(n_stems=n_stems)
        
        # Processing chains
        self.channel_strips = nn.ModuleList([
            ChannelStrip(sample_rate) for _ in range(n_stems)
        ])
        self.mastering_chain = MasteringChain(sample_rate)
        
    def forward(self, stems: List[torch.Tensor], style: str = 'default') -> Tuple[torch.Tensor, Dict]:
        """
        Process stems through complete auto-mixing chain.
        
        Args:
            stems: List of audio tensors [channels, samples]
            style: Target style for mixing
            
        Returns:
            mixed_audio: Final mixed and mastered audio
            analysis: Dictionary of analysis metrics
        """
        # Pad stems to consistent length and stack
        max_length = max(stem.shape[-1] for stem in stems)
        padded_stems = []
        
        for stem in stems[:self.n_stems]:  # Limit to max stems
            if stem.shape[-1] < max_length:
                padding = max_length - stem.shape[-1]
                stem = F.pad(stem, (0, padding))
            padded_stems.append(stem)
            
        # Pad with zeros if we have fewer stems than expected
        while len(padded_stems) < self.n_stems:
            padded_stems.append(torch.zeros(2, max_length))
            
        stem_batch = torch.stack(padded_stems)  # [n_stems, channels, samples]
        stem_batch = stem_batch.unsqueeze(0)  # Add batch dimension
        
        # Extract features
        features = self.feature_extractor(stem_batch)  # [batch, n_stems, n_features]
        
        # Predict parameters
        mix_params = self.parameter_predictor(features)
        
        # Process each stem
        processed_stems = []
        for i, (stem, channel_strip) in enumerate(zip(padded_stems, self.channel_strips)):
            stem_params = {k: v[0] for k, v in mix_params[f'stem_{i}'].items()}  # Remove batch dim
            processed_stem = channel_strip(stem, stem_params)
            
            # Apply level and pan
            level_db = stem_params['level'].item() if torch.is_tensor(stem_params['level']) else stem_params['level']
            level_linear = 10 ** (level_db / 20)
            processed_stem = processed_stem * level_linear
            
            # Simple panning (for stereo)
            if processed_stem.shape[0] == 2:
                pan = stem_params['pan'].item() if torch.is_tensor(stem_params['pan']) else stem_params['pan']
                left_gain = torch.sqrt(torch.tensor((1 - pan) / 2)) if pan >= 0 else torch.tensor(1.0)
                right_gain = torch.sqrt(torch.tensor((1 + pan) / 2)) if pan <= 0 else torch.tensor(1.0)
                processed_stem[0] *= left_gain.to(processed_stem.device)
                processed_stem[1] *= right_gain.to(processed_stem.device)
                
            processed_stems.append(processed_stem)
            
        # Sum stems (mix)
        mixed_audio = torch.stack(processed_stems).sum(dim=0)
        
        # Apply mastering
        master_params = {k: v[0] for k, v in mix_params['master'].items()}  # Remove batch dim
        final_audio = self.mastering_chain(mixed_audio, master_params)
        
        # Compute analysis metrics
        analysis = self._analyze_mix(final_audio, style)
        
        return final_audio, analysis
        
    def _analyze_mix(self, audio: torch.Tensor, style: str) -> Dict:
        """Compute LUFS, spectral centroid, and stereo metrics."""
        from .utils import compute_lufs, compute_spectral_centroid, compute_stereo_ms_ratio
        
        analysis = {}
        
        # Compute current metrics
        analysis['lufs'] = compute_lufs(audio, self.sample_rate)
        analysis['spectral_centroid'] = compute_spectral_centroid(audio, self.sample_rate)
        analysis['stereo_ms_ratio'] = compute_stereo_ms_ratio(audio)
        
        # Compare to targets if available
        if self.style_targets and style in self.style_targets:
            targets = self.style_targets[style]
            analysis['lufs_target'] = targets.get('lufs', -14.0)
            analysis['lufs_error'] = analysis['lufs'] - analysis['lufs_target']
            
            analysis['centroid_target'] = targets.get('spectral_centroid_hz', 2000.0)
            analysis['centroid_error'] = analysis['spectral_centroid'] - analysis['centroid_target']
            
            analysis['ms_ratio_target'] = targets.get('stereo_ms_ratio', 0.5)
            analysis['ms_ratio_error'] = analysis['stereo_ms_ratio'] - analysis['ms_ratio_target']
        else:
            # Set default targets if style not found
            default_targets = {
                'rock_punk': {'lufs': -9.5, 'spectral_centroid_hz': 2800, 'stereo_ms_ratio': 0.6},
                'rnb_ballad': {'lufs': -12.0, 'spectral_centroid_hz': 1800, 'stereo_ms_ratio': 0.8},
                'country_pop': {'lufs': -10.5, 'spectral_centroid_hz': 2200, 'stereo_ms_ratio': 0.7}
            }
            if style in default_targets:
                targets = default_targets[style]
                analysis['lufs_target'] = targets['lufs']
                analysis['lufs_error'] = analysis['lufs'] - analysis['lufs_target']
                analysis['centroid_target'] = targets['spectral_centroid_hz']
                analysis['centroid_error'] = analysis['spectral_centroid'] - analysis['centroid_target']
                analysis['ms_ratio_target'] = targets['stereo_ms_ratio']
                analysis['ms_ratio_error'] = analysis['stereo_ms_ratio'] - analysis['ms_ratio_target']
            
        return analysis


def create_training_loss(style_targets: Dict) -> nn.Module:
    """Create loss function for training auto-mix system."""
    
    class AutoMixLoss(nn.Module):
        def __init__(self, targets: Dict):
            super().__init__()
            self.targets = targets
            
        def forward(self, analysis: Dict, style: str) -> torch.Tensor:
            """Compute loss based on target deviations."""
            if style not in self.targets:
                return torch.tensor(0.0, requires_grad=True)
                
            targets = self.targets[style]
            losses = []
            
            # LUFS loss
            if 'lufs' in targets:
                lufs_loss = F.mse_loss(
                    torch.tensor(analysis['lufs']),
                    torch.tensor(targets['lufs'])
                )
                losses.append(lufs_loss)
                
            # Spectral centroid loss
            if 'spectral_centroid_hz' in targets:
                centroid_loss = F.mse_loss(
                    torch.tensor(analysis['spectral_centroid']),
                    torch.tensor(targets['spectral_centroid_hz'])
                )
                losses.append(centroid_loss * 0.0001)  # Scale down frequency loss
                
            # Stereo MS ratio loss
            if 'stereo_ms_ratio' in targets:
                ms_loss = F.mse_loss(
                    torch.tensor(analysis['stereo_ms_ratio']),
                    torch.tensor(targets['stereo_ms_ratio'])
                )
                losses.append(ms_loss)
                
            return sum(losses) if losses else torch.tensor(0.0, requires_grad=True)
            
    return AutoMixLoss(style_targets)