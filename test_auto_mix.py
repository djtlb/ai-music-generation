#!/usr/bin/env python3
"""
Test suite for the auto-mixing system.

This script provides comprehensive unit and integration tests for the
differentiable mixing chain components.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from mix import (
    AutoMixChain,
    StemFeatureExtractor,
    MixingParameterPredictor,
    ChannelStrip,
    MasteringChain,
    load_style_targets
)
from mix.utils import (
    compute_lufs,
    compute_spectral_centroid,
    compute_stereo_ms_ratio,
    create_white_noise_stems,
    validate_lufs_accuracy,
    validate_spectral_accuracy
)


class TestStemFeatureExtractor(unittest.TestCase):
    """Test cases for StemFeatureExtractor."""
    
    def setUp(self):
        self.extractor = StemFeatureExtractor()
        
    def test_feature_extraction_shape(self):
        """Test that feature extraction returns correct shapes."""
        # Create test audio
        batch_size = 2
        channels = 2
        samples = 48000  # 1 second
        
        audio = torch.randn(batch_size, channels, samples)
        features = self.extractor(audio)
        
        # Should return [batch, n_features]
        self.assertEqual(features.shape, (batch_size, 4))
        
    def test_feature_extraction_mono(self):
        """Test feature extraction with mono audio."""
        audio = torch.randn(1, 1, 48000)
        features = self.extractor(audio)
        
        self.assertEqual(features.shape, (1, 4))
        
    def test_feature_values_reasonable(self):
        """Test that extracted features are in reasonable ranges."""
        # Create quiet sine wave
        t = torch.linspace(0, 1, 48000)
        sine = torch.sin(2 * np.pi * 440 * t) * 0.1  # 440Hz sine at -20dB
        audio = sine.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        features = self.extractor(audio)
        
        # Check that features are finite
        self.assertTrue(torch.isfinite(features).all())
        
        # RMS should be reasonable for sine wave
        log_rms = features[0, 0]  # First feature is log RMS
        self.assertGreater(log_rms, -10)  # Not too quiet
        self.assertLess(log_rms, 0)      # Not too loud


class TestMixingParameterPredictor(unittest.TestCase):
    """Test cases for MixingParameterPredictor."""
    
    def setUp(self):
        self.predictor = MixingParameterPredictor(n_features=4, n_stems=4)
        
    def test_parameter_prediction_shape(self):
        """Test that parameter prediction returns correct structure."""
        batch_size = 2
        n_stems = 4
        n_features = 4
        
        features = torch.randn(batch_size, n_stems, n_features)
        params = self.predictor(features)
        
        # Check structure
        self.assertIn('master', params)
        for i in range(n_stems):
            self.assertIn(f'stem_{i}', params)
            
        # Check parameter shapes
        for i in range(n_stems):
            stem_params = params[f'stem_{i}']
            self.assertEqual(stem_params['eq_low_gain'].shape, (batch_size,))
            self.assertEqual(stem_params['comp_threshold'].shape, (batch_size,))
            
        master_params = params['master']
        self.assertEqual(master_params['bus_comp_threshold'].shape, (batch_size,))
        
    def test_parameter_ranges(self):
        """Test that predicted parameters are in valid ranges."""
        features = torch.randn(1, 4, 4)
        params = self.predictor(features)
        
        # Check EQ gains are reasonable (-12 to +12 dB)
        eq_gain = params['stem_0']['eq_low_gain'][0]
        self.assertGreaterEqual(eq_gain, -12.1)
        self.assertLessEqual(eq_gain, 12.1)
        
        # Check compression ratio is reasonable (1 to 10)
        comp_ratio = params['stem_0']['comp_ratio'][0]
        self.assertGreaterEqual(comp_ratio, 0.9)
        self.assertLessEqual(comp_ratio, 10.1)
        
        # Check limiter threshold (0.7 to 1.0)
        limiter_thresh = params['master']['limiter_threshold'][0]
        self.assertGreaterEqual(limiter_thresh, 0.69)
        self.assertLessEqual(limiter_thresh, 1.01)


class TestChannelStrip(unittest.TestCase):
    """Test cases for ChannelStrip."""
    
    def setUp(self):
        self.channel_strip = ChannelStrip()
        
    def test_channel_strip_processing(self):
        """Test that channel strip processes audio without errors."""
        # Create test audio
        audio = torch.randn(2, 48000) * 0.1  # Stereo, 1 second
        
        # Create test parameters
        params = {
            'eq_low_gain': torch.tensor(0.0),
            'eq_low_mid_gain': torch.tensor(0.0),
            'eq_high_mid_gain': torch.tensor(0.0),
            'eq_high_gain': torch.tensor(0.0),
            'comp_threshold': torch.tensor(-12.0),
            'comp_ratio': torch.tensor(3.0),
            'comp_attack': torch.tensor(0.01),
            'comp_release': torch.tensor(0.1),
            'comp_makeup_gain': torch.tensor(0.0),
            'saturation': torch.tensor(1.0)
        }
        
        processed = self.channel_strip(audio, params)
        
        # Check output shape
        self.assertEqual(processed.shape, audio.shape)
        
        # Check output is finite
        self.assertTrue(torch.isfinite(processed).all())
        
    def test_compression_reduces_dynamics(self):
        """Test that compression actually reduces dynamic range."""
        # Create audio with large dynamic range
        loud_part = torch.ones(24000) * 0.8  # Loud section
        quiet_part = torch.ones(24000) * 0.1  # Quiet section
        audio = torch.stack([
            torch.cat([loud_part, quiet_part]),
            torch.cat([loud_part, quiet_part])
        ])
        
        # Heavy compression
        params = {
            'eq_low_gain': torch.tensor(0.0),
            'eq_low_mid_gain': torch.tensor(0.0),
            'eq_high_mid_gain': torch.tensor(0.0),
            'eq_high_gain': torch.tensor(0.0),
            'comp_threshold': torch.tensor(-20.0),
            'comp_ratio': torch.tensor(10.0),
            'comp_attack': torch.tensor(0.001),
            'comp_release': torch.tensor(0.1),
            'comp_makeup_gain': torch.tensor(6.0),
            'saturation': torch.tensor(1.0)
        }
        
        processed = self.channel_strip(audio, params)
        
        # Dynamic range should be reduced
        original_dynamics = torch.max(torch.abs(audio)) / (torch.mean(torch.abs(audio)) + 1e-8)
        processed_dynamics = torch.max(torch.abs(processed)) / (torch.mean(torch.abs(processed)) + 1e-8)
        
        self.assertLess(processed_dynamics, original_dynamics)


class TestMasteringChain(unittest.TestCase):
    """Test cases for MasteringChain."""
    
    def setUp(self):
        self.mastering = MasteringChain()
        
    def test_mastering_chain_processing(self):
        """Test that mastering chain processes audio without errors."""
        audio = torch.randn(2, 48000) * 0.3  # Stereo input
        
        params = {
            'bus_comp_threshold': torch.tensor(-12.0),
            'bus_comp_ratio': torch.tensor(3.0),
            'bus_comp_attack': torch.tensor(0.01),
            'bus_comp_release': torch.tensor(0.1),
            'bus_comp_makeup_gain': torch.tensor(2.0),
            'master_eq_low_gain': torch.tensor(0.0),
            'master_eq_low_mid_gain': torch.tensor(0.0),
            'master_eq_high_mid_gain': torch.tensor(0.0),
            'master_eq_high_gain': torch.tensor(0.0),
            'stereo_width': torch.tensor(1.2),
            'limiter_threshold': torch.tensor(0.95),
            'limiter_release': torch.tensor(0.05)
        }
        
        processed = self.mastering(audio, params)
        
        # Check output
        self.assertEqual(processed.shape, audio.shape)
        self.assertTrue(torch.isfinite(processed).all())
        
    def test_limiting_prevents_clipping(self):
        """Test that limiting prevents clipping."""
        # Create audio that would clip
        audio = torch.randn(2, 48000) * 1.5  # Deliberately too loud
        
        params = {
            'bus_comp_threshold': torch.tensor(-6.0),
            'bus_comp_ratio': torch.tensor(2.0),
            'bus_comp_attack': torch.tensor(0.01),
            'bus_comp_release': torch.tensor(0.1),
            'bus_comp_makeup_gain': torch.tensor(0.0),
            'master_eq_low_gain': torch.tensor(0.0),
            'master_eq_low_mid_gain': torch.tensor(0.0),
            'master_eq_high_mid_gain': torch.tensor(0.0),
            'master_eq_high_gain': torch.tensor(0.0),
            'stereo_width': torch.tensor(1.0),
            'limiter_threshold': torch.tensor(0.8),
            'limiter_release': torch.tensor(0.05)
        }
        
        processed = self.mastering(audio, params)
        
        # Output should not exceed limiter threshold significantly
        max_amplitude = torch.max(torch.abs(processed))
        self.assertLessEqual(max_amplitude, 1.0)  # Should not clip


class TestAutoMixChain(unittest.TestCase):
    """Test cases for complete AutoMixChain."""
    
    def setUp(self):
        self.targets = load_style_targets()
        self.auto_mix = AutoMixChain(n_stems=4, style_targets=self.targets)
        
    def test_complete_mixing_pipeline(self):
        """Test complete mixing pipeline."""
        # Create test stems
        stems = create_white_noise_stems(4, duration=2.0)
        
        # Process through auto-mix
        mixed_audio, analysis = self.auto_mix(stems, 'rock_punk')
        
        # Check outputs
        self.assertEqual(mixed_audio.shape[0], 2)  # Stereo output
        self.assertGreater(mixed_audio.shape[1], 0)  # Has samples
        self.assertTrue(torch.isfinite(mixed_audio).all())
        
        # Check analysis
        self.assertIn('lufs', analysis)
        self.assertIn('spectral_centroid', analysis)
        self.assertIn('stereo_ms_ratio', analysis)
        
    def test_different_styles(self):
        """Test that different styles produce different results."""
        stems = create_white_noise_stems(4, duration=2.0)
        
        # Process with different styles
        mixed_rock, analysis_rock = self.auto_mix(stems, 'rock_punk')
        mixed_rnb, analysis_rnb = self.auto_mix(stems, 'rnb_ballad')
        
        # Results should be different
        self.assertFalse(torch.allclose(mixed_rock, mixed_rnb, atol=1e-3))
        
        # Analysis should reflect style differences
        lufs_diff = abs(analysis_rock['lufs'] - analysis_rnb['lufs'])
        self.assertGreater(lufs_diff, 0.1)  # Should have some difference
        
    def test_consistent_output_length(self):
        """Test that output length matches input length."""
        # Create stems of different lengths
        stem1 = torch.randn(2, 48000)  # 1 second
        stem2 = torch.randn(2, 96000)  # 2 seconds
        stem3 = torch.randn(2, 24000)  # 0.5 seconds
        
        stems = [stem1, stem2, stem3]
        
        mixed_audio, _ = self.auto_mix(stems, 'country_pop')
        
        # Output should match longest input
        expected_length = 96000
        self.assertEqual(mixed_audio.shape[1], expected_length)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_lufs_computation(self):
        """Test LUFS computation with known signals."""
        # Create -20dB sine wave
        t = torch.linspace(0, 1, 48000)
        sine = torch.sin(2 * np.pi * 440 * t) * 0.1  # -20dB
        audio = torch.stack([sine, sine])  # Stereo
        
        lufs = compute_lufs(audio)
        
        # Should be approximately -20 LUFS (simplified calculation)
        self.assertGreater(lufs, -30)
        self.assertLess(lufs, -10)
        
    def test_spectral_centroid_computation(self):
        """Test spectral centroid computation."""
        # Create 1kHz sine wave
        t = torch.linspace(0, 1, 48000)
        sine = torch.sin(2 * np.pi * 1000 * t) * 0.1
        audio = torch.stack([sine, sine])
        
        centroid = compute_spectral_centroid(audio)
        
        # Should be close to 1000 Hz
        self.assertGreater(centroid, 800)
        self.assertLess(centroid, 1200)
        
    def test_stereo_ms_ratio(self):
        """Test M/S ratio computation."""
        # Create mono signal (should have high M/S ratio)
        mono = torch.randn(48000) * 0.1
        audio = torch.stack([mono, mono])  # Identical L/R
        
        ms_ratio = compute_stereo_ms_ratio(audio)
        
        # Should be very high for mono content
        self.assertGreater(ms_ratio, 100)  # Pure mono has very high ratio
        
        # Create wide stereo signal
        left = torch.randn(48000) * 0.1
        right = -left  # Anti-correlated = pure side
        audio_wide = torch.stack([left, right])
        
        ms_ratio_wide = compute_stereo_ms_ratio(audio_wide)
        
        # Should be much lower for wide content
        self.assertLess(ms_ratio_wide, ms_ratio)
        
    def test_white_noise_generation(self):
        """Test white noise stem generation."""
        stems = create_white_noise_stems(4, duration=1.0)
        
        self.assertEqual(len(stems), 4)
        for stem in stems:
            self.assertEqual(stem.shape, (2, 48000))  # Stereo, 1 second
            self.assertTrue(torch.isfinite(stem).all())
            
    def test_validation_functions(self):
        """Test validation functions."""
        # LUFS validation
        self.assertTrue(validate_lufs_accuracy(-14.0, -14.5, tolerance=1.0))
        self.assertFalse(validate_lufs_accuracy(-14.0, -16.5, tolerance=1.0))
        
        # Spectral validation
        self.assertTrue(validate_spectral_accuracy(2000, 2100, tolerance=200))
        self.assertFalse(validate_spectral_accuracy(2000, 2500, tolerance=200))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_end_to_end_mixing(self):
        """Test complete end-to-end mixing workflow."""
        # Create realistic test scenario
        stems = create_white_noise_stems(6, duration=3.0)
        targets = load_style_targets()
        
        auto_mix = AutoMixChain(n_stems=6, style_targets=targets)
        
        # Mix for each style
        results = {}
        for style in ['rock_punk', 'rnb_ballad', 'country_pop']:
            mixed_audio, analysis = auto_mix(stems, style)
            results[style] = analysis
            
            # Check that audio is finite and not silent
            self.assertTrue(torch.isfinite(mixed_audio).all())
            self.assertGreater(torch.max(torch.abs(mixed_audio)), 1e-6)
            
        # Check that styles produce different LUFS targets
        rock_lufs = results['rock_punk']['lufs']
        rnb_lufs = results['rnb_ballad']['lufs']
        country_lufs = results['country_pop']['lufs']
        
        # Rock should be loudest, RnB should be quietest
        self.assertLess(rock_lufs, country_lufs)  # Lower LUFS = louder
        self.assertLess(country_lufs, rnb_lufs)
        
    def test_performance_benchmarks(self):
        """Test that processing completes in reasonable time."""
        import time
        
        stems = create_white_noise_stems(8, duration=10.0)  # Longer test
        auto_mix = AutoMixChain(n_stems=8)
        
        start_time = time.time()
        mixed_audio, analysis = auto_mix(stems, 'rock_punk')
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process 10 seconds of audio in reasonable time
        # This is a rough benchmark - adjust based on hardware
        self.assertLess(processing_time, 30.0)  # 30 seconds max
        
        print(f"Processed 10s of 8-stem audio in {processing_time:.2f}s")


def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    test_classes = [
        TestStemFeatureExtractor,
        TestMixingParameterPredictor,
        TestChannelStrip,
        TestMasteringChain,
        TestAutoMixChain,
        TestUtilityFunctions,
        TestIntegration
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
        
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)