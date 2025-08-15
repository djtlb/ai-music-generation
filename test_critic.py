"""
Unit tests for the critic package components.
Tests classifier, critic model, and DPO training functionality.
"""

import pytest
import torch
import numpy as np
import tempfile
import json
from pathlib import Path

from critic.classifier import AdherenceClassifier, AdherenceScore, AdherenceDataset
from critic.model import ComprehensiveCritic, CriticScore, extract_mix_features
from critic.dpo_finetune import DPODataset, DPOLoss, DPOTrainer
from critic.evaluate import AdherenceEvaluator

class TestAdherenceClassifier:
    """Test the adherence classifier"""
    
    def setup_method(self):
        self.vocab_size = 100
        self.classifier = AdherenceClassifier(self.vocab_size)
        self.batch_size = 2
        
    def test_classifier_initialization(self):
        """Test classifier initializes correctly"""
        assert self.classifier.text_encoder is not None
        assert self.classifier.control_encoder is not None
        assert self.classifier.token_encoder is not None
        
    def test_classifier_forward(self):
        """Test classifier forward pass"""
        prompts = ["upbeat pop song", "slow ballad"]
        controls = [
            {'style': 'pop', 'bpm': 120, 'key': 'C', 'timefeel': 'straight'},
            {'style': 'ballad', 'bpm': 80, 'key': 'Am', 'timefeel': 'swing'}
        ]
        tokens = torch.randint(0, self.vocab_size, (self.batch_size, 32))
        
        overall_scores, component_scores = self.classifier(prompts, controls, tokens)
        
        assert overall_scores.shape == (self.batch_size, 1)
        assert 'tempo' in component_scores
        assert 'key' in component_scores
        assert component_scores['tempo'].shape == (self.batch_size, 1)
        
    def test_predict_adherence(self):
        """Test structured adherence prediction"""
        prompts = ["upbeat pop song"]
        controls = [{'style': 'pop', 'bpm': 120, 'key': 'C'}]
        tokens = torch.randint(0, self.vocab_size, (1, 32))
        
        scores = self.classifier.predict_adherence(prompts, controls, tokens)
        
        assert len(scores) == 1
        assert isinstance(scores[0], AdherenceScore)
        assert 0 <= scores[0].overall <= 1
        assert 0 <= scores[0].tempo_adherence <= 1
        
    def test_key_to_id_mapping(self):
        """Test key string to ID conversion"""
        encoder = self.classifier.control_encoder
        
        assert encoder._key_to_id('C') == 0
        assert encoder._key_to_id('Am') == 21
        assert encoder._key_to_id('F#') == 6
        assert encoder._key_to_id('Bbm') == 22

class TestComprehensiveCritic:
    """Test the comprehensive critic model"""
    
    def setup_method(self):
        self.vocab_size = 100
        self.critic = ComprehensiveCritic(self.vocab_size)
        self.batch_size = 2
        
    def test_critic_initialization(self):
        """Test critic initializes correctly"""
        assert self.critic.adherence_classifier is not None
        assert self.critic.style_encoder is not None
        assert self.critic.mix_assessor is not None
        
    def test_critic_forward(self):
        """Test critic forward pass"""
        prompts = ["upbeat pop song", "slow ballad"]
        controls = [
            {'style': 'pop', 'bpm': 120, 'key': 'C'},
            {'style': 'ballad', 'bpm': 80, 'key': 'Am'}
        ]
        tokens = torch.randint(0, self.vocab_size, (self.batch_size, 32))
        mel_specs = torch.randn(self.batch_size, 1, 64, 64)
        ref_embeddings = torch.randn(self.batch_size, 512)
        mix_features = torch.randn(self.batch_size, 32)
        
        overall_scores, component_scores = self.critic(
            prompts, controls, tokens, mel_specs, ref_embeddings, mix_features
        )
        
        assert overall_scores.shape == (self.batch_size, 1)
        assert 'adherence' in component_scores
        assert 'style_match' in component_scores
        assert 'mix_quality' in component_scores
        assert 'confidence' in component_scores
        
    def test_evaluate_comprehensive(self):
        """Test comprehensive evaluation"""
        prompts = ["upbeat pop song"]
        controls = [{'style': 'pop', 'bpm': 120, 'key': 'C'}]
        tokens = torch.randint(0, self.vocab_size, (1, 32))
        mel_specs = torch.randn(1, 1, 64, 64)
        ref_embeddings = torch.randn(1, 512)
        mix_features = torch.randn(1, 32)
        
        scores = self.critic.evaluate_comprehensive(
            prompts, controls, tokens, mel_specs, ref_embeddings, mix_features
        )
        
        assert len(scores) == 1
        assert isinstance(scores[0], CriticScore)
        assert 0 <= scores[0].overall <= 1
        assert 0 <= scores[0].confidence <= 1
        assert isinstance(scores[0].notes, list)

class TestMixFeatureExtraction:
    """Test mix feature extraction utilities"""
    
    def test_extract_mix_features_mono(self):
        """Test feature extraction from mono audio"""
        # Create dummy audio (1 second at 44.1kHz)
        audio = np.random.randn(44100) * 0.1
        
        features = extract_mix_features(audio, sample_rate=44100)
        
        assert features.shape == (32,)
        assert features.dtype == np.float32
        
        # Check that features are reasonable
        lufs_approx = features[0]
        assert -60 < lufs_approx < 0  # Reasonable LUFS range
        
    def test_extract_mix_features_stereo(self):
        """Test feature extraction from stereo audio"""
        # Create dummy stereo audio
        audio = np.random.randn(2, 44100) * 0.1
        
        features = extract_mix_features(audio, sample_rate=44100)
        
        assert features.shape == (32,)
        # Stereo width should be non-zero for stereo input
        stereo_width = features[5]  # Assuming stereo width is at index 5
        assert stereo_width >= 0

class TestDPOTraining:
    """Test DPO training components"""
    
    def setup_method(self):
        self.vocab_size = 100
        
    def test_dpo_loss(self):
        """Test DPO loss computation"""
        loss_fn = DPOLoss(beta=0.1, reference_free=True)
        
        # Preferred should have higher log prob
        preferred_logprobs = torch.tensor([0.8, 0.9])
        dispreferred_logprobs = torch.tensor([0.6, 0.7])
        
        loss = loss_fn(preferred_logprobs, dispreferred_logprobs)
        
        assert loss.item() > 0
        assert torch.isfinite(loss)
        
    def test_dpo_loss_with_reference(self):
        """Test DPO loss with reference model"""
        loss_fn = DPOLoss(beta=0.1, reference_free=False)
        
        policy_preferred = torch.tensor([0.8, 0.9])
        policy_dispreferred = torch.tensor([0.6, 0.7])
        ref_preferred = torch.tensor([0.7, 0.8])
        ref_dispreferred = torch.tensor([0.65, 0.75])
        
        loss = loss_fn(
            policy_preferred, policy_dispreferred,
            ref_preferred, ref_dispreferred
        )
        
        assert loss.item() > 0
        assert torch.isfinite(loss)
        
    def test_dpo_dataset(self):
        """Test DPO dataset loading"""
        # Create temporary test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            sample_data = {
                'prompt': 'test prompt',
                'control_json': {'style': 'pop'},
                'preferred_tokens': [1, 2, 3, 4],
                'dispreferred_tokens': [1, 2, 5, 6],
                'preferred_score': 0.8,
                'dispreferred_score': 0.6
            }
            f.write(json.dumps(sample_data) + '\n')
            temp_path = f.name
        
        try:
            dataset = DPODataset(temp_path, max_seq_len=10)
            
            assert len(dataset) == 1
            
            sample = dataset[0]
            assert sample['prompt'] == 'test prompt'
            assert sample['preferred_tokens'].shape == (10,)  # Padded to max_seq_len
            assert sample['dispreferred_tokens'].shape == (10,)
            assert sample['preference_margin'].item() == 0.2
            
        finally:
            Path(temp_path).unlink()

class TestAdherenceEvaluator:
    """Test adherence evaluation functionality"""
    
    def setup_method(self):
        self.vocab_size = 100
        self.critic = ComprehensiveCritic(self.vocab_size)
        self.evaluator = AdherenceEvaluator(self.critic)
        
    def test_evaluator_initialization(self):
        """Test evaluator initializes correctly"""
        assert self.evaluator.critic is not None
        assert self.evaluator.device is not None
        
    def test_average_critic_scores(self):
        """Test averaging multiple CriticScore objects"""
        from critic.classifier import AdherenceScore
        
        # Create dummy adherence details
        adherence_details = [
            AdherenceScore(
                overall=0.8, tempo_adherence=0.9, key_adherence=0.7,
                structure_adherence=0.8, genre_adherence=0.9, 
                instrumentation_adherence=0.8, details={}
            ),
            AdherenceScore(
                overall=0.6, tempo_adherence=0.7, key_adherence=0.5,
                structure_adherence=0.6, genre_adherence=0.7,
                instrumentation_adherence=0.6, details={}
            )
        ]
        
        # Create dummy CriticScore objects
        scores = [
            CriticScore(
                overall=0.8, adherence=0.8, style_match=0.7, mix_quality=0.9,
                adherence_details=adherence_details[0],
                style_details={}, mix_details={}, confidence=0.9, notes=["good"]
            ),
            CriticScore(
                overall=0.6, adherence=0.6, style_match=0.5, mix_quality=0.7,
                adherence_details=adherence_details[1],
                style_details={}, mix_details={}, confidence=0.7, notes=["ok"]
            )
        ]
        
        avg_score = self.evaluator._average_critic_scores(scores)
        
        assert avg_score.overall == 0.7  # (0.8 + 0.6) / 2
        assert avg_score.adherence == 0.7
        assert avg_score.adherence_details.overall == 0.7  # (0.8 + 0.6) / 2
        
    def test_compute_aggregate_metrics(self):
        """Test aggregate metrics computation"""
        from critic.classifier import AdherenceScore
        
        # Create dummy scores
        adherence_details = AdherenceScore(
            overall=0.8, tempo_adherence=0.9, key_adherence=0.7,
            structure_adherence=0.8, genre_adherence=0.9,
            instrumentation_adherence=0.8, details={}
        )
        
        scores = [
            CriticScore(
                overall=0.8, adherence=0.8, style_match=0.7, mix_quality=0.9,
                adherence_details=adherence_details, style_details={}, 
                mix_details={}, confidence=0.9, notes=[]
            ),
            CriticScore(
                overall=0.6, adherence=0.6, style_match=0.5, mix_quality=0.7,
                adherence_details=adherence_details, style_details={}, 
                mix_details={}, confidence=0.7, notes=[]
            )
        ]
        
        metrics = self.evaluator._compute_aggregate_metrics(scores)
        
        assert 'overall_mean' in metrics
        assert 'adherence_mean' in metrics
        assert 'high_quality_ratio' in metrics
        assert metrics['overall_mean'] == 0.7
        assert metrics['num_samples'] == 2

def run_integration_test():
    """Integration test of the full critic pipeline"""
    vocab_size = 100
    
    # Create models
    classifier = AdherenceClassifier(vocab_size)
    critic = ComprehensiveCritic(vocab_size)
    evaluator = AdherenceEvaluator(critic)
    
    # Create test inputs
    prompts = ["upbeat pop song"]
    controls = [{'style': 'pop', 'bpm': 120, 'key': 'C'}]
    tokens = torch.randint(0, vocab_size, (1, 32))
    
    # Test classifier
    adherence_scores = classifier.predict_adherence(prompts, controls, tokens)
    assert len(adherence_scores) == 1
    
    # Test critic
    mel_specs = torch.randn(1, 1, 64, 64)
    ref_embeddings = torch.randn(1, 512)
    mix_features = torch.randn(1, 32)
    
    critic_scores = critic.evaluate_comprehensive(
        prompts, controls, tokens, mel_specs, ref_embeddings, mix_features
    )
    assert len(critic_scores) == 1
    
    # Test aggregation
    metrics = evaluator._compute_aggregate_metrics(critic_scores)
    assert 'overall_mean' in metrics
    
    print("Integration test passed!")

if __name__ == "__main__":
    # Run integration test
    run_integration_test()
    
    # Run pytest if available
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, but integration test passed!")