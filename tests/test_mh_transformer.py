"""
Test suite for Melody & Harmony Transformer model

Tests model architecture, forward pass, and generation functionality.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List

# Import modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.mh_transformer import (
    MelodyHarmonyTransformer, 
    MHTrainingLoss,
    PositionalEncoding,
    StyleEmbedding
)


class TestPositionalEncoding(unittest.TestCase):
    """Test positional encoding module"""
    
    def test_positional_encoding_shape(self):
        """Test that positional encoding produces correct shapes"""
        d_model = 256
        seq_len = 128
        batch_size = 4
        
        pos_enc = PositionalEncoding(d_model, dropout=0.0)
        
        # Create input tensor
        x = torch.randn(seq_len, batch_size, d_model)
        
        # Apply positional encoding
        output = pos_enc(x)
        
        # Check shape is preserved
        self.assertEqual(output.shape, (seq_len, batch_size, d_model))
    
    def test_positional_encoding_deterministic(self):
        """Test that positional encoding is deterministic"""
        d_model = 128
        seq_len = 64
        
        pos_enc = PositionalEncoding(d_model, dropout=0.0)
        
        # Create identical inputs
        x1 = torch.zeros(seq_len, 1, d_model)
        x2 = torch.zeros(seq_len, 1, d_model)
        
        # Apply positional encoding
        output1 = pos_enc(x1)
        output2 = pos_enc(x2)
        
        # Outputs should be identical
        self.assertTrue(torch.allclose(output1, output2))


class TestStyleEmbedding(unittest.TestCase):
    """Test style embedding module"""
    
    def test_style_embedding_shape(self):
        """Test style embedding output shape"""
        d_model = 256
        batch_size = 4
        
        style_emb = StyleEmbedding(d_model, style_vocab_size=3)
        
        # Create inputs
        style_ids = torch.randint(0, 3, (batch_size,))
        key_ids = torch.randint(0, 24, (batch_size,))
        section_ids = torch.randint(0, 5, (batch_size,))
        
        # Forward pass
        output = style_emb(style_ids, key_ids, section_ids)
        
        # Check shape
        self.assertEqual(output.shape, (batch_size, d_model))
    
    def test_style_embedding_with_groove(self):
        """Test style embedding with groove features"""
        d_model = 256
        batch_size = 4
        
        style_emb = StyleEmbedding(d_model, style_vocab_size=3)
        
        # Create inputs including groove features
        style_ids = torch.randint(0, 3, (batch_size,))
        key_ids = torch.randint(0, 24, (batch_size,))
        section_ids = torch.randint(0, 5, (batch_size,))
        groove_features = torch.randn(batch_size, 32)
        
        # Forward pass with groove
        output_with_groove = style_emb(style_ids, key_ids, section_ids, groove_features)
        
        # Forward pass without groove
        output_without_groove = style_emb(style_ids, key_ids, section_ids)
        
        # Outputs should be different
        self.assertFalse(torch.allclose(output_with_groove, output_without_groove))
        
        # Shape should be the same
        self.assertEqual(output_with_groove.shape, output_without_groove.shape)


class TestMelodyHarmonyTransformer(unittest.TestCase):
    """Test main transformer model"""
    
    def setUp(self):
        """Set up test model"""
        self.vocab_size = 1000
        self.d_model = 256
        self.nhead = 8
        self.num_layers = 4
        self.batch_size = 2
        self.seq_len = 64
        
        self.model = MelodyHarmonyTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            style_vocab_size=3,
            chord_vocab_size=60
        )
    
    def test_model_creation(self):
        """Test that model is created successfully"""
        self.assertIsInstance(self.model, MelodyHarmonyTransformer)
        self.assertEqual(self.model.vocab_size, self.vocab_size)
        self.assertEqual(self.model.d_model, self.d_model)
    
    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shapes"""
        # Create inputs
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        style_ids = torch.randint(0, 3, (self.batch_size,))
        key_ids = torch.randint(0, 24, (self.batch_size,))
        section_ids = torch.randint(0, 5, (self.batch_size,))
        
        # Forward pass
        outputs = self.model(input_ids, style_ids, key_ids, section_ids)
        
        # Check output shapes
        self.assertEqual(outputs['logits'].shape, (self.batch_size, self.seq_len, self.vocab_size))
        self.assertEqual(outputs['chord_compatibility'].shape, (self.batch_size, self.seq_len, 60))
        self.assertEqual(outputs['scale_compatibility'].shape, (self.batch_size, self.seq_len, 12))
        self.assertEqual(outputs['hidden_states'].shape, (self.batch_size, self.seq_len, self.d_model))
    
    def test_forward_pass_with_attention_mask(self):
        """Test forward pass with attention mask"""
        # Create inputs with attention mask
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        style_ids = torch.randint(0, 3, (self.batch_size,))
        key_ids = torch.randint(0, 24, (self.batch_size,))
        section_ids = torch.randint(0, 5, (self.batch_size,))
        
        # Create attention mask (mask last 10 tokens)
        attention_mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.bool)
        attention_mask[:, -10:] = True
        
        # Forward pass should not raise error
        outputs = self.model(input_ids, style_ids, key_ids, section_ids, attention_mask=attention_mask)
        
        # Check shapes are still correct
        self.assertEqual(outputs['logits'].shape, (self.batch_size, self.seq_len, self.vocab_size))
    
    def test_causal_mask_creation(self):
        """Test causal mask creation"""
        seq_len = 10
        device = torch.device('cpu')
        
        mask = self.model.create_causal_mask(seq_len, device)
        
        # Check shape
        self.assertEqual(mask.shape, (seq_len, seq_len))
        
        # Check that mask is upper triangular with -inf
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    self.assertEqual(mask[i, j].item(), float('-inf'))
                else:
                    self.assertEqual(mask[i, j].item(), 0.0)
    
    def test_generation(self):
        """Test sequence generation"""
        prompt_len = 10
        max_length = 50
        
        # Create prompt
        prompt_ids = torch.randint(0, self.vocab_size, (1, prompt_len))
        style_ids = torch.tensor([0])  # rock_punk
        key_ids = torch.tensor([0])    # C major
        section_ids = torch.tensor([1])  # verse
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                prompt_ids=prompt_ids,
                style_ids=style_ids,
                key_ids=key_ids,
                section_ids=section_ids,
                max_length=max_length,
                temperature=0.8,
                nucleus_p=0.9
            )
        
        # Check output shape
        self.assertEqual(generated.shape[0], 1)  # batch size
        self.assertLessEqual(generated.shape[1], max_length)  # sequence length
        self.assertGreaterEqual(generated.shape[1], prompt_len)  # at least prompt length
    
    def test_generation_with_constraints(self):
        """Test generation with constraint mask"""
        prompt_len = 5
        max_length = 20
        
        prompt_ids = torch.randint(0, self.vocab_size, (1, prompt_len))
        style_ids = torch.tensor([0])
        key_ids = torch.tensor([0])
        section_ids = torch.tensor([1])
        
        # Create constraint mask that heavily penalizes certain tokens
        constraint_mask = torch.zeros(1, max_length, self.vocab_size)
        constraint_mask[:, :, 100:200] = -1000.0  # Heavy penalty for tokens 100-199
        
        # Generate with constraints
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                prompt_ids=prompt_ids,
                style_ids=style_ids,
                key_ids=key_ids,
                section_ids=section_ids,
                max_length=max_length,
                constraint_mask=constraint_mask
            )
        
        # Check that penalized tokens don't appear (except in prompt)
        generated_new = generated[:, prompt_len:]
        penalized_tokens = torch.arange(100, 200)
        
        for token in penalized_tokens:
            self.assertNotIn(token.item(), generated_new[0].tolist())


class TestMHTrainingLoss(unittest.TestCase):
    """Test training loss with auxiliary constraints"""
    
    def setUp(self):
        """Set up test loss function"""
        self.vocab_size = 1000
        self.chord_vocab_size = 60
        self.batch_size = 2
        self.seq_len = 32
        
        self.criterion = MHTrainingLoss(
            vocab_size=self.vocab_size,
            chord_vocab_size=self.chord_vocab_size,
            scale_penalty_weight=0.1,
            repetition_penalty_weight=0.05,
            chord_compatibility_weight=0.2
        )
    
    def test_loss_computation(self):
        """Test basic loss computation"""
        # Create mock model outputs
        outputs = {
            'logits': torch.randn(self.batch_size, self.seq_len, self.vocab_size),
            'chord_compatibility': torch.randn(self.batch_size, self.seq_len, self.chord_vocab_size),
            'scale_compatibility': torch.randn(self.batch_size, self.seq_len, 12),
            'hidden_states': torch.randn(self.batch_size, self.seq_len, 256)
        }
        
        # Create targets
        target_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        key_ids = torch.randint(0, 24, (self.batch_size,))
        chord_targets = torch.randint(0, self.chord_vocab_size, (self.batch_size, self.seq_len))
        scale_targets = torch.randn(self.batch_size, self.seq_len, 12).sigmoid()  # Binary targets
        
        # Compute loss
        losses = self.criterion(
            outputs=outputs,
            target_ids=target_ids,
            key_ids=key_ids,
            chord_targets=chord_targets,
            scale_targets=scale_targets
        )
        
        # Check that all expected losses are present
        expected_losses = ['main_loss', 'repetition_loss', 'chord_compatibility_loss', 
                          'scale_compatibility_loss', 'total_loss']
        
        for loss_name in expected_losses:
            self.assertIn(loss_name, losses)
            self.assertIsInstance(losses[loss_name], torch.Tensor)
            self.assertEqual(losses[loss_name].dim(), 0)  # Scalar loss
    
    def test_loss_weights(self):
        """Test that loss weights are applied correctly"""
        # Create outputs and targets
        outputs = {
            'logits': torch.randn(self.batch_size, self.seq_len, self.vocab_size),
            'chord_compatibility': torch.randn(self.batch_size, self.seq_len, self.chord_vocab_size),
            'scale_compatibility': torch.randn(self.batch_size, self.seq_len, 12),
            'hidden_states': torch.randn(self.batch_size, self.seq_len, 256)
        }
        
        target_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        key_ids = torch.randint(0, 24, (self.batch_size,))
        
        # Compute loss without auxiliary targets
        losses_without_aux = self.criterion(outputs, target_ids, key_ids)
        
        # Should only have main_loss, repetition_loss, and total_loss
        self.assertIn('main_loss', losses_without_aux)
        self.assertIn('repetition_loss', losses_without_aux)
        self.assertIn('total_loss', losses_without_aux)
        self.assertNotIn('chord_compatibility_loss', losses_without_aux)
        self.assertNotIn('scale_compatibility_loss', losses_without_aux)
    
    def test_repetition_penalty_computation(self):
        """Test repetition penalty computation"""
        # Create sequence with repetition
        logits = torch.randn(1, 10, self.vocab_size)
        input_ids = torch.tensor([[1, 2, 1, 3, 1, 4, 5, 6, 7, 8]])  # Token 1 repeats
        
        penalty = self.criterion.compute_repetition_penalty(input_ids, logits)
        
        # Check shape
        self.assertEqual(penalty.shape, logits.shape)
        
        # Token 1 should have more penalty than others
        # Note: penalty is negative (reduces probability)
        token_1_penalty = penalty[0, -1, 1].item()  # Last position, token 1
        token_4_penalty = penalty[0, -1, 4].item()  # Last position, token 4
        
        self.assertLess(token_1_penalty, token_4_penalty)  # More negative = more penalty


class TestModelIntegration(unittest.TestCase):
    """Integration tests for the complete model"""
    
    def test_training_step_simulation(self):
        """Simulate a complete training step"""
        # Create model and loss
        model = MelodyHarmonyTransformer(
            vocab_size=500,
            d_model=128,
            nhead=4,
            num_layers=2,
            style_vocab_size=3
        )
        
        criterion = MHTrainingLoss(vocab_size=500, chord_vocab_size=30)
        
        # Create batch
        batch_size = 2
        seq_len = 32
        
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        target_ids = torch.randint(0, 500, (batch_size, seq_len))
        style_ids = torch.randint(0, 3, (batch_size,))
        key_ids = torch.randint(0, 24, (batch_size,))
        section_ids = torch.randint(0, 5, (batch_size,))
        chord_targets = torch.randint(0, 30, (batch_size, seq_len))
        scale_targets = torch.randn(batch_size, seq_len, 12).sigmoid()
        
        # Forward pass
        model.train()
        outputs = model(input_ids, style_ids, key_ids, section_ids)
        
        # Compute loss
        losses = criterion(
            outputs=outputs,
            target_ids=target_ids,
            key_ids=key_ids,
            chord_targets=chord_targets,
            scale_targets=scale_targets
        )
        
        total_loss = losses['total_loss']
        
        # Backward pass should work
        total_loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_style_conditioning_effect(self):
        """Test that different styles produce different outputs"""
        model = MelodyHarmonyTransformer(
            vocab_size=100,
            d_model=128,
            nhead=4,
            num_layers=2,
            style_vocab_size=3
        )
        
        # Same input with different styles
        input_ids = torch.randint(0, 100, (1, 20))
        key_ids = torch.tensor([0])
        section_ids = torch.tensor([1])
        
        model.eval()
        with torch.no_grad():
            # Style 0 (rock_punk)
            outputs_style0 = model(input_ids, torch.tensor([0]), key_ids, section_ids)
            
            # Style 1 (rnb_ballad)
            outputs_style1 = model(input_ids, torch.tensor([1]), key_ids, section_ids)
        
        # Outputs should be different
        self.assertFalse(torch.allclose(
            outputs_style0['logits'], 
            outputs_style1['logits'], 
            atol=1e-6
        ))
    
    def test_key_conditioning_effect(self):
        """Test that different keys produce different outputs"""
        model = MelodyHarmonyTransformer(
            vocab_size=100,
            d_model=128,
            nhead=4,
            num_layers=2,
            style_vocab_size=3
        )
        
        # Same input with different keys
        input_ids = torch.randint(0, 100, (1, 20))
        style_ids = torch.tensor([0])
        section_ids = torch.tensor([1])
        
        model.eval()
        with torch.no_grad():
            # C major (key_id = 0)
            outputs_c_major = model(input_ids, style_ids, torch.tensor([0]), section_ids)
            
            # G major (key_id = 14)
            outputs_g_major = model(input_ids, style_ids, torch.tensor([14]), section_ids)
        
        # Outputs should be different
        self.assertFalse(torch.allclose(
            outputs_c_major['logits'], 
            outputs_g_major['logits'], 
            atol=1e-6
        ))


def run_model_tests():
    """Run all model-related tests"""
    print("Running Melody & Harmony Transformer model tests...")
    
    test_classes = [
        TestPositionalEncoding,
        TestStyleEmbedding,
        TestMelodyHarmonyTransformer,
        TestMHTrainingLoss,
        TestModelIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}:")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        passed_tests += result.testsRun - len(result.failures) - len(result.errors)
        
        if result.failures:
            print(f"Failures in {test_class.__name__}:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        
        if result.errors:
            print(f"Errors in {test_class.__name__}:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    print(f"\nTest Summary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("All model tests passed! ✅")
        return True
    else:
        print("Some tests failed! ❌")
        return False


if __name__ == '__main__':
    success = run_model_tests()
    exit(0 if success else 1)