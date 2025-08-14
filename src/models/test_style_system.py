"""
Unit Tests for Style Embeddings and Retrieval System

Tests the audio encoder, FAISS index, and retrieval fusion components
with synthetic data to verify bias functionality.
"""

import unittest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import json

# Import the modules we're testing
import sys
sys.path.append(str(Path(__file__).parent))

from style_encoder import StyleEncoder, LogMelExtractor, create_style_encoder, StyleEncoderLoss
from retrieval_fusion import (
    RetrievalFusion, RetrievalConfig, TokenVocabulary, 
    NGramMatcher, RetrievalBiasedGenerator, create_test_patterns
)

try:
    from style_index import StylePatternIndex, PatternEmbedding
    FAISS_TESTS_AVAILABLE = True
except ImportError:
    FAISS_TESTS_AVAILABLE = False


class TestStyleEncoder(unittest.TestCase):
    """Test the style audio encoder"""
    
    def setUp(self):
        self.config = {
            'n_mels': 128,
            'embedding_dim': 256,  # Smaller for testing
            'n_classes': 3,
            'dropout': 0.2
        }
        self.model = create_style_encoder(self.config)
        
    def test_model_creation(self):
        """Test model can be created with correct architecture"""
        self.assertIsInstance(self.model, StyleEncoder)
        self.assertEqual(self.model.embedding_dim, 256)
        self.assertEqual(self.model.n_classes, 3)
        
    def test_log_mel_extraction(self):
        """Test log-mel spectrogram extraction"""
        # Create dummy audio (10 seconds at 22050 Hz)
        batch_size = 2
        audio_length = 220500  # 10 seconds
        audio = torch.randn(batch_size, audio_length)
        
        # Extract log-mel
        log_mel = self.model.mel_extractor(audio)
        
        # Check output shape
        expected_time_frames = audio_length // 512 + 1  # hop_length = 512
        self.assertEqual(log_mel.shape[0], batch_size)
        self.assertEqual(log_mel.shape[1], 128)  # n_mels
        self.assertAlmostEqual(log_mel.shape[2], expected_time_frames, delta=5)
        
    def test_forward_pass(self):
        """Test complete forward pass"""
        batch_size = 4
        audio = torch.randn(batch_size, 220500)
        
        outputs = self.model(audio)
        
        # Check output structure
        self.assertIn('embeddings', outputs)
        self.assertIn('logits', outputs)
        self.assertIn('features', outputs)
        
        # Check shapes
        self.assertEqual(outputs['embeddings'].shape, (batch_size, 256))
        self.assertEqual(outputs['logits'].shape, (batch_size, 3))
        self.assertEqual(outputs['features'].shape, (batch_size, 256))
        
        # Check embedding normalization (should be in [-1, 1] due to tanh)
        embeddings = outputs['embeddings']
        self.assertTrue(torch.all(embeddings >= -1.1))  # Small tolerance
        self.assertTrue(torch.all(embeddings <= 1.1))
        
    def test_style_encoding_only(self):
        """Test style encoding without classification"""
        audio = torch.randn(2, 220500)
        embeddings = self.model.encode_style(audio)
        
        self.assertEqual(embeddings.shape, (2, 256))
        self.assertTrue(torch.all(embeddings >= -1.1))
        self.assertTrue(torch.all(embeddings <= 1.1))
        
    def test_loss_computation(self):
        """Test the combined loss function"""
        loss_fn = StyleEncoderLoss(
            classification_weight=1.0,
            contrastive_weight=0.5
        )
        
        # Create dummy outputs and labels
        batch_size = 8
        outputs = {
            'embeddings': torch.randn(batch_size, 256),
            'logits': torch.randn(batch_size, 3),
            'features': torch.randn(batch_size, 256)
        }
        labels = torch.randint(0, 3, (batch_size,))
        
        losses = loss_fn(outputs, labels)
        
        # Check loss structure
        self.assertIn('total_loss', losses)
        self.assertIn('classification_loss', losses)
        self.assertIn('contrastive_loss', losses)
        
        # Check that losses are scalar tensors
        for loss_name, loss_value in losses.items():
            self.assertEqual(loss_value.shape, ())
            self.assertTrue(loss_value.item() >= 0)


class TestNGramMatcher(unittest.TestCase):
    """Test n-gram matching functionality"""
    
    def setUp(self):
        self.matcher = NGramMatcher(ngram_size=3)
        
    def test_pattern_addition(self):
        """Test adding patterns and extracting n-grams"""
        tokens = ['A', 'B', 'C', 'D', 'E']
        self.matcher.add_pattern('test_pattern', tokens)
        
        # Should have 3 n-grams: (A,B,C), (B,C,D), (C,D,E)
        ngrams = self.matcher.pattern_ngrams['test_pattern']
        expected_ngrams = {('A', 'B', 'C'), ('B', 'C', 'D'), ('C', 'D', 'E')}
        self.assertEqual(ngrams, expected_ngrams)
        
    def test_continuation_matching(self):
        """Test finding continuations from context"""
        # Add patterns
        self.matcher.add_pattern('pattern1', ['A', 'B', 'C', 'D'])
        self.matcher.add_pattern('pattern2', ['A', 'B', 'X', 'Y'])
        self.matcher.add_pattern('pattern3', ['X', 'Y', 'Z'])
        
        # Test context matching
        context = ['A', 'B']
        continuations = self.matcher.find_matching_continuations(
            context, ['pattern1', 'pattern2', 'pattern3']
        )
        
        # Should find continuations C and X
        self.assertIn('pattern1', continuations)
        self.assertIn('pattern2', continuations)
        self.assertNotIn('pattern3', continuations)
        
        self.assertIn('C', continuations['pattern1'])
        self.assertIn('X', continuations['pattern2'])


class TestRetrievalFusion(unittest.TestCase):
    """Test retrieval fusion functionality"""
    
    def setUp(self):
        # Create test vocabulary
        self.vocab_dict = {
            'KICK': 0, 'SNARE': 1, 'BAR_1': 2, 'POS_1': 3, 'POS_3': 4,
            'CHORD': 5, 'C': 6, 'Am': 7, 'BAR_2': 8, 'PIANO': 9, 
            'C4': 10, 'E4': 11, '<UNK>': 12, '<PAD>': 13
        }
        self.vocab = TokenVocabulary(self.vocab_dict)
        
        # Create fusion config
        self.config = RetrievalConfig(
            enabled=True, 
            retrieval_weight=0.3, 
            ngram_size=3,
            top_k=3
        )
        self.fusion = RetrievalFusion(self.vocab, self.config)
        
    def test_vocabulary_conversion(self):
        """Test token-to-ID conversion"""
        tokens = ['KICK', 'BAR_1', 'POS_1']
        ids = self.vocab.tokens_to_ids(tokens)
        recovered_tokens = self.vocab.ids_to_tokens(ids)
        
        self.assertEqual(ids, [0, 2, 3])
        self.assertEqual(recovered_tokens, tokens)
        
    def test_pattern_update(self):
        """Test updating retrieved patterns"""
        patterns = create_test_patterns()
        similarities = [0.9, 0.8, 0.7]
        
        self.fusion.update_retrieved_patterns(patterns, similarities)
        
        self.assertEqual(len(self.fusion.retrieved_patterns), 3)
        self.assertEqual(len(self.fusion.pattern_weights), 3)
        self.assertEqual(self.fusion.pattern_weights[0], 0.9)
        
    def test_bias_computation(self):
        """Test retrieval bias computation"""
        # Add patterns
        patterns = create_test_patterns()
        similarities = [0.9, 0.8, 0.7]
        self.fusion.update_retrieved_patterns(patterns, similarities)
        
        # Test bias for context that should match
        context = ['KICK', 'BAR_1']  # Should match rock_pattern_1
        bias = self.fusion.compute_retrieval_bias(context)
        
        # Check bias shape
        self.assertEqual(bias.shape[0], len(self.vocab_dict))
        
        # Should have non-zero bias for POS_1 token (continuation of KICK BAR_1)
        pos_1_id = self.vocab_dict['POS_1']
        self.assertGreater(bias[pos_1_id].item(), 0)
        
    def test_shallow_fusion_methods(self):
        """Test different shallow fusion methods"""
        patterns = create_test_patterns()
        similarities = [0.8, 0.7, 0.6]
        self.fusion.update_retrieved_patterns(patterns, similarities)
        
        context = ['KICK', 'BAR_1']
        logits = torch.randn(len(self.vocab_dict))
        
        # Test shallow fusion
        self.fusion.config.fusion_method = "shallow"
        biased_logits_shallow = self.fusion.apply_shallow_fusion(logits, context)
        
        # Test interpolation
        self.fusion.config.fusion_method = "interpolation"
        biased_logits_interp = self.fusion.apply_shallow_fusion(logits, context)
        
        # Both should be different from original
        self.assertFalse(torch.allclose(logits, biased_logits_shallow))
        self.assertFalse(torch.allclose(logits, biased_logits_interp))
        self.assertFalse(torch.allclose(biased_logits_shallow, biased_logits_interp))
        
    def test_no_bias_when_disabled(self):
        """Test that no bias is applied when disabled"""
        patterns = create_test_patterns()
        similarities = [0.8, 0.7, 0.6]
        self.fusion.update_retrieved_patterns(patterns, similarities)
        
        # Disable fusion
        self.fusion.config.enabled = False
        
        context = ['KICK', 'BAR_1']
        logits = torch.randn(len(self.vocab_dict))
        biased_logits = self.fusion.apply_shallow_fusion(logits, context)
        
        # Should be identical when disabled
        self.assertTrue(torch.allclose(logits, biased_logits))


@unittest.skipUnless(FAISS_TESTS_AVAILABLE, "FAISS not available")
class TestStylePatternIndex(unittest.TestCase):
    """Test FAISS-based style pattern index"""
    
    def setUp(self):
        self.embedding_dim = 128
        self.index = StylePatternIndex(
            embedding_dim=self.embedding_dim,
            index_type="Flat",  # Use flat for deterministic tests
            nlist=10,
            nprobe=5
        )
        
    def test_index_creation(self):
        """Test index creation"""
        self.assertEqual(self.index.embedding_dim, 128)
        self.assertEqual(len(self.index.styles), 3)
        self.assertIn("rock_punk", self.index.styles)
        
    def test_pattern_addition(self):
        """Test adding patterns to index"""
        # Create test pattern
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        pattern = PatternEmbedding(
            pattern_id="test_pattern",
            pattern_tokens=["KICK", "SNARE", "BASS"],
            embedding=embedding,
            style="rock_punk",
            bars=2
        )
        
        self.index.add_pattern(pattern)
        
        # Check pattern was added
        self.assertEqual(len(self.index.patterns["rock_punk"]), 1)
        self.assertEqual(self.index.patterns["rock_punk"][0].pattern_id, "test_pattern")
        
    def test_pattern_search(self):
        """Test searching for similar patterns"""
        # Add multiple patterns
        for i in range(10):
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            pattern = PatternEmbedding(
                pattern_id=f"pattern_{i}",
                pattern_tokens=[f"TOKEN_{j}" for j in range(3)],
                embedding=embedding,
                style="rock_punk",
                bars=2
            )
            self.index.add_pattern(pattern)
            
        # Search with a query
        query = np.random.randn(self.embedding_dim).astype(np.float32)
        results = self.index.search(query, "rock_punk", top_k=3)
        
        # Should return up to 3 results
        self.assertLessEqual(len(results), 3)
        
        # Results should be tuples of (pattern, similarity)
        for pattern, similarity in results:
            self.assertIsInstance(pattern, PatternEmbedding)
            self.assertIsInstance(similarity, float)
            self.assertEqual(pattern.style, "rock_punk")
            
    def test_multi_style_search(self):
        """Test searching across multiple styles"""
        # Add patterns to different styles
        for style in ["rock_punk", "rnb_ballad"]:
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            pattern = PatternEmbedding(
                pattern_id=f"{style}_pattern",
                pattern_tokens=["TOKEN_A", "TOKEN_B"],
                embedding=embedding,
                style=style,
                bars=1
            )
            self.index.add_pattern(pattern)
            
        # Multi-style search
        query = np.random.randn(self.embedding_dim).astype(np.float32)
        results = self.index.multi_style_search(query, top_k=2)
        
        self.assertIn("rock_punk", results)
        self.assertIn("rnb_ballad", results)
        
    def test_save_and_load(self):
        """Test saving and loading the index"""
        # Add some patterns
        for i in range(5):
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            pattern = PatternEmbedding(
                pattern_id=f"pattern_{i}",
                pattern_tokens=[f"TOKEN_{i}"],
                embedding=embedding,
                style="rock_punk",
                bars=1
            )
            self.index.add_pattern(pattern)
            
        # Save to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            self.index.save(temp_dir)
            
            # Load and verify
            loaded_index = StylePatternIndex.load(temp_dir)
            
            self.assertEqual(loaded_index.embedding_dim, self.embedding_dim)
            self.assertEqual(len(loaded_index.patterns["rock_punk"]), 5)
            
    def test_statistics(self):
        """Test getting index statistics"""
        # Add patterns with different bar counts
        for i, bars in enumerate([1, 2, 4, 2, 1]):
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            pattern = PatternEmbedding(
                pattern_id=f"pattern_{i}",
                pattern_tokens=[f"TOKEN_{i}"],
                embedding=embedding,
                style="rock_punk",
                bars=bars
            )
            self.index.add_pattern(pattern)
            
        stats = self.index.get_style_statistics()
        
        self.assertIn("rock_punk", stats)
        self.assertEqual(stats["rock_punk"]["count"], 5)
        self.assertEqual(stats["rock_punk"]["avg_bars"], 2.0)  # (1+2+4+2+1)/5


class TestMotifVerification(unittest.TestCase):
    """Test bias verification with synthetic motifs"""
    
    def setUp(self):
        self.vocab_dict = {
            'MOTIF_A': 0, 'MOTIF_B': 1, 'MOTIF_C': 2,
            'NOTE_1': 3, 'NOTE_2': 4, 'NOTE_3': 5,
            'CHORD_X': 6, 'CHORD_Y': 7, 'CHORD_Z': 8,
            '<UNK>': 9, '<PAD>': 10
        }
        self.vocab = TokenVocabulary(self.vocab_dict)
        self.config = RetrievalConfig(enabled=True, retrieval_weight=0.5, ngram_size=2)
        self.fusion = RetrievalFusion(self.vocab, self.config)
        
    def test_motif_continuation_bias(self):
        """Test that retrieved motifs bias toward correct continuations"""
        # Create synthetic motifs with clear patterns
        patterns = [
            {
                'pattern_id': 'motif_sequence_1',
                'tokens': ['MOTIF_A', 'NOTE_1', 'NOTE_2', 'CHORD_X'],
                'style': 'synthetic'
            },
            {
                'pattern_id': 'motif_sequence_2',
                'tokens': ['MOTIF_A', 'NOTE_1', 'NOTE_3', 'CHORD_Y'],
                'style': 'synthetic'
            },
            {
                'pattern_id': 'different_motif',
                'tokens': ['MOTIF_B', 'CHORD_Z', 'NOTE_1'],
                'style': 'synthetic'
            }
        ]
        
        similarities = [0.9, 0.8, 0.3]  # First two are more similar
        self.fusion.update_retrieved_patterns(patterns, similarities)
        
        # Test context that should bias toward NOTE_1 (common in MOTIF_A patterns)
        context = ['MOTIF_A']
        bias = self.fusion.compute_retrieval_bias(context)
        
        # Should have high bias for NOTE_1
        note_1_id = self.vocab_dict['NOTE_1']
        note_2_id = self.vocab_dict['NOTE_2']
        note_3_id = self.vocab_dict['NOTE_3']
        
        # NOTE_1 should have highest bias (appears in both high-similarity patterns)
        self.assertGreater(bias[note_1_id].item(), bias[note_2_id].item())
        self.assertGreater(bias[note_1_id].item(), bias[note_3_id].item())
        
    def test_bias_strength_correlation(self):
        """Test that bias strength correlates with pattern similarity"""
        # Pattern with very high similarity
        high_sim_pattern = {
            'pattern_id': 'high_sim',
            'tokens': ['MOTIF_A', 'NOTE_1', 'CHORD_X'],
            'style': 'test'
        }
        
        # Pattern with lower similarity
        low_sim_pattern = {
            'pattern_id': 'low_sim', 
            'tokens': ['MOTIF_A', 'NOTE_2', 'CHORD_Y'],
            'style': 'test'
        }
        
        # Test with high similarity patterns
        self.fusion.update_retrieved_patterns([high_sim_pattern], [0.95])
        context = ['MOTIF_A']
        high_bias = self.fusion.compute_retrieval_bias(context)
        note_1_high_bias = high_bias[self.vocab_dict['NOTE_1']].item()
        
        # Test with low similarity patterns
        self.fusion.update_retrieved_patterns([low_sim_pattern], [0.3])
        low_bias = self.fusion.compute_retrieval_bias(context)
        note_2_low_bias = low_bias[self.vocab_dict['NOTE_2']].item()
        
        # High similarity should produce stronger bias
        self.assertGreater(note_1_high_bias, note_2_low_bias)
        
    def test_bias_decay_with_position(self):
        """Test that bias decays for patterns ranked lower"""
        patterns = [
            {'pattern_id': 'rank_1', 'tokens': ['MOTIF_A', 'NOTE_1'], 'style': 'test'},
            {'pattern_id': 'rank_2', 'tokens': ['MOTIF_A', 'NOTE_2'], 'style': 'test'},
            {'pattern_id': 'rank_3', 'tokens': ['MOTIF_A', 'NOTE_3'], 'style': 'test'}
        ]
        
        # Equal similarities but different ranking
        similarities = [0.8, 0.8, 0.8]
        self.fusion.update_retrieved_patterns(patterns, similarities)
        
        context = ['MOTIF_A']
        bias = self.fusion.compute_retrieval_bias(context)
        
        note_1_bias = bias[self.vocab_dict['NOTE_1']].item()
        note_2_bias = bias[self.vocab_dict['NOTE_2']].item()
        note_3_bias = bias[self.vocab_dict['NOTE_3']].item()
        
        # Should decay due to decay_factor (0.9^position)
        self.assertGreater(note_1_bias, note_2_bias)
        self.assertGreater(note_2_bias, note_3_bias)


def run_all_tests():
    """Run all test suites"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestStyleEncoder))
    suite.addTest(unittest.makeSuite(TestNGramMatcher))
    suite.addTest(unittest.makeSuite(TestRetrievalFusion))
    suite.addTest(unittest.makeSuite(TestMotifVerification))
    
    if FAISS_TESTS_AVAILABLE:
        suite.addTest(unittest.makeSuite(TestStylePatternIndex))
    else:
        print("Skipping FAISS tests - FAISS not available")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)