"""
Unit tests for hierarchical FAISS index and retrieval fusion.
Tests pattern loading, indexing, retrieval, and bias application.
"""

import unittest
import tempfile
import json
import shutil
from pathlib import Path
import numpy as np

from style.faiss_index import HierarchicalFAISSIndex, StylePattern, build_hierarchical_indices
from style.retrieval_fusion import RetrievalFusion, RetrievalConfig


class TestHierarchicalFAISSIndex(unittest.TestCase):
    """Test FAISS index building and pattern retrieval."""
    
    def setUp(self):
        """Create temporary style packs structure for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.style_packs_dir = Path(self.temp_dir) / "style_packs"
        
        # Create test style pack structure
        self._create_test_style_packs()
        
        self.index = HierarchicalFAISSIndex(embedding_dim=128)
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_style_packs(self):
        """Create test style pack directory structure with sample data."""
        # Pop parent genre
        pop_dir = self.style_packs_dir / "pop"
        pop_refs_midi = pop_dir / "refs_midi"
        pop_refs_midi.mkdir(parents=True)
        
        # Pop parent patterns
        pop_tokens = ["STYLE=pop", "TEMPO=120", "KEY=C", "SECTION=VERSE", 
                     "BAR", "POS=1", "CHORD=C", "NOTE_ON", "60", "VEL=80"]
        
        with open(pop_refs_midi / "pop_sample1.tokens", 'w') as f:
            f.write(' '.join(pop_tokens))
            
        # Pop JSON format
        pop_data = {
            "tokenized_bars": [
                ["STYLE=pop", "TEMPO=120", "CHORD=C", "NOTE_ON", "60", "VEL=80"],
                ["STYLE=pop", "TEMPO=120", "CHORD=F", "NOTE_ON", "65", "VEL=75"]
            ]
        }
        
        with open(pop_refs_midi / "pop_sample2.json", 'w') as f:
            json.dump(pop_data, f)
        
        # Dance pop child genre
        dance_pop_dir = pop_dir / "dance_pop"
        dance_pop_refs_midi = dance_pop_dir / "refs_midi"
        dance_pop_refs_midi.mkdir(parents=True)
        
        # Dance pop child patterns (with slight variations)
        dance_pop_tokens = ["STYLE=dance_pop", "TEMPO=128", "KEY=C", "SECTION=CHORUS",
                           "BAR", "POS=1", "CHORD=C", "NOTE_ON", "60", "VEL=90"]
        
        with open(dance_pop_refs_midi / "dance_pop_sample1.tokens", 'w') as f:
            f.write(' '.join(dance_pop_tokens))
        
        # Rock parent genre  
        rock_dir = self.style_packs_dir / "rock"
        rock_refs_midi = rock_dir / "refs_midi"
        rock_refs_midi.mkdir(parents=True)
        
        rock_tokens = ["STYLE=rock", "TEMPO=140", "KEY=E", "SECTION=VERSE",
                      "BAR", "POS=1", "CHORD=E", "NOTE_ON", "64", "VEL=100"]
        
        with open(rock_refs_midi / "rock_sample1.tokens", 'w') as f:
            f.write(' '.join(rock_tokens))
    
    def test_load_parent_patterns(self):
        """Test loading patterns from parent refs_midi directory."""
        pop_dir = self.style_packs_dir / "pop"
        patterns = self.index._load_parent_patterns(pop_dir, "pop")
        
        # Should load from both .tokens and .json files
        self.assertGreater(len(patterns), 0)
        
        # Check pattern properties
        for pattern in patterns:
            self.assertEqual(pattern.parent_genre, "pop")
            self.assertIsNone(pattern.child_genre)
            self.assertIsInstance(pattern.tokens, list)
            self.assertTrue(len(pattern.tokens) > 0)
    
    def test_load_child_patterns(self):
        """Test loading patterns from child refs_midi directory."""
        dance_pop_dir = self.style_packs_dir / "pop" / "dance_pop"
        patterns = self.index._load_child_patterns(dance_pop_dir, "pop", "dance_pop", 1.5)
        
        self.assertGreater(len(patterns), 0)
        
        # Check child pattern properties
        for pattern in patterns:
            self.assertEqual(pattern.parent_genre, "pop")
            self.assertEqual(pattern.child_genre, "dance_pop")
            self.assertEqual(pattern.weight, 1.5)
    
    def test_create_pattern_embeddings(self):
        """Test embedding creation for patterns."""
        # Create test patterns
        patterns = [
            StylePattern(
                tokens=["STYLE=pop", "TEMPO=120", "CHORD=C"],
                embedding=np.zeros(128),
                parent_genre="pop"
            ),
            StylePattern(
                tokens=["STYLE=pop", "TEMPO=128", "CHORD=F"],
                embedding=np.zeros(128),
                parent_genre="pop"
            )
        ]
        
        embeddings = self.index._create_pattern_embeddings(patterns)
        
        self.assertEqual(embeddings.shape[0], len(patterns))
        self.assertEqual(embeddings.shape[1], self.index.embedding_dim)
        
        # Embeddings should be normalized
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_build_parent_indices(self):
        """Test building FAISS indices for parent genres."""
        self.index.build_parent_indices(str(self.style_packs_dir))
        
        # Should have indices for pop and rock
        self.assertIn("pop", self.index.parent_indices)
        self.assertIn("rock", self.index.parent_indices)
        
        # Check index properties
        pop_index = self.index.parent_indices["pop"]
        self.assertEqual(pop_index.d, self.index.embedding_dim)
        self.assertGreater(pop_index.ntotal, 0)
    
    def test_register_child_patterns(self):
        """Test registering child patterns with higher weights."""
        # First build parent indices
        self.index.build_parent_indices(str(self.style_packs_dir))
        
        # Register child patterns
        self.index.register_child_patterns("pop", "dance_pop", str(self.style_packs_dir), 2.0)
        
        child_key = "pop/dance_pop"
        self.assertIn(child_key, self.index.child_patterns)
        
        # Check child pattern weights
        child_patterns = self.index.child_patterns[child_key]
        for pattern in child_patterns:
            self.assertEqual(pattern.weight, 2.0)
    
    def test_retrieve_similar_patterns(self):
        """Test pattern retrieval with parent and child bias."""
        # Build indices
        self.index.build_parent_indices(str(self.style_packs_dir))
        self.index.register_child_patterns("pop", "dance_pop", str(self.style_packs_dir), 1.5)
        
        # Test retrieval
        query_tokens = ["STYLE=pop", "TEMPO=120", "CHORD=C"]
        results = self.index.retrieve_similar_patterns(
            query_tokens=query_tokens,
            parent_genre="pop",
            k=3
        )
        
        self.assertGreater(len(results), 0)
        
        # Check result format
        for pattern, score in results:
            self.assertIsInstance(pattern, StylePattern)
            self.assertIsInstance(score, float)
        
        # Test with child bias
        results_with_bias = self.index.retrieve_similar_patterns(
            query_tokens=query_tokens,
            parent_genre="pop",
            child_genre="dance_pop",
            child_bias=0.3,
            k=3
        )
        
        self.assertGreater(len(results_with_bias), 0)
    
    def test_save_and_load_indices(self):
        """Test saving and loading FAISS indices."""
        # Build indices
        self.index.build_parent_indices(str(self.style_packs_dir))
        self.index.register_child_patterns("pop", "dance_pop", str(self.style_packs_dir))
        
        # Save
        save_dir = Path(self.temp_dir) / "saved_indices"
        self.index.save_indices(str(save_dir))
        
        self.assertTrue(save_dir.exists())
        self.assertTrue((save_dir / "pop.faiss").exists())
        self.assertTrue((save_dir / "patterns.pkl").exists())
        
        # Load into new index
        new_index = HierarchicalFAISSIndex(embedding_dim=128)
        new_index.load_indices(str(save_dir))
        
        # Verify loaded data
        self.assertEqual(len(new_index.parent_indices), len(self.index.parent_indices))
        self.assertEqual(len(new_index.parent_patterns), len(self.index.parent_patterns))
        self.assertEqual(len(new_index.child_patterns), len(self.index.child_patterns))


class TestRetrievalFusion(unittest.TestCase):
    """Test retrieval fusion for biasing token generation."""
    
    def setUp(self):
        """Setup test retrieval fusion system."""
        # Create test vocabulary
        self.vocab = {
            "STYLE=pop": 0, "TEMPO=120": 1, "CHORD=C": 2, "NOTE_ON": 3,
            "60": 4, "VEL=80": 5, "BAR": 6, "POS=1": 7, "<EOS>": 8, "<UNK>": 9
        }
        
        # Create test FAISS index
        self.faiss_index = HierarchicalFAISSIndex(embedding_dim=64)
        
        # Create test patterns
        self._create_test_patterns()
        
        # Create config
        self.config = RetrievalConfig(
            family_index="pop",
            child_bias=0.3,
            child_genre="dance_pop",
            fusion_weight=0.2,
            ngram_size=3
        )
        
        self.retrieval_fusion = RetrievalFusion(self.faiss_index, self.vocab, self.config)
    
    def _create_test_patterns(self):
        """Create test patterns for retrieval fusion testing."""
        import faiss
        
        # Parent patterns
        parent_patterns = [
            StylePattern(
                tokens=["STYLE=pop", "TEMPO=120", "CHORD=C", "NOTE_ON", "60"],
                embedding=np.random.randn(64),
                parent_genre="pop"
            ),
            StylePattern(
                tokens=["TEMPO=120", "CHORD=C", "NOTE_ON", "VEL=80", "BAR"],
                embedding=np.random.randn(64),
                parent_genre="pop"
            )
        ]
        
        # Child patterns with higher weights
        child_patterns = [
            StylePattern(
                tokens=["STYLE=pop", "TEMPO=120", "CHORD=C", "VEL=80", "POS=1"],
                embedding=np.random.randn(64),
                parent_genre="pop",
                child_genre="dance_pop",
                weight=1.5
            )
        ]
        
        # Create FAISS index
        embeddings = np.array([p.embedding for p in parent_patterns]).astype(np.float32)
        index = faiss.IndexFlatIP(64)
        index.add(embeddings)
        
        self.faiss_index.parent_indices["pop"] = index
        self.faiss_index.parent_patterns["pop"] = parent_patterns
        self.faiss_index.child_patterns["pop/dance_pop"] = child_patterns
    
    def test_build_pattern_ngrams(self):
        """Test n-gram cache building from patterns."""
        # Access private method to test n-gram building
        self.retrieval_fusion._build_pattern_ngrams()
        
        # Should have cached n-grams for parent and child
        self.assertIn("parent", self.retrieval_fusion._pattern_ngrams_cache)
        self.assertIn("child", self.retrieval_fusion._pattern_ngrams_cache)
        
        # Check n-gram structure
        parent_ngrams = self.retrieval_fusion._pattern_ngrams_cache["parent"]
        self.assertIsInstance(parent_ngrams, dict)
    
    def test_get_retrieval_distribution(self):
        """Test next-token distribution from retrieved patterns."""
        # Build n-gram cache
        self.retrieval_fusion._build_pattern_ngrams()
        
        # Test with an n-gram that should be in patterns
        test_ngram = ("STYLE=pop", "TEMPO=120", "CHORD=C")
        distribution = self.retrieval_fusion._get_retrieval_distribution(test_ngram)
        
        if distribution:  # May be None if n-gram not found
            self.assertIsInstance(distribution, dict)
            
            # Check probability distribution properties
            total_prob = sum(distribution.values())
            self.assertAlmostEqual(total_prob, 1.0, places=5)
    
    def test_apply_retrieval_bias(self):
        """Test applying retrieval bias to model logits."""
        import torch
        
        # Create mock logits
        vocab_size = len(self.vocab)
        logits = torch.randn(vocab_size)
        
        # Create generated token sequence
        generated_tokens = [0, 1, 2]  # STYLE=pop TEMPO=120 CHORD=C
        
        # Apply bias
        biased_logits = self.retrieval_fusion.apply_retrieval_bias(logits, generated_tokens)
        
        # Should return tensor of same shape
        self.assertEqual(biased_logits.shape, logits.shape)
        self.assertIsInstance(biased_logits, torch.Tensor)
    
    def test_retrieve_similar_patterns_context(self):
        """Test retrieving similar patterns for broader context."""
        query_tokens = ["STYLE=pop", "TEMPO=120"]
        patterns = self.retrieval_fusion.get_similar_patterns_context(query_tokens, k=2)
        
        # Should return list of patterns
        self.assertIsInstance(patterns, list)
        
        for pattern in patterns:
            self.assertIsInstance(pattern, StylePattern)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete retrieval system."""
    
    def test_build_hierarchical_indices_function(self):
        """Test the main index building function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal style pack structure
            style_packs_dir = Path(temp_dir) / "style_packs"
            pop_refs_midi = style_packs_dir / "pop" / "refs_midi"
            pop_refs_midi.mkdir(parents=True)
            
            # Create test file
            with open(pop_refs_midi / "test.tokens", 'w') as f:
                f.write("STYLE=pop TEMPO=120 CHORD=C")
            
            # Build indices
            index = build_hierarchical_indices(
                style_packs_dir=str(style_packs_dir),
                output_dir=str(Path(temp_dir) / "indices"),
                embedding_dim=64
            )
            
            self.assertIsInstance(index, HierarchicalFAISSIndex)
            self.assertIn("pop", index.parent_indices)


if __name__ == "__main__":
    unittest.main()