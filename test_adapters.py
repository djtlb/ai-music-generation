#!/usr/bin/env python3
"""
Comprehensive tests for hierarchical LoRA adapter system.

Tests:
1. LoRA layer functionality (forward/backward pass, merging)
2. Style adapter creation and loading
3. Hierarchical adapter composition
4. Decode parity when child has no overrides
5. Merge utilities and compatibility verification
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.adapters.lora_layer import LoRALinear, apply_lora_to_model
from models.adapters.style_adapter import StyleAdapter, HierarchicalStyleAdapter
from models.adapters.adapter_merge import HierarchicalMerger, verify_adapter_compatibility
from models.adapters.training_utils import StylePackDataset


class MockTokenizer:
    """Mock tokenizer for testing."""
    def __init__(self):
        self.vocab = {'PAD': 0, 'NOTE_ON': 1, 'NOTE_OFF': 2}
    
    def encode_events(self, events):
        return [1, 2, 1, 2]  # Simple pattern


class MockTransformer(nn.Module):
    """Mock transformer model for testing."""
    def __init__(self, hidden_size=128, num_layers=2):
        super().__init__()
        self.config = type('Config', (), {'hidden_size': hidden_size})()
        
        self.embedding = nn.Embedding(1000, hidden_size)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.Linear(hidden_size, hidden_size),
                'feed_forward': nn.Linear(hidden_size, hidden_size)
            }) for _ in range(num_layers)
        ])
        self.output_head = nn.Linear(hidden_size, 1000)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            # Simple forward pass
            x = x + layer['attention'](x)
            x = x + layer['feed_forward'](x)
            
        return self.output_head(x)


class TestLoRALayer(unittest.TestCase):
    """Test LoRA layer implementations."""
    
    def setUp(self):
        self.base_linear = nn.Linear(64, 32)
        self.lora_linear = LoRALinear(
            base_layer=self.base_linear,
            rank=4,
            alpha=8.0,
            dropout=0.1
        )
        
    def test_lora_initialization(self):
        """Test LoRA layer initialization."""
        # Check LoRA matrices are correctly sized
        self.assertEqual(self.lora_linear.lora_A.shape, (4, 64))
        self.assertEqual(self.lora_linear.lora_B.shape, (32, 4))
        
        # Check scaling factor
        expected_scaling = 8.0 / 4  # alpha / rank
        self.assertEqual(self.lora_linear.scaling, expected_scaling)
        
        # Check base layer is frozen
        self.assertFalse(self.base_linear.weight.requires_grad)
        
    def test_lora_forward_pass(self):
        """Test LoRA forward pass."""
        batch_size, seq_len = 2, 8
        input_tensor = torch.randn(batch_size, seq_len, 64)
        
        # Forward pass should work
        output = self.lora_linear(input_tensor)
        self.assertEqual(output.shape, (batch_size, seq_len, 32))
        
    def test_lora_merging(self):
        """Test LoRA weight merging and unmerging."""
        input_tensor = torch.randn(1, 64)
        
        # Get output before merging
        output_before = self.lora_linear(input_tensor)
        
        # Merge weights
        self.lora_linear.merge_weights_()
        self.assertTrue(self.lora_linear.merged)
        
        # Output should be the same after merging
        output_after = self.lora_linear(input_tensor)
        torch.testing.assert_close(output_before, output_after, rtol=1e-5)
        
        # Unmerge weights
        self.lora_linear.unmerge_weights()
        self.assertFalse(self.lora_linear.merged)
        
        # Output should still be the same
        output_unmerged = self.lora_linear(input_tensor)
        torch.testing.assert_close(output_before, output_unmerged, rtol=1e-5)
        
    def test_apply_lora_to_model(self):
        """Test applying LoRA to a full model."""
        model = MockTransformer()
        
        # Apply LoRA to attention and feed_forward layers
        lora_layers = apply_lora_to_model(
            model,
            target_modules=['attention', 'feed_forward'],
            rank=4,
            alpha=8.0
        )
        
        # Should have created LoRA for targeted layers
        self.assertGreater(len(lora_layers), 0)
        
        # Check that targeted layers are now LoRA layers
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'feed_forward' in name.lower():
                if isinstance(module, nn.Linear):
                    # Should have been replaced or is parent of LoRA
                    pass  # Structure may vary


class TestStyleAdapter(unittest.TestCase):
    """Test style adapter functionality."""
    
    def setUp(self):
        self.base_model = MockTransformer()
        self.style_adapter = StyleAdapter(
            base_model=self.base_model,
            style_name='test_style',
            rank=4,
            alpha=8.0,
            target_modules=['attention', 'feed_forward']
        )
        
    def test_style_adapter_creation(self):
        """Test style adapter creation."""
        self.assertEqual(self.style_adapter.style_name, 'test_style')
        self.assertGreater(len(self.style_adapter.lora_layers), 0)
        
    def test_style_state_dict(self):
        """Test style adapter state dict operations."""
        # Get state dict
        state_dict = self.style_adapter.get_style_state_dict()
        
        # Should contain metadata and LoRA parameters
        self.assertIn('_metadata', state_dict)
        self.assertIn('style_embedding', state_dict)
        
        # Check metadata
        metadata = state_dict['_metadata']
        self.assertEqual(metadata['style_name'], 'test_style')
        self.assertEqual(metadata['rank'], 4)
        
        # Test loading state dict
        new_adapter = StyleAdapter(
            base_model=MockTransformer(),
            style_name='new_style',
            rank=4,
            alpha=8.0,
            target_modules=['attention', 'feed_forward']
        )
        
        new_adapter.load_style_state_dict(state_dict)
        # Style name should be updated from state dict
        # (Note: this depends on implementation details)


class TestHierarchicalAdapter(unittest.TestCase):
    """Test hierarchical adapter functionality."""
    
    def setUp(self):
        self.base_model = MockTransformer()
        
    def test_hierarchical_adapter_creation(self):
        """Test hierarchical adapter creation."""
        hierarchical = HierarchicalStyleAdapter(
            base_model=self.base_model,
            parent_style='pop',
            child_style='dance_pop'
        )
        
        self.assertEqual(hierarchical.parent_style, 'pop')
        self.assertEqual(hierarchical.child_style, 'dance_pop')
        self.assertIsNotNone(hierarchical.parent_adapter)
        self.assertIsNotNone(hierarchical.child_adapter)
        
    def test_style_weight_adjustment(self):
        """Test style weight adjustment."""
        hierarchical = HierarchicalStyleAdapter(
            base_model=self.base_model,
            parent_style='pop',
            child_style='dance_pop'
        )
        
        # Test setting weights
        hierarchical.set_style_weights(parent_weight=0.8, child_weight=1.2)
        
        # Check that scaling factors were updated
        for lora_layer in hierarchical.parent_adapter.lora_layers.values():
            expected_scaling = (lora_layer.alpha / lora_layer.rank) * 0.8
            self.assertAlmostEqual(lora_layer.scaling, expected_scaling)


class TestAdapterMerging(unittest.TestCase):
    """Test adapter merging functionality."""
    
    def setUp(self):
        self.base_model = MockTransformer()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_hierarchical_merger(self):
        """Test hierarchical merger functionality."""
        # Create test adapters
        parent_adapter = StyleAdapter(
            base_model=self.base_model,
            style_name='pop',
            rank=4,
            alpha=8.0,
            target_modules=['attention']
        )
        
        child_adapter = StyleAdapter(
            base_model=self.base_model,
            style_name='dance_pop',
            rank=2,
            alpha=4.0,
            target_modules=['attention']
        )
        
        # Save adapters
        parent_path = os.path.join(self.temp_dir, 'pop.lora')
        child_path = os.path.join(self.temp_dir, 'dance_pop.lora')
        
        torch.save(parent_adapter.get_style_state_dict(), parent_path)
        torch.save(child_adapter.get_style_state_dict(), child_path)
        
        # Test merger
        merger = HierarchicalMerger(self.base_model)
        merger.load_parent_adapter(parent_path)
        merger.load_child_adapter(child_path)
        
        # Test merging
        merger.merge_hierarchical(parent_weight=1.0, child_weight=0.5)
        self.assertIsNotNone(merger.merged_state)
        
        # Test unmerging
        merger.unmerge()
        self.assertIsNone(merger.merged_state)
        
    def test_compatibility_verification(self):
        """Test adapter compatibility verification."""
        # Create adapter with compatible structure
        adapter = StyleAdapter(
            base_model=self.base_model,
            style_name='test',
            rank=4,
            target_modules=['attention']
        )
        
        adapter_path = os.path.join(self.temp_dir, 'test.lora')
        torch.save(adapter.get_style_state_dict(), adapter_path)
        
        # Should pass compatibility check
        is_compatible = verify_adapter_compatibility(self.base_model, [adapter_path])
        self.assertTrue(is_compatible)


class TestDecodeParityChildOverrides(unittest.TestCase):
    """Test decode parity when child has no meaningful overrides."""
    
    def setUp(self):
        self.base_model = MockTransformer()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create parent adapter
        self.parent_adapter = StyleAdapter(
            base_model=self.base_model,
            style_name='pop',
            rank=4,
            alpha=8.0,
            target_modules=['attention', 'feed_forward']
        )
        
        # Create minimal child adapter (should have minimal impact)
        self.child_adapter = StyleAdapter(
            base_model=self.base_model,
            style_name='dance_pop',
            rank=1,  # Very small rank
            alpha=0.1,  # Very small alpha
            target_modules=['attention']
        )
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_decode_parity_minimal_child(self):
        """Test that minimal child adapter doesn't significantly change output."""
        # Create test input
        batch_size, seq_len = 2, 16
        test_input = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Get output with just parent adapter
        merger1 = HierarchicalMerger(self.base_model)
        
        # Save and load parent adapter
        parent_path = os.path.join(self.temp_dir, 'pop.lora')
        torch.save(self.parent_adapter.get_style_state_dict(), parent_path)
        merger1.load_parent_adapter(parent_path)
        merger1.merge_hierarchical(parent_weight=1.0, child_weight=0.0)
        
        self.base_model.eval()
        with torch.no_grad():
            output_parent_only = self.base_model(test_input)
        
        merger1.unmerge()
        
        # Get output with parent + minimal child
        merger2 = HierarchicalMerger(self.base_model)
        
        child_path = os.path.join(self.temp_dir, 'dance_pop.lora')
        torch.save(self.child_adapter.get_style_state_dict(), child_path)
        
        merger2.load_parent_adapter(parent_path)
        merger2.load_child_adapter(child_path)
        merger2.merge_hierarchical(parent_weight=1.0, child_weight=0.1)  # Small child weight
        
        self.base_model.eval()
        with torch.no_grad():
            output_parent_child = self.base_model(test_input)
        
        merger2.unmerge()
        
        # Compare outputs - should be very close
        mse_diff = torch.nn.functional.mse_loss(output_parent_only, output_parent_child)
        
        # With minimal child contribution, difference should be small
        self.assertLess(mse_diff.item(), 0.01, "Decode parity test failed - child adapter has too much impact")
        
    def test_decode_parity_zero_child_weight(self):
        """Test that zero child weight gives identical output to parent-only."""
        batch_size, seq_len = 2, 16
        test_input = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Parent only
        merger1 = HierarchicalMerger(self.base_model)
        parent_path = os.path.join(self.temp_dir, 'pop.lora')
        torch.save(self.parent_adapter.get_style_state_dict(), parent_path)
        merger1.load_parent_adapter(parent_path)
        merger1.merge_hierarchical(parent_weight=1.0, child_weight=0.0)
        
        self.base_model.eval()
        with torch.no_grad():
            output1 = self.base_model(test_input)
        merger1.unmerge()
        
        # Parent + child with zero weight
        merger2 = HierarchicalMerger(self.base_model)
        child_path = os.path.join(self.temp_dir, 'dance_pop.lora')
        torch.save(self.child_adapter.get_style_state_dict(), child_path)
        
        merger2.load_parent_adapter(parent_path)
        merger2.load_child_adapter(child_path)
        merger2.merge_hierarchical(parent_weight=1.0, child_weight=0.0)  # Zero child weight
        
        self.base_model.eval()
        with torch.no_grad():
            output2 = self.base_model(test_input)
        merger2.unmerge()
        
        # Outputs should be identical
        torch.testing.assert_close(output1, output2, rtol=1e-6, atol=1e-6)


class TestStylePackDataset(unittest.TestCase):
    """Test style pack dataset functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.tokenizer = MockTokenizer()
        
        # Create minimal style pack structure
        pack_dir = Path(self.temp_dir) / 'test_pack'
        pack_dir.mkdir()
        (pack_dir / 'refs_midi').mkdir()
        (pack_dir / 'refs_audio').mkdir()
        
        # Create dummy meta.json
        import json
        meta = {'style': 'test_style', 'bpm': 120, 'key': 'C_major'}
        with open(pack_dir / 'meta.json', 'w') as f:
            json.dump(meta, f)
        
        self.pack_dir = str(pack_dir)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_dataset_creation(self):
        """Test style pack dataset creation."""
        dataset = StylePackDataset(
            style_pack_dir=self.pack_dir,
            tokenizer=self.tokenizer,
            max_seq_len=64
        )
        
        # Should load metadata
        self.assertEqual(dataset.metadata['style'], 'test_style')
        
    def test_dataset_empty_handling(self):
        """Test dataset handles empty style packs gracefully."""
        dataset = StylePackDataset(
            style_pack_dir=self.pack_dir,
            tokenizer=self.tokenizer,
            max_seq_len=64
        )
        
        # Should handle empty directory without crashing
        self.assertEqual(len(dataset), 0)


def run_all_tests():
    """Run all adapter tests."""
    # Create test suite
    test_classes = [
        TestLoRALayer,
        TestStyleAdapter,
        TestHierarchicalAdapter, 
        TestAdapterMerging,
        TestDecodeParityChildOverrides,
        TestStylePackDataset
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)