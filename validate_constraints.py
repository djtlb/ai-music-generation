#!/usr/bin/env python3
"""
Validation script for decoding constraints

Quick smoke test to verify all functions can be imported and work with toy data.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, '/workspaces/spark-template')

try:
    import torch
    print("✓ PyTorch available")
except ImportError:
    print("⚠ PyTorch not available, using mock")
    # Create minimal torch mock for testing
    class MockTensor:
        def __init__(self, data):
            if isinstance(data, (list, tuple)):
                self.data = data
                self.shape = (len(data),)
            else:
                self.data = [data]
                self.shape = (1,)
        
        def __getitem__(self, idx):
            return self.data[idx]
        
        def __setitem__(self, idx, val):
            self.data[idx] = val
            
        def __len__(self):
            return len(self.data)
            
        def clone(self):
            return MockTensor(self.data.copy())
            
        def __add__(self, other):
            if hasattr(other, 'data'):
                return MockTensor([a + b for a, b in zip(self.data, other.data)])
            return MockTensor([a + other for a in self.data])
            
        def __mul__(self, other):
            if hasattr(other, 'data'):
                return MockTensor([a * b for a, b in zip(self.data, other.data)])
            return MockTensor([a * other for a in self.data])
            
        def __truediv__(self, other):
            if hasattr(other, 'data'):
                return MockTensor([a / b for a, b in zip(self.data, other.data)])
            return MockTensor([a / other for a in self.data])
            
        def __lt__(self, other):
            if hasattr(other, 'data'):
                return all(a < b for a, b in zip(self.data, other.data))
            return all(a < other for a in self.data)
    
    class torch:
        @staticmethod
        def ones(size, dtype=None):
            if isinstance(size, int):
                return MockTensor([1.0] * size)
            return MockTensor([1.0] * size[0])
        
        @staticmethod
        def zeros(size, dtype=None):
            if isinstance(size, int):
                return MockTensor([0.0] * size)
            return MockTensor([0.0] * size[0])
        
        @staticmethod
        def randn(size):
            import random
            if isinstance(size, int):
                return MockTensor([random.gauss(0, 1) for _ in range(size)])
            return MockTensor([random.gauss(0, 1) for _ in range(size[0])])
        
        @staticmethod
        def where(condition, x, y):
            result = []
            for i in range(len(condition.data)):
                if condition.data[i]:
                    result.append(x)
                else:
                    result.append(y)
            return MockTensor(result)
        
        bool = bool
        float = float

def test_basic_imports():
    """Test that all functions can be imported"""
    try:
        from decoding.constraints import (
            section_mask, key_mask, groove_mask, 
            repetition_penalty, apply_all
        )
        print("✓ All constraint functions imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality with toy data"""
    try:
        from decoding.constraints import (
            section_mask, key_mask, groove_mask, 
            repetition_penalty, apply_all
        )
        
        # Create toy vocabulary
        vocab = {
            'PAD': 0, 'EOS': 1, 'BAR': 2,
            'NOTE_ON_60': 3, 'NOTE_ON_62': 4, 'NOTE_ON_64': 5,
            'KICK': 6, 'SNARE': 7, 'LEAD': 8, 'VOCAL': 9
        }
        
        # Mock tokens class
        class TestTokens:
            def __init__(self, vocab):
                self.vocab = vocab
                self.shape = (len(vocab),)
            def __len__(self):
                return len(self.vocab)
        
        tokens = TestTokens(vocab)
        
        # Test section_mask
        plan = {
            'sections': [{'type': 'INTRO', 'bars': 4}],
            'vocab': vocab
        }
        mask = section_mask(tokens, bar_idx=1, plan=plan)
        print("✓ section_mask works")
        
        # Test key_mask  
        weights = key_mask(tokens, key='C', tolerance=1)
        print("✓ key_mask works")
        
        # Test groove_mask
        groove_template = {
            'drum_pattern': {},
            'time_feel': 'straight',
            'current_pos': 0
        }
        groove_weights = groove_mask(tokens, groove_template)
        print("✓ groove_mask works")
        
        # Test repetition_penalty
        logits = torch.randn(len(vocab))
        history = [3, 3, 4]  # Some repeated tokens
        penalized = repetition_penalty(logits, history, gamma=1.2)
        print("✓ repetition_penalty works")
        
        # Test apply_all
        state = {
            'bar_idx': 1,
            'history': [3, 4],
            'current_pos': 0
        }
        result = apply_all(logits, state, plan)
        print("✓ apply_all works")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests"""
    print("=== Decoding Constraints Validation ===\n")
    
    success = True
    
    print("1. Testing imports...")
    success &= test_basic_imports()
    
    print("\n2. Testing basic functionality...")  
    success &= test_basic_functionality()
    
    print(f"\n=== Results ===")
    if success:
        print("✓ All tests passed! Constraints module is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())