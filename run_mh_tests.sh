#!/bin/bash

# Test runner for Melody & Harmony Transformer components
# Validates that all modules can be imported and basic functionality works

echo "🎵 Testing Melody & Harmony Transformer Implementation"
echo "======================================================"

# Check if Python files can be parsed
echo "📁 Checking Python file syntax..."

# Function to check Python syntax
check_syntax() {
    local file=$1
    if [ -f "$file" ]; then
        echo -n "  Checking $file... "
        if python3 -m py_compile "$file" 2>/dev/null; then
            echo "✅ OK"
            return 0
        else
            echo "❌ SYNTAX ERROR"
            python3 -m py_compile "$file"
            return 1
        fi
    else
        echo "  ⚠️  File not found: $file"
        return 1
    fi
}

# Check main model files
check_syntax "/workspaces/spark-template/src/models/mh_transformer.py"
check_syntax "/workspaces/spark-template/src/utils/constraints.py"
check_syntax "/workspaces/spark-template/train_mh.py"
check_syntax "/workspaces/spark-template/sample_mh.py"

echo ""
echo "📊 Checking test files..."
check_syntax "/workspaces/spark-template/tests/test_constraints.py"
check_syntax "/workspaces/spark-template/tests/test_mh_transformer.py"

echo ""
echo "📝 Checking configuration files..."

# Check YAML configuration files
echo -n "  Checking mh_transformer.yaml... "
if [ -f "/workspaces/spark-template/configs/mh_transformer.yaml" ]; then
    echo "✅ Found"
else
    echo "❌ Missing"
fi

echo -n "  Checking mh_sampling.yaml... "
if [ -f "/workspaces/spark-template/configs/mh_sampling.yaml" ]; then
    echo "✅ Found"
else
    echo "❌ Missing"
fi

echo ""
echo "🔍 Checking required imports..."

# Create a simple import test
cat > /tmp/test_imports.py << 'EOF'
import sys
import os
sys.path.append('/workspaces/spark-template/src')

try:
    import torch
    print("✅ PyTorch available")
except ImportError:
    print("❌ PyTorch not available")

try:
    import torch.nn as nn
    import torch.nn.functional as F
    print("✅ PyTorch neural network modules available")
except ImportError:
    print("❌ PyTorch neural network modules not available")

try:
    from typing import Dict, List, Optional, Tuple, Union
    print("✅ Typing modules available")
except ImportError:
    print("❌ Typing modules not available")

try:
    import json
    import math
    import numpy as np
    print("✅ Standard libraries available")
except ImportError:
    print("❌ Some standard libraries not available")

# Test model import
try:
    from models.mh_transformer import MelodyHarmonyTransformer, MHTrainingLoss
    print("✅ Melody & Harmony Transformer can be imported")
except ImportError as e:
    print(f"❌ Cannot import MH Transformer: {e}")

# Test constraints import
try:
    from utils.constraints import ConstraintMaskGenerator, RepetitionController
    print("✅ Constraint utilities can be imported")
except ImportError as e:
    print(f"❌ Cannot import constraints: {e}")

EOF

python3 /tmp/test_imports.py

echo ""
echo "🧪 Running basic functionality tests..."

# Create a simple functionality test
cat > /tmp/test_functionality.py << 'EOF'
import sys
import os
sys.path.append('/workspaces/spark-template/src')

def test_musical_constraints():
    """Test basic musical constraint functionality"""
    try:
        from utils.constraints import MusicalConstraints
        
        # Test scale generation
        c_major = MusicalConstraints.get_scale_notes(0, is_major=True)
        expected = [0, 2, 4, 5, 7, 9, 11]
        assert c_major == expected, f"Expected {expected}, got {c_major}"
        
        # Test chord generation
        c_maj_chord = MusicalConstraints.get_chord_notes(0, 'maj')
        expected_chord = [0, 4, 7]
        assert c_maj_chord == expected_chord, f"Expected {expected_chord}, got {c_maj_chord}"
        
        print("✅ Musical constraints basic functionality works")
        return True
    except Exception as e:
        print(f"❌ Musical constraints test failed: {e}")
        return False

def test_model_creation():
    """Test model can be created"""
    try:
        import torch
        from models.mh_transformer import MelodyHarmonyTransformer
        
        model = MelodyHarmonyTransformer(
            vocab_size=1000,
            d_model=256,
            nhead=8,
            num_layers=4,
            style_vocab_size=3
        )
        
        # Test forward pass shape
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        style_ids = torch.randint(0, 3, (batch_size,))
        key_ids = torch.randint(0, 24, (batch_size,))
        section_ids = torch.randint(0, 5, (batch_size,))
        
        with torch.no_grad():
            outputs = model(input_ids, style_ids, key_ids, section_ids)
        
        expected_shape = (batch_size, seq_len, 1000)
        actual_shape = outputs['logits'].shape
        assert actual_shape == expected_shape, f"Expected {expected_shape}, got {actual_shape}"
        
        print("✅ Model creation and forward pass works")
        return True
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False

def test_constraint_masks():
    """Test constraint mask generation"""
    try:
        from utils.constraints import ConstraintMaskGenerator
        
        vocab = {
            'PAD': 0, 'EOS': 1, 'BAR': 2,
            'NOTE_ON_60': 5, 'NOTE_ON_62': 6, 'NOTE_ON_64': 7,
            'CHORD_C_maj': 8, 'STYLE_rock_punk': 9
        }
        
        mask_gen = ConstraintMaskGenerator(vocab)
        
        # Test scale constraint
        scale_mask = mask_gen.create_scale_constraint_mask(key=0, seq_len=10)
        expected_shape = (10, len(vocab))
        assert scale_mask.shape == expected_shape, f"Expected {expected_shape}, got {scale_mask.shape}"
        
        print("✅ Constraint mask generation works")
        return True
    except Exception as e:
        print(f"❌ Constraint mask test failed: {e}")
        return False

# Run tests
success_count = 0
total_tests = 3

if test_musical_constraints():
    success_count += 1

if test_model_creation():
    success_count += 1
    
if test_constraint_masks():
    success_count += 1

print(f"\n📈 Test Results: {success_count}/{total_tests} tests passed")

if success_count == total_tests:
    print("🎉 All basic functionality tests passed!")
    exit(0)
else:
    print("⚠️  Some tests failed")
    exit(1)
EOF

python3 /tmp/test_functionality.py

echo ""
echo "🏗️  Checking directory structure..."

# Check if all required files exist
required_files=(
    "/workspaces/spark-template/src/models/mh_transformer.py"
    "/workspaces/spark-template/src/utils/constraints.py"
    "/workspaces/spark-template/train_mh.py"
    "/workspaces/spark-template/sample_mh.py"
    "/workspaces/spark-template/tests/test_constraints.py"
    "/workspaces/spark-template/tests/test_mh_transformer.py"
    "/workspaces/spark-template/configs/mh_transformer.yaml"
    "/workspaces/spark-template/configs/mh_sampling.yaml"
)

missing_files=0
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (missing)"
        ((missing_files++))
    fi
done

echo ""
echo "📊 Summary"
echo "=========="
echo "Required files: ${#required_files[@]}"
echo "Missing files: $missing_files"

if [ $missing_files -eq 0 ]; then
    echo "🎯 All implementation files are present!"
else
    echo "⚠️  Some files are missing"
fi

echo ""
echo "🎵 Melody & Harmony Transformer implementation check complete!"

# Clean up
rm -f /tmp/test_imports.py /tmp/test_functionality.py