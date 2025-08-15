# Hierarchical LoRA Adapter Training System

A sophisticated system for training hierarchical style adapters using Low-Rank Adaptation (LoRA) for music generation models. This enables efficient fine-tuning of large models for specific music styles while maintaining parameter efficiency.

## Overview

The system implements a hierarchical approach where:
- **Parent adapters** capture broad genre characteristics (e.g., pop, rock, country)
- **Child adapters** learn specific sub-style variations (e.g., dance_pop, indie_rock)
- **Merge utilities** combine adapters for inference with configurable blending

## Architecture

```
Base Model (frozen)
    ↓
Parent Adapter (e.g., "pop")         ← Training: entire style pack
    ↓
Child Adapter (e.g., "dance_pop")    ← Training: child-specific data
    ↓
Generated Output
```

### Key Components

- **LoRA Layers** (`src/models/adapters/lora_layer.py`): Low-rank adaptation implementation
- **Style Adapters** (`src/models/adapters/style_adapter.py`): Single and hierarchical style wrappers
- **Adapter Merging** (`src/models/adapters/adapter_merge.py`): Utilities for combining adapters
- **Training Utils** (`src/models/adapters/training_utils.py`): Parent and child trainer classes

## Installation & Setup

1. **Dependencies**: Ensure PyTorch and other requirements are installed:
   ```bash
   pip install torch torchvision torchaudio
   pip install pyyaml tensorboard
   ```

2. **Base Model**: Ensure you have a trained base model checkpoint at `checkpoints/base_model.pt`

3. **Tokenizer**: Ensure tokenizer vocabulary exists at `vocab.json`

4. **Style Packs**: Organize style pack data under `/style_packs/` following the expected structure

## Usage

### 1. Train Parent Adapter

Train a parent adapter for a broad genre (e.g., pop):

```bash
python train_parent_adapter.py \
    --parent pop \
    --pack /style_packs/pop \
    --epochs 15 \
    --batch-size 8 \
    --rank 16 \
    --alpha 32.0 \
    --output-dir ./checkpoints/adapters
```

**Parameters:**
- `--parent`: Parent style name (must match genre config)
- `--pack`: Path to style pack directory containing training data
- `--rank`: LoRA rank (higher = more capacity, default: 16)
- `--alpha`: LoRA scaling factor (higher = stronger adaptation, default: 32.0)
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Training batch size (default: 8)

### 2. Train Child Adapter

Train a child adapter that inherits from a parent:

```bash
python train_child_adapter.py \
    --parent pop \
    --child dance_pop \
    --pack /style_packs/pop/dance_pop \
    --parent_lora checkpoints/adapters/pop/pop.lora \
    --epochs 8 \
    --batch-size 4 \
    --rank 8 \
    --alpha 16.0
```

**Parameters:**
- `--parent`: Parent style name 
- `--child`: Child style name
- `--parent_lora`: Path to parent adapter checkpoint
- Child adapters typically use lower rank/alpha and fewer epochs

### 3. Merge Adapters for Inference

Combine adapters into inference-ready models:

```bash
# Single parent adapter
python merge_adapters.py \
    --base-model checkpoints/base_model.pt \
    --parent-adapter checkpoints/adapters/pop/pop.lora \
    --output merged_models/pop_model.pt \
    --create-config

# Hierarchical (parent + child)
python merge_adapters.py \
    --base-model checkpoints/base_model.pt \
    --parent-adapter checkpoints/adapters/pop/pop.lora \
    --child-adapter checkpoints/adapters/pop/dance_pop/dance_pop.lora \
    --output merged_models/dance_pop_model.pt \
    --blend-mode additive \
    --parent-weight 1.0 \
    --child-weight 1.0 \
    --create-config
```

**Blend Modes:**
- `additive`: Base + α×Parent + β×Child (default)
- `interpolative`: Base + (α×Parent + β×Child)/(α+β)

### 4. Test System

Run comprehensive tests to verify functionality:

```bash
python test_adapters.py
```

Tests include:
- LoRA layer functionality (forward/backward, merging)
- Style adapter creation and state management
- Hierarchical adapter composition
- **Decode parity verification** (child with no overrides = parent only)
- Merge utility compatibility checks

## Configuration

### LoRA Training Config (`configs/lora_adapter_config.yaml`)

```yaml
lora:
  parent:
    rank: 16              # Higher rank for parent adapters
    alpha: 32.0           # Strong adaptation signal
    target_modules: ["attention", "feed_forward"]
    
  child:
    rank: 8               # Lower rank for focused changes  
    alpha: 16.0           # Subtler adaptation
    target_modules: ["attention", "feed_forward"]

training:
  parent:
    num_epochs: 15        # More epochs for parent
    learning_rate: 1e-4
    
  child:
    num_epochs: 8         # Fewer epochs for child
    learning_rate: 5e-5   # Lower learning rate for fine-tuning
```

### Genre Configuration (`configs/genres/<parent>.yaml`)

Each parent genre can specify LoRA-specific settings:

```yaml
# Pop genre configuration
name: "pop"
# ... other genre settings ...

# LoRA training overrides
lora:
  rank: 16
  alpha: 32.0
  target_modules: ["attention", "feed_forward", "output_projection"]

training_overrides:
  num_epochs: 20          # Pop needs more epochs
  batch_size: 6           # Adjust for data size
```

### Child Style Configuration (`configs/styles/<parent>/<child>.yaml`)

Child styles inherit parent settings but can override:

```yaml
# Dance Pop configuration
parent: "pop"
name: "dance_pop"

# Child-specific LoRA settings
lora:
  rank: 6                 # Smaller than parent
  alpha: 12.0            # Lower adaptation strength
  target_modules: ["attention"]  # Fewer modules
```

## File Structure

```
checkpoints/adapters/
├── pop/
│   ├── pop.lora                    # Parent adapter
│   ├── pop_best.pt                 # Full training checkpoint
│   ├── dance_pop/
│   │   ├── dance_pop.lora          # Child adapter
│   │   └── dance_pop_best.pt       # Child training checkpoint
│   ├── synth_pop/
│   │   └── synth_pop.lora
│   └── logs/                       # Training logs
├── rock/
│   ├── rock.lora
│   ├── indie_rock/
│   └── ...
└── ...

merged_models/
├── pop_model.pt                    # Parent-only merged model
├── pop_model.json                  # Inference configuration
├── dance_pop_model.pt              # Hierarchical merged model
├── dance_pop_model.json
└── ...
```

## Advanced Features

### Decode Parity Testing

The system includes tests to verify that when a child adapter has no meaningful overrides, the output is identical to parent-only:

```python
# Test that child_weight=0 gives identical output to parent-only
assert output_parent_only ≈ output_parent_child_zero_weight
```

### Compatibility Verification

Before merging, the system verifies adapter compatibility:

```bash
python merge_adapters.py --test-compatibility \
    --base-model checkpoints/base_model.pt \
    --parent-adapter checkpoints/pop.lora \
    --child-adapter checkpoints/dance_pop.lora
```

### Custom Weight Combinations

Fine-tune the balance between parent and child influences:

```bash
# Emphasize parent characteristics
python merge_adapters.py \
    --parent-weight 1.0 --child-weight 0.3 \
    --output dance_pop_parent_heavy.pt

# Emphasize child characteristics  
python merge_adapters.py \
    --parent-weight 0.7 --child-weight 1.2 \
    --output dance_pop_child_heavy.pt
```

## Design Principles

1. **Parameter Efficiency**: LoRA adapters add minimal parameters while achieving effective adaptation

2. **Hierarchical Learning**: Parent adapters capture broad characteristics, children learn specific variations

3. **Composability**: Adapters can be mixed and matched with different weights and blending strategies

4. **Reproducibility**: All training runs are logged and checkpointed for reproducibility

5. **Testing**: Comprehensive test suite ensures decode parity and system reliability

## Troubleshooting

### Common Issues

1. **"Tokenizer vocabulary not found"**: Ensure `vocab.json` exists in the current directory
2. **"Base model not found"**: Verify `checkpoints/base_model.pt` exists and is a valid checkpoint
3. **Style pack validation fails**: Check that style packs contain `refs_midi/` and `refs_audio/` directories
4. **Decode parity test fails**: This may indicate child adapter has too much influence - reduce child rank/alpha

### Performance Tips

1. **Start with smaller ranks** (4-8) and increase if needed
2. **Use lower learning rates for child training** (5e-5 vs 1e-4 for parent)
3. **Child adapters need fewer epochs** (5-8 vs 10-15 for parent)
4. **Monitor training loss** - child loss should converge faster than parent

## Examples

See `examples_lora_training.sh` for comprehensive usage examples including:
- Training complete genre hierarchies
- Batch processing multiple styles
- Advanced merging strategies
- Testing and validation workflows

## Contributing

When adding new functionality:
1. Add comprehensive tests to `test_adapters.py`
2. Update configuration examples
3. Verify decode parity for new adapter types
4. Document any new parameters or options