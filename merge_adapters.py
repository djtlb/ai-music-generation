#!/usr/bin/env python3
"""
Adapter merge utility for combining base models with hierarchical LoRA adapters.

Provides command-line interface for merging adapters with various strategies:
- Single adapter merge (base + parent OR base + child)
- Hierarchical merge (base + parent + child)
- Inference-optimized merging

Usage:
    # Merge single parent adapter
    python merge_adapters.py --base-model checkpoints/base_model.pt \
        --parent-adapter checkpoints/pop.lora --output merged_pop_model.pt

    # Merge hierarchical (parent + child)
    python merge_adapters.py --base-model checkpoints/base_model.pt \
        --parent-adapter checkpoints/pop.lora \
        --child-adapter checkpoints/dance_pop.lora \
        --output merged_dance_pop_model.pt
"""

import os
import sys
import argparse
import torch
import yaml
from pathlib import Path
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.adapters.adapter_merge import HierarchicalMerger, verify_adapter_compatibility
from models.adapters.style_adapter import load_style_adapter
from models.mh_transformer import MelodyHarmonyTransformer
from models.tokenizer import MIDITokenizer


def load_base_model(model_path: str) -> torch.nn.Module:
    """Load base model from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Base model not found: {model_path}")
    
    # Load tokenizer for vocab size
    tokenizer = MIDITokenizer()
    if os.path.exists('vocab.json'):
        tokenizer.load_vocab('vocab.json')
    else:
        raise FileNotFoundError("Tokenizer vocabulary not found")
    
    # Load model config from checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'config' in checkpoint:
        model_config = checkpoint['config']['model']
    else:
        # Fallback to default config
        model_config = {
            'hidden_size': 512,
            'num_layers': 8,
            'num_heads': 8,
            'max_seq_len': 512,
            'dropout': 0.1,
            'vocab_size': len(tokenizer.vocab)
        }
    
    # Ensure vocab size is set
    model_config['vocab_size'] = len(tokenizer.vocab)
    
    # Create and load model
    model = MelodyHarmonyTransformer(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded base model from {model_path}")
    return model


def merge_single_adapter(
    base_model: torch.nn.Module,
    adapter_path: str,
    output_path: str,
    weight: float = 1.0
) -> None:
    """Merge a single adapter into base model."""
    print(f"Merging single adapter: {adapter_path}")
    
    # Load adapter
    adapter = load_style_adapter(base_model, adapter_path)
    
    # Apply LoRA weights directly to base model
    for name, lora_layer in adapter.lora_layers.items():
        delta_w = weight * lora_layer.scaling * (lora_layer.lora_B @ lora_layer.lora_A)
        lora_layer.base_layer.weight.data += delta_w
    
    # Save merged model
    torch.save({
        'model_state_dict': base_model.state_dict(),
        'adapter_info': {
            'style': adapter.style_name,
            'weight': weight,
            'type': 'single_adapter'
        }
    }, output_path)
    
    print(f"Saved merged model to {output_path}")


def merge_hierarchical_adapters(
    base_model: torch.nn.Module,
    parent_adapter_path: str,
    child_adapter_path: str,
    output_path: str,
    parent_weight: float = 1.0,
    child_weight: float = 1.0,
    blend_mode: str = 'additive'
) -> None:
    """Merge parent and child adapters hierarchically."""
    print(f"Merging hierarchical adapters:")
    print(f"  Parent: {parent_adapter_path} (weight: {parent_weight})")
    print(f"  Child: {child_adapter_path} (weight: {child_weight})")
    print(f"  Blend mode: {blend_mode}")
    
    # Create hierarchical merger
    merger = HierarchicalMerger(base_model)
    
    # Load adapters
    merger.load_parent_adapter(parent_adapter_path, parent_weight)
    merger.load_child_adapter(child_adapter_path, child_weight=child_weight)
    
    # Merge with specified strategy
    merger.merge_hierarchical(
        parent_weight=parent_weight,
        child_weight=child_weight,
        blend_mode=blend_mode
    )
    
    # Save merged checkpoint
    merger.create_merged_checkpoint(output_path)
    
    print(f"Saved hierarchical merged model to {output_path}")


def test_decode_parity(
    base_model: torch.nn.Module,
    parent_adapter_path: str,
    child_adapter_path: str,
    num_test_samples: int = 10
) -> bool:
    """
    Test decode parity when child adapter has no meaningful overrides.
    
    Verifies that base + parent + (empty child) ≈ base + parent
    """
    print("Testing decode parity...")
    
    from models.adapters.adapter_merge import HierarchicalMerger
    import torch.nn.functional as F
    
    # Create test input
    batch_size = 2
    seq_len = 64
    vocab_size = 1000  # Placeholder
    
    test_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Test 1: Base + Parent only
    merger1 = HierarchicalMerger(base_model)
    merger1.load_parent_adapter(parent_adapter_path)
    merger1.merge_hierarchical(parent_weight=1.0, child_weight=0.0)
    
    with torch.no_grad():
        base_model.eval()
        output1 = base_model(test_input)
    
    merger1.unmerge()
    
    # Test 2: Base + Parent + Child (with minimal child weight)
    merger2 = HierarchicalMerger(base_model)
    merger2.load_parent_adapter(parent_adapter_path)
    merger2.load_child_adapter(child_adapter_path)
    merger2.merge_hierarchical(parent_weight=1.0, child_weight=0.01)
    
    with torch.no_grad():
        base_model.eval()
        output2 = base_model(test_input)
    
    merger2.unmerge()
    
    # Compare outputs
    if hasattr(output1, 'logits'):
        diff = F.mse_loss(output1.logits, output2.logits)
    else:
        diff = F.mse_loss(output1, output2)
    
    parity_threshold = 1e-3
    passed = diff.item() < parity_threshold
    
    print(f"Decode parity test: {'PASSED' if passed else 'FAILED'}")
    print(f"MSE difference: {diff.item():.6f} (threshold: {parity_threshold})")
    
    return passed


def create_inference_config(
    output_path: str,
    parent_style: str,
    child_style: str = None,
    parent_weight: float = 1.0,
    child_weight: float = 1.0,
    blend_mode: str = 'additive'
) -> None:
    """Create configuration file for inference usage."""
    config = {
        'model_type': 'hierarchical_adapter' if child_style else 'single_adapter',
        'parent_style': parent_style,
        'child_style': child_style,
        'weights': {
            'parent': parent_weight,
            'child': child_weight if child_style else None
        },
        'blend_mode': blend_mode,
        'usage': {
            'description': f"Merged model for {child_style or parent_style} style",
            'inference_ready': True,
            'requires_adapters': False
        }
    }
    
    config_path = Path(output_path).with_suffix('.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created inference config: {config_path}")


def main():
    parser = argparse.ArgumentParser(description='Merge LoRA adapters with base model')
    parser.add_argument('--base-model', type=str, required=True,
                        help='Path to base model checkpoint')
    parser.add_argument('--parent-adapter', type=str, required=True,
                        help='Path to parent LoRA adapter (.lora file)')
    parser.add_argument('--child-adapter', type=str, default=None,
                        help='Path to child LoRA adapter (.lora file)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for merged model')
    parser.add_argument('--parent-weight', type=float, default=1.0,
                        help='Scaling weight for parent adapter')
    parser.add_argument('--child-weight', type=float, default=1.0,
                        help='Scaling weight for child adapter')
    parser.add_argument('--blend-mode', type=str, default='additive',
                        choices=['additive', 'interpolative'],
                        help='Blending strategy for hierarchical merge')
    parser.add_argument('--test-compatibility', action='store_true',
                        help='Test adapter compatibility before merging')
    parser.add_argument('--test-decode-parity', action='store_true',
                        help='Test decode parity for hierarchical adapters')
    parser.add_argument('--create-config', action='store_true',
                        help='Create inference configuration file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for operations (cpu, cuda)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.base_model):
        print(f"Error: Base model not found: {args.base_model}")
        return 1
    
    if not os.path.exists(args.parent_adapter):
        print(f"Error: Parent adapter not found: {args.parent_adapter}")
        return 1
    
    if args.child_adapter and not os.path.exists(args.child_adapter):
        print(f"Error: Child adapter not found: {args.child_adapter}")
        return 1
    
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    try:
        # Load base model
        base_model = load_base_model(args.base_model)
        base_model = base_model.to(device)
        
        # Test compatibility if requested
        if args.test_compatibility:
            adapter_paths = [args.parent_adapter]
            if args.child_adapter:
                adapter_paths.append(args.child_adapter)
            
            if not verify_adapter_compatibility(base_model, adapter_paths):
                print("Error: Adapter compatibility test failed")
                return 1
        
        # Test decode parity if requested
        if args.test_decode_parity and args.child_adapter:
            if not test_decode_parity(base_model, args.parent_adapter, args.child_adapter):
                print("Warning: Decode parity test failed")
        
        # Perform merging
        if args.child_adapter:
            # Hierarchical merge
            merge_hierarchical_adapters(
                base_model=base_model,
                parent_adapter_path=args.parent_adapter,
                child_adapter_path=args.child_adapter,
                output_path=args.output,
                parent_weight=args.parent_weight,
                child_weight=args.child_weight,
                blend_mode=args.blend_mode
            )
        else:
            # Single adapter merge
            merge_single_adapter(
                base_model=base_model,
                adapter_path=args.parent_adapter,
                output_path=args.output,
                weight=args.parent_weight
            )
        
        # Create inference config if requested
        if args.create_config:
            # Extract style names from adapter paths
            parent_style = Path(args.parent_adapter).stem.replace('.lora', '')
            child_style = None
            if args.child_adapter:
                child_style = Path(args.child_adapter).stem.replace('.lora', '')
            
            create_inference_config(
                output_path=args.output,
                parent_style=parent_style,
                child_style=child_style,
                parent_weight=args.parent_weight,
                child_weight=args.child_weight,
                blend_mode=args.blend_mode
            )
        
        print("Adapter merging completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error during merging: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())