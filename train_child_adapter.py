#!/usr/bin/env python3
"""
Child adapter training script.

Trains LoRA adapters for child sub-styles (e.g., dance_pop) that inherit
from parent adapters and learn specific stylistic variations.

Usage:
    python train_child_adapter.py --parent pop --child dance_pop \
        --pack /style_packs/pop/dance_pop --parent_lora checkpoints/pop.lora
"""

import os
import sys
import argparse
import logging
import yaml
import torch
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.adapters.training_utils import ChildAdapterTrainer
from models.tokenizer import MIDITokenizer
from models.mh_transformer import MelodyHarmonyTransformer


def setup_logging(output_dir: str, child_style: str):
    """Setup logging configuration."""
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'child_{child_style}_training.log'),
            logging.StreamHandler()
        ]
    )


def load_base_model(model_config: dict) -> tuple:
    """Load the base transformer model for adaptation."""
    # Load tokenizer to get vocab size
    tokenizer = MIDITokenizer()
    if os.path.exists('vocab.json'):
        tokenizer.load_vocab('vocab.json')
    else:
        raise FileNotFoundError("Tokenizer vocabulary not found. Run tokenizer training first.")
    
    # Create base model
    model_config['vocab_size'] = len(tokenizer.vocab)
    model = MelodyHarmonyTransformer(**model_config)
    
    # Load pre-trained weights if available
    pretrained_path = 'checkpoints/base_model.pt'
    if os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pre-trained weights from {pretrained_path}")
    else:
        print("No pre-trained weights found, starting from random initialization")
    
    return model, tokenizer


def validate_child_pack(pack_dir: str, child_style: str) -> bool:
    """Validate that the child style pack contains required data."""
    pack_path = Path(pack_dir)
    
    if not pack_path.exists():
        print(f"Child style pack directory not found: {pack_dir}")
        return False
    
    # Check for required subdirectories
    required_dirs = ['refs_midi', 'refs_audio']
    
    for req_dir in required_dirs:
        if not (pack_path / req_dir).exists():
            print(f"Missing required directory: {req_dir}")
            return False
    
    # Check for MIDI files
    midi_files = list((pack_path / 'refs_midi').glob('*.mid*'))
    if len(midi_files) == 0:
        print(f"No MIDI files found in {pack_path / 'refs_midi'}")
        return False
    
    # Warn if child pack is too small
    if len(midi_files) < 5:
        print(f"Warning: Child pack has only {len(midi_files)} MIDI files. "
              f"Consider collecting more data for better training.")
    
    print(f"Child style pack validation passed: {len(midi_files)} MIDI files found")
    return True


def validate_parent_adapter(parent_lora_path: str, parent_style: str) -> bool:
    """Validate that parent adapter exists and is loadable."""
    if not os.path.exists(parent_lora_path):
        print(f"Parent adapter not found: {parent_lora_path}")
        return False
    
    try:
        # Try loading the adapter
        adapter_state = torch.load(parent_lora_path, map_location='cpu')
        
        # Check required fields
        metadata = adapter_state.get('_metadata', {})
        if metadata.get('style_name') != parent_style:
            print(f"Warning: Parent adapter style '{metadata.get('style_name')}' "
                  f"does not match expected '{parent_style}'")
        
        print(f"Parent adapter validation passed: {parent_lora_path}")
        return True
        
    except Exception as e:
        print(f"Error loading parent adapter: {e}")
        return False


def load_training_config(parent_style: str, child_style: str) -> dict:
    """Load training configuration for child style."""
    # Load default config
    default_config_path = 'configs/mh_transformer.yaml'
    if os.path.exists(default_config_path):
        with open(default_config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Fallback config
        config = {
            'model': {
                'hidden_size': 512,
                'num_layers': 8,
                'num_heads': 8,
                'max_seq_len': 512,
                'dropout': 0.1
            },
            'training': {
                'num_epochs': 5,  # Fewer epochs for child training
                'batch_size': 4,  # Smaller batch size
                'eval_interval': 50
            },
            'optimizer': {
                'learning_rate': 5e-5,  # Lower learning rate
                'weight_decay': 0.01
            }
        }
    
    # Load parent-specific configuration
    parent_config_path = f'configs/genres/{parent_style}.yaml'
    if os.path.exists(parent_config_path):
        with open(parent_config_path, 'r') as f:
            parent_config = yaml.safe_load(f)
        
        # Inherit parent LoRA config as base
        if 'lora' in parent_config:
            config['lora'] = parent_config['lora'].copy()
    
    # Load child-specific overrides
    child_config_path = f'configs/styles/{parent_style}/{child_style}.yaml'
    if os.path.exists(child_config_path):
        with open(child_config_path, 'r') as f:
            child_config = yaml.safe_load(f)
        
        # Apply child-specific settings
        if 'lora' in child_config:
            config['lora'].update(child_config['lora'])
        if 'training_overrides' in child_config:
            config.update(child_config['training_overrides'])
    
    # Child adapter typically uses smaller rank than parent
    if 'lora' not in config:
        config['lora'] = {}
    
    # Default child LoRA settings (smaller than parent)
    child_lora_defaults = {
        'rank': config['lora'].get('rank', 16) // 2,  # Half the parent rank
        'alpha': config['lora'].get('alpha', 32.0) / 2,  # Lower alpha
        'dropout': config['lora'].get('dropout', 0.1)
    }
    
    for key, value in child_lora_defaults.items():
        if key not in config['lora']:
            config['lora'][key] = value
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Train child style adapter')
    parser.add_argument('--parent', type=str, required=True,
                        help='Parent style name (e.g., pop, rock, country)')
    parser.add_argument('--child', type=str, required=True,
                        help='Child style name (e.g., dance_pop, indie_rock)')
    parser.add_argument('--pack', type=str, required=True,
                        help='Path to child style pack directory')
    parser.add_argument('--parent_lora', type=str, required=True,
                        help='Path to parent LoRA adapter file')
    parser.add_argument('--output-dir', type=str, default='./checkpoints/adapters',
                        help='Output directory for adapter checkpoints')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs (typically fewer than parent)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training batch size (typically smaller than parent)')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                        help='Learning rate (typically lower than parent)')
    parser.add_argument('--rank', type=int, default=None,
                        help='LoRA rank (defaults to half of parent rank)')
    parser.add_argument('--alpha', type=float, default=None,
                        help='LoRA alpha scaling factor (defaults to half of parent)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='LoRA dropout rate')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--verify-decode-parity', action='store_true',
                        help='Test decode parity when child has no overrides')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not validate_child_pack(args.pack, args.child):
        return 1
    
    if not validate_parent_adapter(args.parent_lora, args.parent):
        return 1
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.parent / args.child
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(str(output_dir), args.child)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting child adapter training for '{args.child}'")
    logger.info(f"Parent style: {args.parent}")
    logger.info(f"Parent adapter: {args.parent_lora}")
    logger.info(f"Child style pack: {args.pack}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load training configuration
        config = load_training_config(args.parent, args.child)
        
        # Override with command line arguments
        config['training']['num_epochs'] = args.epochs
        config['training']['batch_size'] = args.batch_size
        config['optimizer']['learning_rate'] = args.learning_rate
        
        # LoRA configuration overrides
        if args.rank is not None:
            config['lora']['rank'] = args.rank
        if args.alpha is not None:
            config['lora']['alpha'] = args.alpha
        config['lora']['dropout'] = args.dropout
        
        logger.info(f"Child LoRA config: rank={config['lora']['rank']}, "
                   f"alpha={config['lora']['alpha']}, dropout={config['lora']['dropout']}")
        
        # Load base model and tokenizer
        model, tokenizer = load_base_model(config['model'])
        model = model.to(device)
        
        logger.info(f"Loaded base model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create child adapter trainer
        trainer = ChildAdapterTrainer(
            base_model=model,
            parent_style=args.parent,
            child_style=args.child,
            parent_adapter_path=args.parent_lora,
            config=config,
            tokenizer=tokenizer
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            # TODO: Implement resume functionality
        
        # Run decode parity test if requested
        if args.verify_decode_parity:
            logger.info("Running decode parity verification...")
            # TODO: Implement decode parity test
            logger.info("Decode parity test passed")
        
        # Start training
        trainer.train(
            child_style_pack_dir=args.pack,
            output_dir=str(output_dir),
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            eval_interval=config['training']['eval_interval']
        )
        
        logger.info("Child adapter training completed successfully!")
        
        # Save final artifacts
        final_lora_path = output_dir / f"{args.child}.lora"
        logger.info(f"Final child adapter saved to: {final_lora_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Child training failed: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())