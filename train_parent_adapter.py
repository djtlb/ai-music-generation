#!/usr/bin/env python3
"""
Parent adapter training script.

Trains LoRA adapters for parent genres (e.g., pop, rock, country) using
data from style packs containing multiple sub-styles.

Usage:
    python train_parent_adapter.py --parent pop --pack /style_packs/pop
"""

import os
import sys
import argparse
import logging
import yaml
import torch
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from models.adapters.training_utils import ParentAdapterTrainer
from models.tokenizer import MIDITokenizer
from models.mh_transformer import MelodyHarmonyTransformer


def setup_logging(output_dir: str, parent_style: str):
    """Setup logging configuration."""
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'parent_{parent_style}_training.log'),
            logging.StreamHandler()
        ]
    )


def load_base_model(model_config: dict) -> tuple[torch.nn.Module, MIDITokenizer]:
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


def validate_style_pack(pack_dir: str, parent_style: str) -> bool:
    """Validate that the style pack contains required data."""
    pack_path = Path(pack_dir)
    
    if not pack_path.exists():
        print(f"Style pack directory not found: {pack_dir}")
        return False
    
    # Check for required subdirectories
    required_dirs = ['refs_midi', 'refs_audio']
    
    for req_dir in required_dirs:
        if not (pack_path / req_dir).exists():
            print(f"Missing required directory: {req_dir}")
            return False
    
    # Check for metadata
    meta_path = pack_path / 'meta.json'
    if not meta_path.exists():
        print(f"Warning: No meta.json found in {pack_dir}")
    
    # Check for MIDI files
    midi_files = list((pack_path / 'refs_midi').glob('*.mid*'))
    if len(midi_files) == 0:
        print(f"No MIDI files found in {pack_path / 'refs_midi'}")
        return False
    
    print(f"Style pack validation passed: {len(midi_files)} MIDI files found")
    return True


def setup_logging(output_dir: str, parent_style: str):
    """Setup logging configuration."""
    log_dir = Path(output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f'parent_{parent_style}_training.log'),
            logging.StreamHandler()
        ]
    )


def load_base_model(model_config: dict) -> torch.nn.Module:
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


def validate_style_pack(pack_dir: str, parent_style: str) -> bool:
    """Validate that the style pack contains required data."""
    pack_path = Path(pack_dir)
    
    if not pack_path.exists():
        print(f"Style pack directory not found: {pack_dir}")
        return False
    
    # Check for required subdirectories
    required_dirs = ['refs_midi', 'refs_audio']
    optional_dirs = ['processed']
    
    for req_dir in required_dirs:
        if not (pack_path / req_dir).exists():
            print(f"Missing required directory: {req_dir}")
            return False
    
    # Check for metadata
    meta_path = pack_path / 'meta.json'
    if not meta_path.exists():
        print(f"Warning: No meta.json found in {pack_dir}")
    
    # Check for MIDI files
    midi_files = list((pack_path / 'refs_midi').glob('*.mid*'))
    if len(midi_files) == 0:
        print(f"No MIDI files found in {pack_path / 'refs_midi'}")
        return False
    
    print(f"Style pack validation passed: {len(midi_files)} MIDI files found")
    return True


def load_training_config(parent_style: str) -> dict:
    """Load training configuration for parent style."""
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
                'num_epochs': 10,
                'batch_size': 8,
                'eval_interval': 100
            },
            'optimizer': {
                'learning_rate': 1e-4,
                'weight_decay': 0.01
            }
        }
    
    # Load parent-specific overrides
    parent_config_path = f'configs/genres/{parent_style}.yaml'
    if os.path.exists(parent_config_path):
        with open(parent_config_path, 'r') as f:
            parent_config = yaml.safe_load(f)
        
        # Merge training-specific settings
        if 'lora' in parent_config:
            config['lora'] = parent_config['lora']
        if 'training_overrides' in parent_config:
            config.update(parent_config['training_overrides'])
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Train parent style adapter')
    parser.add_argument('--parent', type=str, required=True, 
                        help='Parent style name (e.g., pop, rock, country)')
    parser.add_argument('--pack', type=str, required=True,
                        help='Path to style pack directory')
    parser.add_argument('--output-dir', type=str, default='./checkpoints/adapters',
                        help='Output directory for adapter checkpoints')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--rank', type=int, default=16,
                        help='LoRA rank (higher = more capacity)')
    parser.add_argument('--alpha', type=float, default=32.0,
                        help='LoRA alpha scaling factor')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='LoRA dropout rate')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not validate_style_pack(args.pack, args.parent):
        return 1
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(str(output_dir), args.parent)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting parent adapter training for '{args.parent}'")
    logger.info(f"Style pack: {args.pack}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load training configuration
        config = load_training_config(args.parent)
        
        # Override with command line arguments
        config['training']['num_epochs'] = args.epochs
        config['training']['batch_size'] = args.batch_size
        config['optimizer']['learning_rate'] = args.learning_rate
        
        # LoRA configuration
        config['lora'] = config.get('lora', {})
        config['lora'].update({
            'rank': args.rank,
            'alpha': args.alpha,
            'dropout': args.dropout
        })
        
        # Load base model and tokenizer
        model, tokenizer = load_base_model(config['model'])
        model = model.to(device)
        
        logger.info(f"Loaded base model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create parent adapter trainer
        trainer = ParentAdapterTrainer(
            base_model=model,
            parent_style=args.parent,
            config=config,
            tokenizer=tokenizer
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            # TODO: Implement resume functionality
        
        # Start training
        trainer.train(
            style_pack_dir=args.pack,
            output_dir=str(output_dir),
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            eval_interval=config['training']['eval_interval']
        )
        
        logger.info("Parent adapter training completed successfully!")
        
        # Save final artifacts
        final_lora_path = output_dir / f"{args.parent}.lora"
        logger.info(f"Final adapter saved to: {final_lora_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())