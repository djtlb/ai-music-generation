#!/usr/bin/env python3
"""
Training script for Arrangement Transformer

Usage:
    python scripts/train_arrangement.py --config configs/arrangement/default.yaml
    python scripts/train_arrangement.py --config configs/arrangement/fast.yaml --gpus 1
"""

import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import sys
import os
from pathlib import Path

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.arrangement_transformer import ArrangementTransformer, load_config
from models.arrangement_dataset import ArrangementDataModule


def main():
    parser = argparse.ArgumentParser(description="Train Arrangement Transformer")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to configuration YAML file")
    parser.add_argument("--gpus", type=int, default=None,
                       help="Number of GPUs to use (overrides config)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode (fast dev run)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # Override GPU setting if provided
    if args.gpus is not None:
        config['hardware']['devices'] = args.gpus
    
    # Set up logging
    logger = None
    if 'wandb_project' in config['logging']:
        logger = WandbLogger(
            project=config['logging']['wandb_project'],
            save_dir=config['logging']['checkpoint_dir']
        )
    
    # Initialize model
    model = ArrangementTransformer(config)
    
    # Initialize data module
    data_module = ArrangementDataModule(config, model.tokenizer)
    
    # Set up callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['logging']['checkpoint_dir'],
        filename='arrangement-{epoch:02d}-{val_loss:.3f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor=config['training']['early_stopping_monitor'],
        patience=config['training']['early_stopping_patience'],
        mode='min',
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['hardware']['accelerator'],
        devices=config['hardware']['devices'],
        precision=config['hardware']['precision'],
        strategy=config['hardware']['strategy'],
        
        callbacks=callbacks,
        logger=logger,
        
        # Gradient clipping
        gradient_clip_val=config['training']['gradient_clip_val'],
        
        # Validation settings
        val_check_interval=config['logging']['val_check_interval'],
        
        # Logging
        log_every_n_steps=config['logging']['log_every_n_steps'],
        
        # Debug mode
        fast_dev_run=args.debug,
        
        # Resume from checkpoint
        resume_from_checkpoint=args.resume
    )
    
    # Print model info
    print("\nModel Summary:")
    print(f"Vocabulary size: {model.vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Print data info
    data_module.setup()
    print(f"\nDataset info:")
    print(f"Training samples: {len(data_module.train_dataset)}")
    print(f"Validation samples: {len(data_module.val_dataset)}")
    print(f"Test samples: {len(data_module.test_dataset)}")
    
    # Start training
    print("\nStarting training...")
    trainer.fit(model, data_module)
    
    # Test the model
    if not args.debug:
        print("\nRunning final test...")
        trainer.test(model, data_module)
    
    print("Training completed!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()