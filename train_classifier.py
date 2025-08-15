#!/usr/bin/env python3
"""
CLI for training the adherence classifier.
Usage: python train_classifier.py --train_data data.jsonl --val_data val.jsonl --vocab_size 1000
"""

import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from critic.classifier import AdherenceClassifier, AdherenceDataset, train_classifier

def main():
    parser = argparse.ArgumentParser(description="Train Adherence Classifier")
    parser.add_argument("--train_data", required=True, help="Training data JSON lines file")
    parser.add_argument("--val_data", required=True, help="Validation data JSON lines file")
    parser.add_argument("--vocab_size", type=int, required=True, help="Tokenizer vocabulary size")
    parser.add_argument("--output_dir", default="./models/classifier", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    logger.info(f"Loading training data from {args.train_data}")
    train_dataset = AdherenceDataset(args.train_data)
    
    logger.info(f"Loading validation data from {args.val_data}")
    val_dataset = AdherenceDataset(args.val_data)
    
    # Create model
    logger.info(f"Creating classifier with vocab_size={args.vocab_size}")
    model = AdherenceClassifier(args.vocab_size)
    
    # Train model
    logger.info("Starting training...")
    train_classifier(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # Save model
    model_path = output_dir / "adherence_classifier.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save config
    config = {
        'vocab_size': args.vocab_size,
        'model_class': 'AdherenceClassifier',
        'training_args': vars(args)
    }
    
    import json
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Config saved to {config_path}")
    logger.info("Training completed!")

if __name__ == "__main__":
    main()