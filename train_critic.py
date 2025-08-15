#!/usr/bin/env python3
"""
CLI for training the comprehensive critic model.
Usage: python train_critic.py --train_data data.jsonl --val_data val.jsonl --vocab_size 1000
"""

import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import json

from critic.model import ComprehensiveCritic

class CriticDataset(Dataset):
    """Dataset for training the comprehensive critic"""
    
    def __init__(self, data_path: str):
        self.samples = []
        self._load_data(data_path)
    
    def _load_data(self, data_path: str):
        """Load training data with all critic inputs"""
        with open(data_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'prompt': sample['prompt'],
            'control_json': sample['control_json'],
            'tokens': torch.tensor(sample['tokens'], dtype=torch.long),
            'mel_spec': torch.tensor(sample['mel_spec'], dtype=torch.float),
            'ref_embedding': torch.tensor(sample['ref_embedding'], dtype=torch.float),
            'mix_features': torch.tensor(sample['mix_features'], dtype=torch.float),
            'overall_score': torch.tensor(sample['overall_score'], dtype=torch.float),
            'adherence_score': torch.tensor(sample['adherence_score'], dtype=torch.float),
            'style_score': torch.tensor(sample['style_score'], dtype=torch.float),
            'mix_score': torch.tensor(sample['mix_score'], dtype=torch.float)
        }

def train_critic(
    model: ComprehensiveCritic,
    train_dataset: CriticDataset,
    val_dataset: CriticDataset,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = 'cuda'
):
    """Train the comprehensive critic model"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    logger = logging.getLogger(__name__)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            prompts = batch['prompt']
            controls = batch['control_json']
            tokens = batch['tokens'].to(device)
            mel_specs = batch['mel_spec'].to(device)
            ref_embeddings = batch['ref_embedding'].to(device)
            mix_features = batch['mix_features'].to(device)
            
            targets = batch['overall_score'].to(device)
            
            overall_pred, component_pred = model(
                prompts, controls, tokens, mel_specs, ref_embeddings, mix_features
            )
            
            # Main loss
            loss = criterion(overall_pred.squeeze(), targets)
            
            # Component losses
            if 'adherence_score' in batch:
                adherence_targets = batch['adherence_score'].to(device)
                loss += 0.3 * criterion(component_pred['adherence'].squeeze(), adherence_targets)
            
            if 'style_score' in batch:
                style_targets = batch['style_score'].to(device)
                loss += 0.3 * criterion(component_pred['style_match'].squeeze(), style_targets)
            
            if 'mix_score' in batch:
                mix_targets = batch['mix_score'].to(device)
                loss += 0.3 * criterion(component_pred['mix_quality'].squeeze(), mix_targets)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                prompts = batch['prompt']
                controls = batch['control_json']
                tokens = batch['tokens'].to(device)
                mel_specs = batch['mel_spec'].to(device)
                ref_embeddings = batch['ref_embedding'].to(device)
                mix_features = batch['mix_features'].to(device)
                targets = batch['overall_score'].to(device)
                
                overall_pred, _ = model(
                    prompts, controls, tokens, mel_specs, ref_embeddings, mix_features
                )
                loss = criterion(overall_pred.squeeze(), targets)
                val_loss += loss.item()
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}")
        logger.info(f"Val Loss: {val_loss/len(val_loader):.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train Comprehensive Critic")
    parser.add_argument("--train_data", required=True, help="Training data JSON lines file")
    parser.add_argument("--val_data", required=True, help="Validation data JSON lines file")
    parser.add_argument("--vocab_size", type=int, required=True, help="Tokenizer vocabulary size")
    parser.add_argument("--output_dir", default="./models/critic", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--adherence_weight", type=float, default=0.4, help="Adherence component weight")
    parser.add_argument("--style_weight", type=float, default=0.3, help="Style component weight")
    parser.add_argument("--mix_weight", type=float, default=0.3, help="Mix component weight")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    logger.info(f"Loading training data from {args.train_data}")
    train_dataset = CriticDataset(args.train_data)
    
    logger.info(f"Loading validation data from {args.val_data}")
    val_dataset = CriticDataset(args.val_data)
    
    # Create model
    logger.info(f"Creating critic with vocab_size={args.vocab_size}")
    model = ComprehensiveCritic(
        vocab_size=args.vocab_size,
        adherence_weight=args.adherence_weight,
        style_weight=args.style_weight,
        mix_weight=args.mix_weight
    )
    
    # Train model
    logger.info("Starting training...")
    train_critic(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # Save model
    model_path = output_dir / "comprehensive_critic.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save config
    config = {
        'vocab_size': args.vocab_size,
        'model_class': 'ComprehensiveCritic',
        'adherence_weight': args.adherence_weight,
        'style_weight': args.style_weight,
        'mix_weight': args.mix_weight,
        'training_args': vars(args)
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Config saved to {config_path}")
    logger.info("Training completed!")

if __name__ == "__main__":
    main()