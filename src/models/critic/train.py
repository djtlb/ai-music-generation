"""
Training script for the critic reward model.

Implements training loop with:
- Multi-task loss on quality dimensions
- Validation monitoring
- Model checkpointing
- Logging and metrics tracking
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging
from tqdm import tqdm

from .model import CriticModel, CriticLoss, create_critic_model
from .dataset import create_preference_dataloader, create_mock_preference_data


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


class CriticTrainer:
    """Trainer class for critic reward model."""
    
    def __init__(
        self,
        model: CriticModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        log_dir: str = './logs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Setup optimizer and loss
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = CriticLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=5,
            verbose=True
        )
        
        # Logging
        self.logger = setup_logging(log_dir)
        self.writer = SummaryWriter(log_dir)
        self.log_dir = Path(log_dir)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'quality_loss': 0.0,
            'overall_loss': 0.0,
            'consistency_loss': 0.0
        }
        
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch in progress_bar:
            # Move data to device
            mel_spectrograms = batch['mel_spectrogram'].to(self.device)
            style_ids = batch['style_id'].squeeze(-1).to(self.device)
            aux_features = batch['aux_features'].to(self.device)
            
            # Target scores
            target_scores = {}
            for key, values in batch['quality_scores'].items():
                target_scores[key] = values.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            predicted_scores = self.model(mel_spectrograms, style_ids, aux_features)
            
            # Compute loss
            total_loss, loss_breakdown = self.criterion(predicted_scores, target_scores)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in loss_breakdown.items():
                if key in epoch_losses:
                    epoch_losses[key] += value.item()
            
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
    
    def validate(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate model performance."""
        self.model.eval()
        
        val_losses = {
            'total_loss': 0.0,
            'quality_loss': 0.0,
            'overall_loss': 0.0,
            'consistency_loss': 0.0
        }
        
        # Track accuracy metrics
        quality_accuracies = {
            'hook_strength': [],
            'harmonic_stability': [], 
            'arrangement_contrast': [],
            'mix_quality': [],
            'style_match': [],
            'overall': []
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                mel_spectrograms = batch['mel_spectrogram'].to(self.device)
                style_ids = batch['style_id'].squeeze(-1).to(self.device)
                aux_features = batch['aux_features'].to(self.device)
                
                # Target scores
                target_scores = {}
                for key, values in batch['quality_scores'].items():
                    target_scores[key] = values.to(self.device)
                
                # Forward pass
                predicted_scores = self.model(mel_spectrograms, style_ids, aux_features)
                
                # Compute loss
                total_loss, loss_breakdown = self.criterion(predicted_scores, target_scores)
                
                # Accumulate losses
                for key, value in loss_breakdown.items():
                    if key in val_losses:
                        val_losses[key] += value.item()
                
                # Compute accuracy (within threshold)
                threshold = 0.1
                for quality_name in quality_accuracies:
                    if quality_name in predicted_scores and quality_name in target_scores:
                        pred = predicted_scores[quality_name].cpu().numpy()
                        target = target_scores[quality_name].cpu().numpy()
                        
                        # Accuracy within threshold
                        accuracy = np.mean(np.abs(pred - target) < threshold)
                        quality_accuracies[quality_name].append(accuracy)
                
                num_batches += 1
        
        # Average losses and accuracies
        for key in val_losses:
            val_losses[key] /= num_batches
            
        avg_accuracies = {}
        for quality_name, accuracies in quality_accuracies.items():
            if accuracies:
                avg_accuracies[quality_name] = np.mean(accuracies)
            else:
                avg_accuracies[quality_name] = 0.0
        
        return val_losses, avg_accuracies
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }
        
        # Save latest checkpoint
        checkpoint_path = self.log_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.log_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint with val_loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, num_epochs: int, save_every: int = 5):
        """Main training loop."""
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            
            # Train epoch
            train_losses = self.train_epoch()
            
            # Validate
            val_losses, val_accuracies = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_losses['total_loss'])
            
            # Log metrics
            self.log_metrics(train_losses, val_losses, val_accuracies)
            
            # Save checkpoint
            is_best = val_losses['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total_loss']
            
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
            
            # Store training history
            self.training_history.append({
                'epoch': self.epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies
            })
        
        self.logger.info("Training completed!")
        self.writer.close()
    
    def log_metrics(
        self, 
        train_losses: Dict[str, float],
        val_losses: Dict[str, float],
        val_accuracies: Dict[str, float]
    ):
        """Log training metrics."""
        # Console logging
        self.logger.info(
            f"Epoch {self.epoch} - "
            f"Train Loss: {train_losses['total_loss']:.4f}, "
            f"Val Loss: {val_losses['total_loss']:.4f}, "
            f"Val Acc: {val_accuracies['overall']:.3f}"
        )
        
        # TensorBoard logging
        for key, value in train_losses.items():
            self.writer.add_scalar(f'Train/{key}', value, self.epoch)
            
        for key, value in val_losses.items():
            self.writer.add_scalar(f'Validation/{key}', value, self.epoch)
            
        for key, value in val_accuracies.items():
            self.writer.add_scalar(f'Accuracy/{key}', value, self.epoch)
        
        # Learning rate
        self.writer.add_scalar(
            'Learning_Rate', 
            self.optimizer.param_groups[0]['lr'], 
            self.epoch
        )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train critic reward model')
    parser.add_argument('--data_csv', type=str, required=True,
                        help='Path to preference data CSV')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--log_dir', type=str, default='./logs/critic_training',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--mock_data', action='store_true',
                        help='Create mock data for testing')
    
    args = parser.parse_args()
    
    # Create mock data if requested
    if args.mock_data:
        print("Creating mock preference data...")
        create_mock_preference_data(args.data_csv, num_clips=200)
        print(f"Mock data saved to {args.data_csv}")
        return
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader = create_preference_dataloader(
        preference_csv=args.data_csv,
        audio_dir=args.audio_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_split=args.validation_split
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    model = create_critic_model(num_styles=3, device=device)
    
    # Create trainer
    trainer = CriticTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        log_dir=args.log_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train model
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()