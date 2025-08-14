#!/usr/bin/env python3
"""
Training script for Melody & Harmony Transformer

Trains a style-conditioned transformer for generating melody and chord progressions
with constraints for musicality and style consistency.
"""

import os
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Local imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.mh_transformer import MelodyHarmonyTransformer, MHTrainingLoss
from models.tokenizer import MIDITokenizer
from utils.constraints import ConstraintMaskGenerator, RepetitionController
import wandb


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MHDataset(Dataset):
    """Dataset for melody and harmony training"""
    
    def __init__(
        self,
        data_dir: str,
        tokenizer: MIDITokenizer,
        max_seq_len: int = 512,
        style_mapping: Optional[Dict[str, int]] = None
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Default style mapping
        self.style_mapping = style_mapping or {
            'rock_punk': 0,
            'rnb_ballad': 1,
            'country_pop': 2
        }
        
        # Load data files
        self.data_files = self._load_data_files()
        
    def _load_data_files(self) -> List[Dict]:
        """Load and process data files"""
        data_files = []
        
        # Look for processed MIDI files with metadata
        for style_dir in self.data_dir.iterdir():
            if not style_dir.is_dir():
                continue
                
            style_name = style_dir.name
            if style_name not in self.style_mapping:
                continue
                
            # Look for melody/harmony files
            mh_files = list(style_dir.glob("**/melody_harmony.json"))
            
            for mh_file in mh_files:
                try:
                    with open(mh_file, 'r') as f:
                        data = json.load(f)
                    
                    # Add style information
                    data['style'] = style_name
                    data['style_id'] = self.style_mapping[style_name]
                    data['file_path'] = str(mh_file)
                    
                    data_files.append(data)
                    
                except Exception as e:
                    logger.warning(f"Failed to load {mh_file}: {e}")
        
        logger.info(f"Loaded {len(data_files)} melody/harmony files")
        return data_files
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get training sample"""
        data = self.data_files[idx]
        
        # Extract basic information
        style_id = data['style_id']
        key_str = data.get('key', 'C_major')
        section_str = data.get('section', 'verse')
        
        # Parse key signature
        key_parts = key_str.split('_')
        key_root = self._key_to_int(key_parts[0])
        is_major = len(key_parts) > 1 and key_parts[1] == 'major'
        key_id = key_root * 2 + (0 if is_major else 1)  # 0-23 encoding
        
        # Parse section
        section_mapping = {
            'intro': 0, 'verse': 1, 'chorus': 2, 'bridge': 3, 'outro': 4
        }
        section_id = section_mapping.get(section_str.lower(), 1)
        
        # Get MIDI events
        midi_events = data.get('midi_events', [])
        
        # Tokenize MIDI events
        try:
            token_sequence = self.tokenizer.encode_events(midi_events)
        except Exception as e:
            logger.warning(f"Failed to tokenize events in sample {idx}: {e}")
            # Return dummy sequence
            token_sequence = [self.tokenizer.vocab['PAD']] * 64
        
        # Truncate or pad sequence
        if len(token_sequence) > self.max_seq_len:
            token_sequence = token_sequence[:self.max_seq_len]
        else:
            # Pad with PAD tokens
            pad_length = self.max_seq_len - len(token_sequence)
            token_sequence.extend([self.tokenizer.vocab['PAD']] * pad_length)
        
        # Create input and target sequences (shift by 1 for teacher forcing)
        input_ids = torch.tensor(token_sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_sequence[1:], dtype=torch.long)
        
        # Create attention mask (ignore padding)
        attention_mask = (input_ids == self.tokenizer.vocab['PAD'])
        
        # Extract chord progression if available
        chord_sequence = data.get('chord_progression', [])
        chord_targets = self._encode_chord_sequence(chord_sequence, len(input_ids))
        
        # Create scale compatibility targets
        scale_targets = self._create_scale_targets(key_root, is_major, len(input_ids))
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask,
            'style_ids': torch.tensor(style_id, dtype=torch.long),
            'key_ids': torch.tensor(key_id, dtype=torch.long),
            'section_ids': torch.tensor(section_id, dtype=torch.long),
            'chord_targets': chord_targets,
            'scale_targets': scale_targets,
        }
    
    def _key_to_int(self, key_str: str) -> int:
        """Convert key string to integer (0-11, C=0)"""
        key_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        return key_map.get(key_str, 0)
    
    def _encode_chord_sequence(self, chord_sequence: List[str], seq_len: int) -> torch.Tensor:
        """Encode chord sequence for training"""
        # Simplified chord encoding - map to integers
        chord_map = {
            'C': 0, 'Dm': 1, 'Em': 2, 'F': 3, 'G': 4, 'Am': 5, 'Bdim': 6,
            'C7': 7, 'Dm7': 8, 'Em7': 9, 'Fmaj7': 10, 'G7': 11, 'Am7': 12
        }
        
        if not chord_sequence:
            return torch.zeros(seq_len, dtype=torch.long)
        
        # Distribute chords across sequence
        chord_targets = torch.zeros(seq_len, dtype=torch.long)
        chords_per_segment = max(1, seq_len // len(chord_sequence))
        
        for i, chord in enumerate(chord_sequence):
            chord_id = chord_map.get(chord, 0)
            start_idx = i * chords_per_segment
            end_idx = min((i + 1) * chords_per_segment, seq_len)
            chord_targets[start_idx:end_idx] = chord_id
        
        return chord_targets
    
    def _create_scale_targets(self, key_root: int, is_major: bool, seq_len: int) -> torch.Tensor:
        """Create scale compatibility targets"""
        # Major scale intervals
        major_intervals = [0, 2, 4, 5, 7, 9, 11]
        minor_intervals = [0, 2, 3, 5, 7, 8, 10]
        
        intervals = major_intervals if is_major else minor_intervals
        scale_notes = [(key_root + interval) % 12 for interval in intervals]
        
        # Create binary targets for each chromatic note
        scale_targets = torch.zeros(seq_len, 12, dtype=torch.float)
        for note in scale_notes:
            scale_targets[:, note] = 1.0
        
        return scale_targets


def create_data_loaders(
    config: Dict,
    tokenizer: MIDITokenizer
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    # Training dataset
    train_dataset = MHDataset(
        data_dir=config['data']['train_dir'],
        tokenizer=tokenizer,
        max_seq_len=config['model']['max_seq_len'],
        style_mapping=config['data']['style_mapping']
    )
    
    # Validation dataset
    val_dataset = MHDataset(
        data_dir=config['data']['val_dir'],
        tokenizer=tokenizer,
        max_seq_len=config['model']['max_seq_len'],
        style_mapping=config['data']['style_mapping']
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_epoch(
    model: MelodyHarmonyTransformer,
    criterion: MHTrainingLoss,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    log_interval: int = 100
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        style_ids = batch['style_ids'].to(device)
        key_ids = batch['key_ids'].to(device)
        section_ids = batch['section_ids'].to(device)
        chord_targets = batch['chord_targets'].to(device)
        scale_targets = batch['scale_targets'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            style_ids=style_ids,
            key_ids=key_ids,
            section_ids=section_ids,
            attention_mask=attention_mask
        )
        
        # Compute loss
        losses = criterion(
            outputs=outputs,
            target_ids=target_ids,
            key_ids=key_ids,
            chord_targets=chord_targets,
            scale_targets=scale_targets
        )
        
        loss = losses['total_loss']
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        
        total_loss += loss.item()
        
        # Logging
        if batch_idx % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            
            logger.info(
                f'Epoch: {epoch}, Batch: {batch_idx}/{num_batches}, '
                f'Loss: {loss.item():.4f}, LR: {current_lr:.6f}'
            )
            
            # TensorBoard logging
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('Train/Total_Loss', loss.item(), global_step)
            writer.add_scalar('Train/Learning_Rate', current_lr, global_step)
            
            # Log component losses
            for loss_name, loss_value in losses.items():
                if loss_name != 'total_loss':
                    writer.add_scalar(f'Train/{loss_name}', loss_value.item(), global_step)
            
            # Log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    'train/total_loss': loss.item(),
                    'train/learning_rate': current_lr,
                    'epoch': epoch,
                    'batch': batch_idx
                })
    
    return total_loss / num_batches


def validate_epoch(
    model: MelodyHarmonyTransformer,
    criterion: MHTrainingLoss,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter
) -> float:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            style_ids = batch['style_ids'].to(device)
            key_ids = batch['key_ids'].to(device)
            section_ids = batch['section_ids'].to(device)
            chord_targets = batch['chord_targets'].to(device)
            scale_targets = batch['scale_targets'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                style_ids=style_ids,
                key_ids=key_ids,
                section_ids=section_ids,
                attention_mask=attention_mask
            )
            
            # Compute loss
            losses = criterion(
                outputs=outputs,
                target_ids=target_ids,
                key_ids=key_ids,
                chord_targets=chord_targets,
                scale_targets=scale_targets
            )
            
            loss = losses['total_loss']
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    
    # TensorBoard logging
    writer.add_scalar('Val/Total_Loss', avg_loss, epoch)
    
    # Log to wandb if available
    if wandb.run is not None:
        wandb.log({
            'val/total_loss': avg_loss,
            'epoch': epoch
        })
    
    logger.info(f'Validation Epoch: {epoch}, Avg Loss: {avg_loss:.4f}')
    
    return avg_loss


def save_checkpoint(
    model: MelodyHarmonyTransformer,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    config: Dict,
    checkpoint_dir: str,
    is_best: bool = False
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        logger.info(f'Saved best model to {best_path}')
    
    logger.info(f'Saved checkpoint to {checkpoint_path}')


def main():
    parser = argparse.ArgumentParser(description='Train Melody & Harmony Transformer')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--output-dir', type=str, default='./outputs/mh_training', help='Output directory')
    parser.add_argument('--wandb-project', type=str, help='Wandb project name')
    parser.add_argument('--wandb-run-name', type=str, help='Wandb run name')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb if specified
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=config
        )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load tokenizer
    tokenizer = MIDITokenizer()
    if os.path.exists('vocab.json'):
        tokenizer.load_vocab('vocab.json')
    else:
        logger.error('Tokenizer vocabulary not found. Please run tokenizer training first.')
        return
    
    # Create model
    model_config = config['model']
    model_config['vocab_size'] = len(tokenizer.vocab)
    model = MelodyHarmonyTransformer(**model_config)
    model = model.to(device)
    
    logger.info(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')
    
    # Create loss criterion
    loss_config = config['loss']
    criterion = MHTrainingLoss(
        vocab_size=len(tokenizer.vocab),
        **loss_config
    )
    
    # Create optimizer
    optimizer_config = config['optimizer']
    optimizer = optim.AdamW(
        model.parameters(),
        lr=optimizer_config['learning_rate'],
        weight_decay=optimizer_config['weight_decay']
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config, tokenizer)
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        logger.info(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f'Starting epoch {epoch}/{num_epochs}')
        
        # Training
        train_loss = train_epoch(
            model, criterion, optimizer, train_loader, device, epoch, writer
        )
        
        # Validation
        val_loss = validate_epoch(
            model, criterion, val_loader, device, epoch, writer
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        save_checkpoint(
            model, optimizer, epoch, val_loss, config, args.output_dir, is_best
        )
        
        logger.info(f'Epoch {epoch} completed. Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    logger.info('Training completed!')
    
    # Close writer
    writer.close()
    
    # Finish wandb run
    if wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    main()