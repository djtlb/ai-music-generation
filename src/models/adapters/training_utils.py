"""
Training utilities for hierarchical LoRA adapters.

Provides:
- ParentAdapterTrainer: Train parent genre adapters from style packs
- ChildAdapterTrainer: Train child sub-style adapters inheriting from parents
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Any, Tuple
import yaml
import json
import logging
from pathlib import Path
import argparse

from .style_adapter import StyleAdapter, HierarchicalStyleAdapter
from .lora_layer import freeze_non_lora_parameters, get_lora_parameters


logger = logging.getLogger(__name__)


class StylePackDataset(Dataset):
    """Dataset loader for style pack training data."""
    
    def __init__(
        self,
        style_pack_dir: str,
        tokenizer: Any,
        max_seq_len: int = 512,
        augment: bool = True
    ):
        self.style_pack_dir = Path(style_pack_dir)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.augment = augment
        
        # Load metadata
        meta_path = self.style_pack_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            
        # Load reference MIDI files
        self.midi_files = list((self.style_pack_dir / "refs_midi").glob("*.mid"))
        self.midi_files.extend(list((self.style_pack_dir / "refs_midi").glob("*.midi")))
        
        # Load processed data if available
        self.processed_files = list((self.style_pack_dir / "processed").glob("*.json"))
        
        self.data_files = self.processed_files if self.processed_files else self.midi_files
        print(f"Loaded {len(self.data_files)} files from {style_pack_dir}")
        
    def __len__(self) -> int:
        return len(self.data_files)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get training sample from style pack."""
        file_path = self.data_files[idx]
        
        if file_path.suffix == '.json':
            # Load processed data
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract MIDI events
            midi_events = data.get('midi_events', [])
            style_info = data.get('style_info', {})
            
        else:
            # Process MIDI file on-the-fly
            midi_events = self._process_midi_file(file_path)
            style_info = self.metadata
            
        # Tokenize events
        try:
            token_sequence = self.tokenizer.encode_events(midi_events)
        except Exception as e:
            logger.warning(f"Tokenization failed for {file_path}: {e}")
            # Return dummy sequence
            token_sequence = [self.tokenizer.vocab.get('PAD', 0)] * 64
            
        # Apply data augmentation
        if self.augment:
            token_sequence = self._augment_sequence(token_sequence)
            
        # Truncate/pad to max length
        if len(token_sequence) > self.max_seq_len:
            token_sequence = token_sequence[:self.max_seq_len]
        else:
            pad_token = self.tokenizer.vocab.get('PAD', 0)
            token_sequence.extend([pad_token] * (self.max_seq_len - len(token_sequence)))
            
        # Create input/target sequences for language modeling
        input_ids = torch.tensor(token_sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(token_sequence[1:], dtype=torch.long)
        
        # Create attention mask
        pad_token = self.tokenizer.vocab.get('PAD', 0)
        attention_mask = (input_ids != pad_token).float()
        
        # Extract style features
        bpm = style_info.get('bpm', 120)
        key_sig = style_info.get('key', 'C_major')
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask,
            'bpm': torch.tensor(bpm, dtype=torch.float),
            'key_signature': key_sig,
            'file_path': str(file_path)
        }
        
    def _process_midi_file(self, midi_path: Path) -> List[Dict]:
        """Process MIDI file into event sequence."""
        # Simplified MIDI processing - implement with mido or similar
        # For now, return dummy events
        return [
            {'type': 'note_on', 'note': 60, 'velocity': 80, 'time': 0},
            {'type': 'note_off', 'note': 60, 'time': 480}
        ]
        
    def _augment_sequence(self, sequence: List[int]) -> List[int]:
        """Apply data augmentation to token sequence."""
        # Implement sequence-level augmentations:
        # - Transpose
        # - Time stretch
        # - Velocity scaling
        # For now, return original sequence
        return sequence


class ParentAdapterTrainer:
    """
    Trainer for parent genre adapters.
    
    Trains LoRA adapters to capture broad genre characteristics
    from style pack data (e.g., all pop sub-styles).
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        parent_style: str,
        config: Dict[str, Any],
        tokenizer: Any
    ):
        self.base_model = base_model
        self.parent_style = parent_style
        self.config = config
        self.tokenizer = tokenizer
        
        # Load parent genre config
        self.parent_config = self._load_parent_config(parent_style)
        
        # Create parent style adapter
        lora_config = self.parent_config.get('lora', {})
        self.adapter = StyleAdapter(
            base_model=base_model,
            style_name=parent_style,
            rank=lora_config.get('rank', 16),
            alpha=lora_config.get('alpha', 32.0),
            dropout=lora_config.get('dropout', 0.1),
            target_modules=lora_config.get('target_modules', ['attention', 'feed_forward'])
        )
        
        # Freeze base model, only train LoRA parameters
        freeze_non_lora_parameters(self.adapter)
        
        # Setup training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.vocab.get('PAD', 0))
        
    def _load_parent_config(self, parent_style: str) -> Dict:
        """Load parent genre configuration."""
        config_path = Path(f"configs/genres/{parent_style}.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"No config found for parent style: {parent_style}")
            return {}
            
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for LoRA parameters only."""
        lora_params = get_lora_parameters(self.adapter)
        
        optimizer_config = self.config.get('optimizer', {})
        return optim.AdamW(
            lora_params.values(),
            lr=optimizer_config.get('learning_rate', 1e-4),
            weight_decay=optimizer_config.get('weight_decay', 0.01)
        )
        
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['num_epochs'],
            eta_min=1e-6
        )
        
    def train(
        self,
        style_pack_dir: str,
        output_dir: str,
        num_epochs: int = 10,
        batch_size: int = 8,
        eval_interval: int = 100
    ) -> None:
        """
        Train parent adapter on style pack data.
        
        Args:
            style_pack_dir: Directory containing style pack data
            output_dir: Output directory for checkpoints
            num_epochs: Number of training epochs
            batch_size: Training batch size
            eval_interval: Steps between evaluations
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create dataset and dataloader
        dataset = StylePackDataset(
            style_pack_dir=style_pack_dir,
            tokenizer=self.tokenizer,
            max_seq_len=self.config['model']['max_seq_len']
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Setup logging
        writer = SummaryWriter(output_path / 'tensorboard')
        device = next(self.adapter.parameters()).device
        
        # Training loop
        self.adapter.train()
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = len(dataloader)
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass
                outputs = self.adapter(input_ids, attention_mask=attention_mask)
                
                # Compute loss (assuming outputs.logits exists)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                    
                loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), max_norm=1.0)
                
                # Update parameters
                self.optimizer.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                # Logging
                if global_step % eval_interval == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    lr = self.optimizer.param_groups[0]['lr']
                    
                    logger.info(
                        f"Epoch {epoch}, Step {global_step}, "
                        f"Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, LR: {lr:.6f}"
                    )
                    
                    writer.add_scalar('Train/Loss', loss.item(), global_step)
                    writer.add_scalar('Train/LR', lr, global_step)
                    
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            self.scheduler.step()
            
            # Save checkpoint
            is_best = avg_epoch_loss < best_loss
            if is_best:
                best_loss = avg_epoch_loss
                
            self._save_checkpoint(
                output_path, epoch, avg_epoch_loss, is_best
            )
            
            logger.info(f"Epoch {epoch} completed. Avg Loss: {avg_epoch_loss:.4f}")
            
        writer.close()
        logger.info(f"Parent adapter training completed for '{self.parent_style}'")
        
    def _save_checkpoint(
        self,
        output_dir: Path,
        epoch: int,
        loss: float,
        is_best: bool = False
    ) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'adapter_state_dict': self.adapter.get_style_state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'parent_config': self.parent_config
        }
        
        # Save latest checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = output_dir / f"{self.parent_style}_best.pt"
            torch.save(checkpoint, best_path)
            
            # Also save LoRA-only checkpoint
            lora_path = output_dir / f"{self.parent_style}.lora"
            torch.save(self.adapter.get_style_state_dict(), lora_path)
            
        logger.info(f"Saved checkpoint to {checkpoint_path}")


class ChildAdapterTrainer:
    """
    Trainer for child sub-style adapters.
    
    Trains adapters that inherit from parent adapters and learn
    specific sub-style variations (e.g., dance_pop from pop).
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        parent_style: str,
        child_style: str,
        parent_adapter_path: str,
        config: Dict[str, Any],
        tokenizer: Any
    ):
        self.base_model = base_model
        self.parent_style = parent_style
        self.child_style = child_style
        self.config = config
        self.tokenizer = tokenizer
        
        # Load parent adapter
        self.parent_adapter = self._load_parent_adapter(parent_adapter_path)
        
        # Create hierarchical adapter
        self.hierarchical_adapter = HierarchicalStyleAdapter(
            base_model=base_model,
            parent_style=parent_style,
            child_style=child_style
        )
        
        # Load parent weights into hierarchical adapter
        self.hierarchical_adapter.parent_adapter.load_style_state_dict(
            self.parent_adapter.get_style_state_dict()
        )
        
        # Freeze parent adapter, only train child
        self._freeze_parent_parameters()
        
        # Setup training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.vocab.get('PAD', 0))
        
    def _load_parent_adapter(self, parent_adapter_path: str) -> StyleAdapter:
        """Load parent adapter from checkpoint.""" 
        from .style_adapter import load_style_adapter
        return load_style_adapter(self.base_model, parent_adapter_path)
        
    def _freeze_parent_parameters(self) -> None:
        """Freeze parent adapter parameters."""
        for param in self.hierarchical_adapter.parent_adapter.parameters():
            param.requires_grad = False
            
        # Only child adapter parameters should be trainable
        if self.hierarchical_adapter.child_adapter:
            for param in self.hierarchical_adapter.child_adapter.parameters():
                param.requires_grad = True
                
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for child LoRA parameters only."""
        if not self.hierarchical_adapter.child_adapter:
            raise ValueError("Child adapter not initialized")
            
        child_params = get_lora_parameters(self.hierarchical_adapter.child_adapter)
        
        optimizer_config = self.config.get('optimizer', {})
        return optim.AdamW(
            child_params.values(),
            lr=optimizer_config.get('learning_rate', 5e-5),  # Lower LR for child
            weight_decay=optimizer_config.get('weight_decay', 0.01)
        )
        
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['num_epochs'],
            eta_min=1e-7
        )
        
    def train(
        self,
        child_style_pack_dir: str,
        output_dir: str,
        num_epochs: int = 5,  # Fewer epochs for child training
        batch_size: int = 4,
        eval_interval: int = 50
    ) -> None:
        """
        Train child adapter on child-specific style pack data.
        
        Args:
            child_style_pack_dir: Directory containing child style data  
            output_dir: Output directory for checkpoints
            num_epochs: Number of training epochs
            batch_size: Training batch size
            eval_interval: Steps between evaluations
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create dataset focused on child style differences
        dataset = StylePackDataset(
            style_pack_dir=child_style_pack_dir,
            tokenizer=self.tokenizer,
            max_seq_len=self.config['model']['max_seq_len'],
            augment=True  # More augmentation for smaller child datasets
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        # Setup logging
        writer = SummaryWriter(output_path / 'tensorboard')
        device = next(self.hierarchical_adapter.parameters()).device
        
        # Training loop
        self.hierarchical_adapter.train()
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = len(dataloader)
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass through hierarchical adapter
                outputs = self.hierarchical_adapter(input_ids, attention_mask=attention_mask)
                
                # Compute loss
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                    
                loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.hierarchical_adapter.child_adapter.parameters(), max_norm=0.5
                )
                
                # Update parameters
                self.optimizer.step()
                
                epoch_loss += loss.item()
                global_step += 1
                
                # Logging
                if global_step % eval_interval == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    lr = self.optimizer.param_groups[0]['lr']
                    
                    logger.info(
                        f"Child Training - Epoch {epoch}, Step {global_step}, "
                        f"Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, LR: {lr:.6f}"
                    )
                    
                    writer.add_scalar('Train/Child_Loss', loss.item(), global_step)
                    writer.add_scalar('Train/Child_LR', lr, global_step)
                    
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            self.scheduler.step()
            
            # Save checkpoint
            is_best = avg_epoch_loss < best_loss
            if is_best:
                best_loss = avg_epoch_loss
                
            self._save_checkpoint(
                output_path, epoch, avg_epoch_loss, is_best
            )
            
            logger.info(f"Child Epoch {epoch} completed. Avg Loss: {avg_epoch_loss:.4f}")
            
        writer.close()
        logger.info(f"Child adapter training completed for '{self.child_style}'")
        
    def _save_checkpoint(
        self,
        output_dir: Path,
        epoch: int,
        loss: float,
        is_best: bool = False
    ) -> None:
        """Save child training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'hierarchical_state_dict': self.hierarchical_adapter.get_hierarchical_state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = output_dir / f"child_checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = output_dir / f"{self.child_style}_best.pt"
            torch.save(checkpoint, best_path)
            
            # Save child LoRA-only checkpoint
            if self.hierarchical_adapter.child_adapter:
                child_lora_path = output_dir / f"{self.child_style}.lora"
                torch.save(
                    self.hierarchical_adapter.child_adapter.get_style_state_dict(),
                    child_lora_path
                )
                
        logger.info(f"Saved child checkpoint to {checkpoint_path}")