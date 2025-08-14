"""
Direct Preference Optimization (DPO) for fine-tuning symbolic music generation models.

Implements DPO training to align music generation with human preferences
using the trained critic model as a reward function.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from tqdm import tqdm
import math

from .model import CriticModel


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization loss function.
    
    Implements the DPO objective from "Direct Preference Optimization:
    Your Language Model is Secretly a Reward Model"
    """
    
    def __init__(self, beta: float = 0.1, reference_free: bool = False):
        """
        Initialize DPO loss.
        
        Args:
            beta: Temperature parameter controlling strength of KL penalty
            reference_free: Whether to use reference-free DPO variant
        """
        super().__init__()
        self.beta = beta
        self.reference_free = reference_free
    
    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: Optional[torch.Tensor] = None,
        reference_rejected_logps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute DPO loss.
        
        Args:
            policy_chosen_logps: Log probabilities of chosen sequences under policy
            policy_rejected_logps: Log probabilities of rejected sequences under policy  
            reference_chosen_logps: Log probabilities under reference model (if not reference_free)
            reference_rejected_logps: Log probabilities under reference model
            
        Returns:
            loss: DPO loss value
            stats: Dictionary of loss statistics
        """
        if self.reference_free:
            # Reference-free DPO
            logits = self.beta * (policy_chosen_logps - policy_rejected_logps)
        else:
            # Standard DPO with reference model
            if reference_chosen_logps is None or reference_rejected_logps is None:
                raise ValueError("Reference log probabilities required for standard DPO")
            
            chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
            rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
            logits = chosen_rewards - rejected_rewards
        
        # DPO loss: negative log-sigmoid of preference logits
        loss = -F.logsigmoid(logits).mean()
        
        # Compute statistics
        with torch.no_grad():
            # Accuracy: fraction where chosen is preferred
            accuracy = (logits > 0).float().mean()
            
            # Reward margins
            reward_margin = logits.mean()
            
            # KL divergence approximation
            if not self.reference_free and reference_chosen_logps is not None:
                kl_chosen = (policy_chosen_logps - reference_chosen_logps).mean()
                kl_rejected = (policy_rejected_logps - reference_rejected_logps).mean()
                kl_divergence = (kl_chosen + kl_rejected) / 2
            else:
                kl_divergence = torch.tensor(0.0)
        
        stats = {
            'loss': loss,
            'accuracy': accuracy,
            'reward_margin': reward_margin,
            'kl_divergence': kl_divergence,
            'logits_mean': logits.mean(),
            'logits_std': logits.std()
        }
        
        return loss, stats


class MusicGeneratorWrapper(nn.Module):
    """
    Wrapper for music generation model to interface with DPO training.
    
    This would wrap your actual symbolic music generation model
    (e.g., arrangement transformer, melody/harmony model).
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_sequence_length: int = 1024
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, sequence_length]
            attention_mask: Attention mask [batch_size, sequence_length]
            
        Returns:
            logits: Output logits [batch_size, sequence_length, vocab_size]
        """
        batch_size, seq_length = input_ids.shape
        
        # Create position IDs
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        causal_mask = causal_mask.to(input_ids.device)
        
        # Transformer forward pass
        hidden_states = self.transformer(hidden_states, mask=causal_mask)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        return logits
    
    def compute_log_probabilities(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute log probabilities of target sequences.
        
        Args:
            input_ids: Input token IDs [batch_size, sequence_length]
            target_ids: Target token IDs [batch_size, sequence_length]
            
        Returns:
            log_probs: Log probabilities of target sequence [batch_size]
        """
        logits = self.forward(input_ids)
        
        # Shift logits and targets for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = target_ids[..., 1:].contiguous()
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probabilities for target tokens
        target_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=shift_targets.unsqueeze(-1)
        ).squeeze(-1)
        
        # Sum over sequence length (ignoring padding tokens)
        valid_mask = (shift_targets != 0).float()  # Assuming 0 is padding token
        sequence_log_probs = (target_log_probs * valid_mask).sum(dim=1)
        
        return sequence_log_probs


class DPOTrainer:
    """Trainer for DPO fine-tuning of music generation models."""
    
    def __init__(
        self,
        policy_model: MusicGeneratorWrapper,
        reference_model: Optional[MusicGeneratorWrapper],
        critic_model: CriticModel,
        device: str = 'cuda',
        learning_rate: float = 5e-6,
        beta: float = 0.1,
        log_dir: str = './logs/dpo'
    ):
        self.policy_model = policy_model.to(device)
        self.reference_model = reference_model
        if self.reference_model:
            self.reference_model.to(device)
            self.reference_model.eval()  # Keep reference model frozen
        
        self.critic_model = critic_model.to(device)
        self.critic_model.eval()  # Critic is pre-trained and frozen
        
        self.device = device
        
        # Setup optimizer (only train policy model)
        self.optimizer = optim.AdamW(
            self.policy_model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # DPO loss
        self.dpo_loss = DPOLoss(beta=beta, reference_free=(reference_model is None))
        
        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_win_rate = 0.0
        
    def compute_reward(self, sequences: torch.Tensor, style_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute rewards for sequences using the critic model.
        
        Args:
            sequences: Generated token sequences [batch_size, seq_length]
            style_ids: Style IDs for each sequence [batch_size]
            
        Returns:
            rewards: Reward scores [batch_size]
        """
        # This is a placeholder - in practice you would:
        # 1. Convert token sequences back to MIDI/audio
        # 2. Extract mel spectrograms and auxiliary features
        # 3. Run through critic model
        
        batch_size = sequences.size(0)
        
        # Mock implementation - return random rewards for demonstration
        # In practice, this would involve sequence -> MIDI -> audio -> features -> critic
        mock_rewards = torch.randn(batch_size, device=self.device) * 0.1 + 0.5
        return torch.clamp(mock_rewards, 0, 1)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step."""
        self.policy_model.train()
        
        # Extract batch data
        chosen_sequences = batch['chosen_sequences'].to(self.device)
        rejected_sequences = batch['rejected_sequences'].to(self.device)
        style_ids = batch['style_ids'].to(self.device)
        
        # Compute log probabilities under policy model
        policy_chosen_logps = self.policy_model.compute_log_probabilities(
            chosen_sequences[:, :-1], chosen_sequences
        )
        policy_rejected_logps = self.policy_model.compute_log_probabilities(
            rejected_sequences[:, :-1], rejected_sequences
        )
        
        # Compute log probabilities under reference model (if available)
        reference_chosen_logps = None
        reference_rejected_logps = None
        
        if self.reference_model:
            with torch.no_grad():
                reference_chosen_logps = self.reference_model.compute_log_probabilities(
                    chosen_sequences[:, :-1], chosen_sequences
                )
                reference_rejected_logps = self.reference_model.compute_log_probabilities(
                    rejected_sequences[:, :-1], rejected_sequences
                )
        
        # Compute DPO loss
        loss, stats = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Convert stats to float for logging
        float_stats = {k: v.item() if torch.is_tensor(v) else v for k, v in stats.items()}
        
        return float_stats
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model performance."""
        self.policy_model.eval()
        
        total_stats = {
            'accuracy': 0.0,
            'reward_margin': 0.0,
            'kl_divergence': 0.0
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Extract batch data
                chosen_sequences = batch['chosen_sequences'].to(self.device)
                rejected_sequences = batch['rejected_sequences'].to(self.device)
                
                # Compute log probabilities
                policy_chosen_logps = self.policy_model.compute_log_probabilities(
                    chosen_sequences[:, :-1], chosen_sequences
                )
                policy_rejected_logps = self.policy_model.compute_log_probabilities(
                    rejected_sequences[:, :-1], rejected_sequences
                )
                
                reference_chosen_logps = None
                reference_rejected_logps = None
                
                if self.reference_model:
                    reference_chosen_logps = self.reference_model.compute_log_probabilities(
                        chosen_sequences[:, :-1], chosen_sequences
                    )
                    reference_rejected_logps = self.reference_model.compute_log_probabilities(
                        rejected_sequences[:, :-1], rejected_sequences
                    )
                
                # Compute stats
                _, stats = self.dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps
                )
                
                # Accumulate stats
                for key in total_stats:
                    if key in stats:
                        total_stats[key] += stats[key].item()
                
                num_batches += 1
        
        # Average stats
        for key in total_stats:
            total_stats[key] /= num_batches
            
        return total_stats
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        num_epochs: int,
        eval_every: int = 100,
        save_every: int = 500
    ):
        """Main training loop."""
        self.logger.info(f"Starting DPO training for {num_epochs} epochs")
        
        total_steps = len(train_dataloader) * num_epochs
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            epoch_stats = {
                'accuracy': 0.0,
                'reward_margin': 0.0,
                'kl_divergence': 0.0,
                'loss': 0.0
            }
            
            num_batches = 0
            
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch}')
            
            for batch in progress_bar:
                self.step += 1
                
                # Training step
                stats = self.train_step(batch)
                
                # Accumulate stats
                for key in epoch_stats:
                    if key in stats:
                        epoch_stats[key] += stats[key]
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{stats['loss']:.4f}",
                    'acc': f"{stats['accuracy']:.3f}",
                    'reward': f"{stats['reward_margin']:.3f}"
                })
                
                # Evaluation
                if self.step % eval_every == 0:
                    eval_stats = self.evaluate(eval_dataloader)
                    
                    self.logger.info(
                        f"Step {self.step} - "
                        f"Eval Accuracy: {eval_stats['accuracy']:.3f}, "
                        f"Eval Reward Margin: {eval_stats['reward_margin']:.3f}"
                    )
                    
                    # Log to tensorboard
                    for key, value in eval_stats.items():
                        self.writer.add_scalar(f'Eval/{key}', value, self.step)
                    
                    # Update best win rate
                    if eval_stats['accuracy'] > self.best_win_rate:
                        self.best_win_rate = eval_stats['accuracy']
                        self.save_checkpoint(is_best=True)
                
                # Save checkpoint
                if self.step % save_every == 0:
                    self.save_checkpoint()
                
                # Log training stats
                for key, value in stats.items():
                    self.writer.add_scalar(f'Train/{key}', value, self.step)
            
            # Average epoch stats
            for key in epoch_stats:
                epoch_stats[key] /= num_batches
            
            self.logger.info(
                f"Epoch {epoch} completed - "
                f"Average Loss: {epoch_stats['loss']:.4f}, "
                f"Average Accuracy: {epoch_stats['accuracy']:.3f}"
            )
        
        self.logger.info("DPO training completed!")
        self.writer.close()
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.policy_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_win_rate': self.best_win_rate
        }
        
        # Save latest checkpoint
        checkpoint_path = self.log_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.log_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint with win rate: {self.best_win_rate:.3f}")


def create_mock_dpo_data(output_dir: str, num_pairs: int = 1000):
    """Create mock preference pairs for DPO training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mock vocabulary size and sequence length
    vocab_size = 1000
    seq_length = 128
    
    data = []
    
    for i in range(num_pairs):
        # Generate random sequences
        chosen = torch.randint(1, vocab_size, (seq_length,)).tolist()
        rejected = torch.randint(1, vocab_size, (seq_length,)).tolist()
        style_id = np.random.choice([0, 1, 2])  # rock_punk, rnb_ballad, country_pop
        
        data.append({
            'chosen_sequences': chosen,
            'rejected_sequences': rejected,
            'style_ids': style_id
        })
    
    # Save as JSON
    with open(output_dir / 'dpo_pairs.json', 'w') as f:
        json.dump(data, f)
    
    print(f"Created {num_pairs} mock DPO training pairs")


def main():
    """Main DPO training function."""
    parser = argparse.ArgumentParser(description='DPO finetuning for music generation')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to DPO training data')
    parser.add_argument('--critic_checkpoint', type=str, required=True,
                        help='Path to trained critic model checkpoint')
    parser.add_argument('--policy_checkpoint', type=str, default=None,
                        help='Path to policy model checkpoint to start from')
    parser.add_argument('--reference_checkpoint', type=str, default=None,
                        help='Path to reference model checkpoint')
    parser.add_argument('--log_dir', type=str, default='./logs/dpo_training',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-6,
                        help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='DPO temperature parameter')
    parser.add_argument('--mock_data', action='store_true',
                        help='Create mock data for testing')
    
    args = parser.parse_args()
    
    # Create mock data if requested
    if args.mock_data:
        create_mock_dpo_data(args.data_path, num_pairs=1000)
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load models (placeholder - implement actual loading)
    # In practice, you would load your actual music generation models here
    policy_model = MusicGeneratorWrapper(vocab_size=1000)
    reference_model = MusicGeneratorWrapper(vocab_size=1000) if args.reference_checkpoint else None
    
    # Load critic model
    from .model import create_critic_model
    critic_model = create_critic_model(device=device)
    if args.critic_checkpoint:
        checkpoint = torch.load(args.critic_checkpoint, map_location=device)
        critic_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create trainer
    trainer = DPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        critic_model=critic_model,
        device=device,
        learning_rate=args.learning_rate,
        beta=args.beta,
        log_dir=args.log_dir
    )
    
    # Note: In practice, you would implement actual data loading here
    # This is a placeholder for the training call
    print("DPO trainer created successfully!")
    print("Note: Implement actual data loading and training loop for your specific use case")


if __name__ == "__main__":
    main()