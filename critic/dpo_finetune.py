"""
DPO (Direct Preference Optimization) Finetuning for Music Generation
Optimizes the generator model to maximize critic scores using preference pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Any, Optional
import json
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path
import wandb
from tqdm import tqdm

from .model import ComprehensiveCritic, CriticScore
from .classifier import AdherenceClassifier

logger = logging.getLogger(__name__)

@dataclass
class DPOSample:
    """Single DPO training sample with preference pair"""
    prompt: str
    control_json: Dict[str, Any]
    preferred_tokens: List[int]
    dispreferred_tokens: List[int]
    preferred_score: float
    dispreferred_score: float
    preference_margin: float

class DPODataset(Dataset):
    """Dataset for DPO training with preference pairs"""
    
    def __init__(self, data_path: str, max_seq_len: int = 512):
        self.samples = []
        self.max_seq_len = max_seq_len
        self._load_data(data_path)
    
    def _load_data(self, data_path: str):
        """Load preference pairs from JSON lines format"""
        with open(data_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                sample = DPOSample(
                    prompt=data['prompt'],
                    control_json=data['control_json'],
                    preferred_tokens=data['preferred_tokens'],
                    dispreferred_tokens=data['dispreferred_tokens'],
                    preferred_score=data['preferred_score'],
                    dispreferred_score=data['dispreferred_score'],
                    preference_margin=data['preferred_score'] - data['dispreferred_score']
                )
                self.samples.append(sample)
        
        logger.info(f"Loaded {len(self.samples)} DPO preference pairs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Pad or truncate sequences
        preferred = self._pad_sequence(sample.preferred_tokens)
        dispreferred = self._pad_sequence(sample.dispreferred_tokens)
        
        return {
            'prompt': sample.prompt,
            'control_json': sample.control_json,
            'preferred_tokens': torch.tensor(preferred, dtype=torch.long),
            'dispreferred_tokens': torch.tensor(dispreferred, dtype=torch.long),
            'preferred_score': torch.tensor(sample.preferred_score, dtype=torch.float),
            'dispreferred_score': torch.tensor(sample.dispreferred_score, dtype=torch.float),
            'preference_margin': torch.tensor(sample.preference_margin, dtype=torch.float)
        }
    
    def _pad_sequence(self, tokens: List[int]) -> List[int]:
        """Pad or truncate token sequence to max_seq_len"""
        if len(tokens) >= self.max_seq_len:
            return tokens[:self.max_seq_len]
        else:
            return tokens + [0] * (self.max_seq_len - len(tokens))  # 0 = PAD token

class DPOLoss(nn.Module):
    """Direct Preference Optimization loss function"""
    
    def __init__(self, beta: float = 0.1, reference_free: bool = False):
        super().__init__()
        self.beta = beta  # Temperature parameter
        self.reference_free = reference_free
    
    def forward(
        self,
        policy_preferred_logprobs: torch.Tensor,
        policy_dispreferred_logprobs: torch.Tensor,
        reference_preferred_logprobs: Optional[torch.Tensor] = None,
        reference_dispreferred_logprobs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute DPO loss
        
        Args:
            policy_preferred_logprobs: Log probabilities from current policy for preferred sequences
            policy_dispreferred_logprobs: Log probabilities from current policy for dispreferred sequences
            reference_preferred_logprobs: Log probabilities from reference model for preferred sequences
            reference_dispreferred_logprobs: Log probabilities from reference model for dispreferred sequences
            
        Returns:
            loss: DPO loss value
        """
        if self.reference_free:
            # Simplified DPO without reference model
            logits = self.beta * (policy_preferred_logprobs - policy_dispreferred_logprobs)
        else:
            # Standard DPO with reference model
            assert reference_preferred_logprobs is not None
            assert reference_dispreferred_logprobs is not None
            
            policy_ratio = policy_preferred_logprobs - policy_dispreferred_logprobs
            reference_ratio = reference_preferred_logprobs - reference_dispreferred_logprobs
            logits = self.beta * (policy_ratio - reference_ratio)
        
        # Binary cross-entropy loss (preferred should have higher probability)
        loss = -F.logsigmoid(logits).mean()
        return loss

class DPOTrainer:
    """Trainer for DPO finetuning of music generation models"""
    
    def __init__(
        self,
        policy_model: nn.Module,
        critic_model: ComprehensiveCritic,
        reference_model: Optional[nn.Module] = None,
        beta: float = 0.1,
        learning_rate: float = 1e-5,
        device: str = 'cuda'
    ):
        self.policy_model = policy_model.to(device)
        self.critic_model = critic_model.to(device)
        self.reference_model = reference_model.to(device) if reference_model else None
        self.device = device
        
        # Freeze critic and reference models
        self.critic_model.eval()
        for param in self.critic_model.parameters():
            param.requires_grad = False
            
        if self.reference_model:
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
        
        # Loss and optimizer
        self.dpo_loss = DPOLoss(beta=beta, reference_free=(reference_model is None))
        self.optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
        # Metrics tracking
        self.step = 0
        self.epoch = 0
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.policy_model.train()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            loss, accuracy = self._train_step(batch)
            total_loss += loss
            total_accuracy += accuracy
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss:.4f}',
                'acc': f'{accuracy:.3f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to wandb
            if wandb.run:
                wandb.log({
                    'train/loss': loss,
                    'train/accuracy': accuracy,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/step': self.step
                })
            
            self.step += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Move batch to device
        preferred_tokens = batch['preferred_tokens'].to(self.device)
        dispreferred_tokens = batch['dispreferred_tokens'].to(self.device)
        
        # Get log probabilities from policy model
        policy_preferred_logprobs = self._get_sequence_logprobs(
            self.policy_model, preferred_tokens
        )
        policy_dispreferred_logprobs = self._get_sequence_logprobs(
            self.policy_model, dispreferred_tokens
        )
        
        # Get reference log probabilities if available
        reference_preferred_logprobs = None
        reference_dispreferred_logprobs = None
        
        if self.reference_model:
            with torch.no_grad():
                reference_preferred_logprobs = self._get_sequence_logprobs(
                    self.reference_model, preferred_tokens
                )
                reference_dispreferred_logprobs = self._get_sequence_logprobs(
                    self.reference_model, dispreferred_tokens
                )
        
        # Compute DPO loss
        loss = self.dpo_loss(
            policy_preferred_logprobs,
            policy_dispreferred_logprobs,
            reference_preferred_logprobs,
            reference_dispreferred_logprobs
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Compute accuracy (how often preferred has higher probability)
        with torch.no_grad():
            preferred_better = (policy_preferred_logprobs > policy_dispreferred_logprobs).float()
            accuracy = preferred_better.mean().item()
        
        return loss.item(), accuracy
    
    def _get_sequence_logprobs(self, model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for token sequences"""
        # Assuming model outputs logits for next token prediction
        with torch.set_grad_enabled(model.training):
            outputs = model(tokens[:, :-1])  # Input all but last token
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Get log probabilities for actual next tokens
            log_probs = F.log_softmax(logits, dim=-1)
            target_tokens = tokens[:, 1:]  # Target is shifted by 1
            
            # Gather log probabilities for target tokens
            gathered_log_probs = log_probs.gather(
                dim=-1, 
                index=target_tokens.unsqueeze(-1)
            ).squeeze(-1)
            
            # Sum over sequence length (excluding padding tokens)
            mask = (target_tokens != 0).float()  # Assuming 0 is PAD token
            sequence_log_probs = (gathered_log_probs * mask).sum(dim=1)
            
            return sequence_log_probs
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.policy_model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        adherence_improvements = []
        style_improvements = []
        mix_improvements = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # DPO metrics
                preferred_tokens = batch['preferred_tokens'].to(self.device)
                dispreferred_tokens = batch['dispreferred_tokens'].to(self.device)
                
                policy_preferred_logprobs = self._get_sequence_logprobs(
                    self.policy_model, preferred_tokens
                )
                policy_dispreferred_logprobs = self._get_sequence_logprobs(
                    self.policy_model, dispreferred_tokens
                )
                
                reference_preferred_logprobs = None
                reference_dispreferred_logprobs = None
                
                if self.reference_model:
                    reference_preferred_logprobs = self._get_sequence_logprobs(
                        self.reference_model, preferred_tokens
                    )
                    reference_dispreferred_logprobs = self._get_sequence_logprobs(
                        self.reference_model, dispreferred_tokens
                    )
                
                loss = self.dpo_loss(
                    policy_preferred_logprobs,
                    policy_dispreferred_logprobs,
                    reference_preferred_logprobs,
                    reference_dispreferred_logprobs
                )
                
                total_loss += loss.item()
                
                # Accuracy
                preferred_better = (policy_preferred_logprobs > policy_dispreferred_logprobs).float()
                total_accuracy += preferred_better.mean().item()
                
                # Critic-based improvements (if we can generate samples)
                # This would require actually generating new samples and evaluating with critic
                # For now, use the provided scores
                preferred_scores = batch['preferred_score']
                dispreferred_scores = batch['dispreferred_score']
                improvements = preferred_scores - dispreferred_scores
                
                adherence_improvements.extend(improvements.tolist())
        
        num_batches = len(dataloader)
        
        metrics = {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches,
            'adherence_improvement_mean': np.mean(adherence_improvements),
            'adherence_improvement_std': np.std(adherence_improvements)
        }
        
        return metrics
    
    def save_checkpoint(self, path: str, metrics: Dict[str, float]):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.policy_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'metrics': metrics
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint['metrics']

def create_preference_pairs(
    samples: List[Dict[str, Any]],
    critic_model: ComprehensiveCritic,
    output_path: str
):
    """
    Create preference pairs from generated samples using critic scores
    
    Args:
        samples: List of dicts with 'prompt', 'control_json', 'tokens', 'audio_features'
        critic_model: Trained critic model for scoring
        output_path: Where to save preference pairs
    """
    pairs = []
    
    # Group samples by (prompt, control_json) pairs
    grouped_samples = {}
    for sample in samples:
        key = (sample['prompt'], json.dumps(sample['control_json'], sort_keys=True))
        if key not in grouped_samples:
            grouped_samples[key] = []
        grouped_samples[key].append(sample)
    
    for key, group_samples in grouped_samples.items():
        if len(group_samples) < 2:
            continue
        
        # Score all samples in group
        scored_samples = []
        for sample in group_samples:
            # This would need actual implementation to convert sample to critic inputs
            # For now, assume we have a scoring function
            score = score_sample_with_critic(sample, critic_model)
            scored_samples.append((sample, score))
        
        # Create all pairwise preferences
        for i in range(len(scored_samples)):
            for j in range(i + 1, len(scored_samples)):
                sample_i, score_i = scored_samples[i]
                sample_j, score_j = scored_samples[j]
                
                if abs(score_i - score_j) < 0.1:  # Skip if scores too close
                    continue
                
                if score_i > score_j:
                    preferred, dispreferred = sample_i, sample_j
                    pref_score, dispref_score = score_i, score_j
                else:
                    preferred, dispreferred = sample_j, sample_i
                    pref_score, dispref_score = score_j, score_i
                
                pair = {
                    'prompt': preferred['prompt'],
                    'control_json': preferred['control_json'],
                    'preferred_tokens': preferred['tokens'],
                    'dispreferred_tokens': dispreferred['tokens'],
                    'preferred_score': pref_score,
                    'dispreferred_score': dispref_score
                }
                pairs.append(pair)
    
    # Save pairs
    with open(output_path, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')
    
    logger.info(f"Created {len(pairs)} preference pairs, saved to {output_path}")

def score_sample_with_critic(sample: Dict[str, Any], critic: ComprehensiveCritic) -> float:
    """Score a single sample using the critic model"""
    # This is a placeholder - actual implementation would need to:
    # 1. Convert sample tokens to proper format
    # 2. Extract audio features
    # 3. Get style embeddings
    # 4. Run through critic model
    
    # For now, return a dummy score
    return np.random.random()

def main():
    """Main DPO training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DPO Finetuning for Music Generation")
    parser.add_argument("--train_data", required=True, help="Path to training preference pairs")
    parser.add_argument("--val_data", required=True, help="Path to validation preference pairs")
    parser.add_argument("--policy_model", required=True, help="Path to policy model checkpoint")
    parser.add_argument("--critic_model", required=True, help="Path to critic model checkpoint")
    parser.add_argument("--reference_model", help="Path to reference model checkpoint")
    parser.add_argument("--output_dir", required=True, help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature parameter")
    
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(project="music-dpo-finetuning", config=vars(args))
    
    # Load models
    # This would need actual model loading implementation
    policy_model = None  # Load your music generation model
    critic_model = None  # Load trained critic
    reference_model = None  # Load reference model if provided
    
    # Create datasets
    train_dataset = DPODataset(args.train_data)
    val_dataset = DPODataset(args.val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create trainer
    trainer = DPOTrainer(
        policy_model=policy_model,
        critic_model=critic_model,
        reference_model=reference_model,
        beta=args.beta,
        learning_rate=args.learning_rate
    )
    
    # Training loop
    best_accuracy = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for epoch in range(args.epochs):
        trainer.epoch = epoch
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validate
        val_metrics = trainer.evaluate(val_loader)
        
        # Log epoch metrics
        epoch_metrics = {
            f'epoch/train_{k}': v for k, v in train_metrics.items()
        }
        epoch_metrics.update({
            f'epoch/val_{k}': v for k, v in val_metrics.items()
        })
        
        if wandb.run:
            wandb.log(epoch_metrics)
        
        logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
                   f"Val Accuracy: {val_metrics['accuracy']:.3f}")
        
        # Save checkpoint if improved
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            checkpoint_path = output_dir / f"best_model_epoch_{epoch}.pt"
            trainer.save_checkpoint(str(checkpoint_path), val_metrics)
    
    logger.info(f"Training completed. Best validation accuracy: {best_accuracy:.3f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()