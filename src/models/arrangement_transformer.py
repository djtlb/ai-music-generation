"""
Transformer Decoder for Song Arrangement Generation

Generates sequences of section tokens with bar counts based on style, BPM, and target duration.
Includes teacher forcing, coverage penalty to prevent loops, and style conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Any
import math
import yaml
from pathlib import Path


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CoverageMechanism(nn.Module):
    """Coverage mechanism to prevent repetitive sequences"""
    
    def __init__(self, d_model: int, max_repeat_length: int = 4):
        super().__init__()
        self.max_repeat_length = max_repeat_length
        self.coverage_projection = nn.Linear(d_model, 1)
        
    def compute_coverage_penalty(self, 
                                logits: torch.Tensor, 
                                generated_tokens: torch.Tensor,
                                penalty_weight: float = 0.3) -> torch.Tensor:
        """
        Compute coverage penalty to discourage repetitive patterns
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
            generated_tokens: [batch_size, seq_len] previously generated tokens
            penalty_weight: strength of penalty
            
        Returns:
            Modified logits with coverage penalty applied
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        if seq_len < self.max_repeat_length:
            return logits
            
        penalty = torch.zeros_like(logits)
        
        # Check for recent repetitions
        for i in range(seq_len):
            if i >= self.max_repeat_length:
                # Get recent window
                recent_window = generated_tokens[:, i-self.max_repeat_length:i]
                
                # Check if current position would create repetition
                for j in range(vocab_size):
                    # Count occurrences of token j in recent window
                    token_count = (recent_window == j).sum(dim=1).float()
                    
                    # Apply penalty proportional to recent frequency
                    penalty[:, i, j] = penalty_weight * token_count
                    
        return logits - penalty


class ArrangementTokenizer:
    """Tokenizer for arrangement sequences"""
    
    def __init__(self):
        # Section types
        self.section_types = ['INTRO', 'VERSE', 'CHORUS', 'BRIDGE', 'OUTRO']
        
        # Bar counts (common lengths)
        self.bar_counts = [2, 4, 8, 16, 32]
        
        # Styles
        self.styles = ['rock_punk', 'rnb_ballad', 'country_pop']
        
        # Special tokens
        self.special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>']
        
        # Build vocabulary
        self.token_to_id = {}
        self.id_to_token = {}
        
        token_id = 0
        
        # Add special tokens
        for token in self.special_tokens:
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
            token_id += 1
            
        # Add section tokens with bar counts
        for section in self.section_types:
            for bars in self.bar_counts:
                token = f"{section}_{bars}"
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
                token_id += 1
                
        self.vocab_size = len(self.token_to_id)
        self.pad_token_id = self.token_to_id['<PAD>']
        self.start_token_id = self.token_to_id['<START>']
        self.end_token_id = self.token_to_id['<END>']
        
    def encode_arrangement(self, sections: List[Dict]) -> List[int]:
        """Encode arrangement sections to token IDs"""
        tokens = [self.start_token_id]
        
        for section in sections:
            section_type = section['type']
            bar_count = section['length_bars']
            
            # Find closest bar count in vocabulary
            closest_bars = min(self.bar_counts, key=lambda x: abs(x - bar_count))
            token = f"{section_type}_{closest_bars}"
            
            if token in self.token_to_id:
                tokens.append(self.token_to_id[token])
            else:
                tokens.append(self.token_to_id['<UNK>'])
                
        tokens.append(self.end_token_id)
        return tokens
        
    def decode_arrangement(self, token_ids: List[int]) -> List[Dict]:
        """Decode token IDs back to arrangement sections"""
        sections = []
        current_bar = 0
        
        for token_id in token_ids:
            if token_id in [self.pad_token_id, self.start_token_id, self.end_token_id]:
                continue
                
            token = self.id_to_token.get(token_id, '<UNK>')
            if '_' in token and token != '<UNK>':
                section_type, bar_count_str = token.rsplit('_', 1)
                if section_type in self.section_types:
                    try:
                        bar_count = int(bar_count_str)
                        sections.append({
                            'type': section_type,
                            'start_bar': current_bar,
                            'length_bars': bar_count
                        })
                        current_bar += bar_count
                    except ValueError:
                        continue
                        
        return sections


class ArrangementTransformer(pl.LightningModule):
    """Transformer decoder for arrangement generation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        
        # Configuration
        self.config = config
        model_config = config['model']
        training_config = config['training']
        
        # Model parameters
        self.d_model = model_config['d_model']
        self.n_heads = model_config['n_heads']
        self.n_layers = model_config['n_layers']
        self.d_ff = model_config['d_ff']
        self.dropout = model_config['dropout']
        self.max_seq_length = model_config['max_seq_length']
        
        # Coverage penalty parameters
        self.coverage_penalty = model_config['coverage_penalty']
        self.max_repeat_length = model_config['max_repeat_length']
        
        # Style conditioning
        self.style_embedding_dim = model_config['style_embedding_dim']
        
        # Training parameters
        self.learning_rate = training_config['learning_rate']
        self.teacher_forcing_ratio = training_config['teacher_forcing_ratio']
        self.teacher_forcing_decay = training_config['teacher_forcing_decay']
        self.min_teacher_forcing = training_config['min_teacher_forcing']
        
        # Initialize tokenizer
        self.tokenizer = ArrangementTokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        
        # Style embeddings
        self.style_to_id = {style: i for i, style in enumerate(self.tokenizer.styles)}
        self.style_embedding = nn.Embedding(len(self.tokenizer.styles), self.style_embedding_dim)
        
        # Tempo and duration embeddings (normalized)
        self.tempo_projection = nn.Linear(1, self.style_embedding_dim)
        self.duration_projection = nn.Linear(1, self.style_embedding_dim)
        
        # Condition projection
        self.condition_projection = nn.Linear(self.style_embedding_dim * 3, self.d_model)
        
        # Token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, self.max_seq_length, self.dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, self.n_layers)
        
        # Coverage mechanism
        self.coverage = CoverageMechanism(self.d_model, self.max_repeat_length)
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create padding mask for attention"""
        return (x == self.tokenizer.pad_token_id)
        
    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for decoder"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask.bool().to(self.device)
        
    def encode_conditions(self, 
                         styles: torch.Tensor,
                         tempos: torch.Tensor, 
                         durations: torch.Tensor) -> torch.Tensor:
        """Encode style, tempo, and duration conditions"""
        batch_size = styles.shape[0]
        
        # Style embeddings
        style_emb = self.style_embedding(styles)  # [batch_size, style_dim]
        
        # Tempo embeddings (normalize to 0-1 range)
        tempo_norm = tempos.float().unsqueeze(-1) / 200.0  # Assume max tempo 200
        tempo_emb = self.tempo_projection(tempo_norm)  # [batch_size, style_dim]
        
        # Duration embeddings (normalize to 0-1 range)
        duration_norm = durations.float().unsqueeze(-1) / 128.0  # Assume max duration 128 bars
        duration_emb = self.duration_projection(duration_norm)  # [batch_size, style_dim]
        
        # Concatenate and project
        conditions = torch.cat([style_emb, tempo_emb, duration_emb], dim=-1)
        condition_encoding = self.condition_projection(conditions)  # [batch_size, d_model]
        
        return condition_encoding.unsqueeze(1)  # [batch_size, 1, d_model]
        
    def forward(self, 
                input_ids: torch.Tensor,
                styles: torch.Tensor,
                tempos: torch.Tensor,
                durations: torch.Tensor,
                target_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len] input token IDs
            styles: [batch_size] style IDs
            tempos: [batch_size] tempo values
            durations: [batch_size] target duration in bars
            target_ids: [batch_size, seq_len] target token IDs for teacher forcing
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_emb = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        token_emb = self.pos_encoding(token_emb.transpose(0, 1)).transpose(0, 1)
        
        # Condition encoding (acts as memory for decoder)
        condition_memory = self.encode_conditions(styles, tempos, durations)
        condition_memory = condition_memory.expand(-1, seq_len, -1)
        
        # Create masks
        tgt_padding_mask = self.create_padding_mask(input_ids)
        tgt_causal_mask = self.create_causal_mask(seq_len)
        
        # Transformer decoder
        decoder_output = self.transformer_decoder(
            tgt=token_emb,
            memory=condition_memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        # Output projection
        logits = self.output_projection(decoder_output)
        
        # Apply coverage penalty if we have target sequence
        if target_ids is not None:
            logits = self.coverage.compute_coverage_penalty(
                logits, target_ids, self.coverage_penalty
            )
            
        return logits
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with teacher forcing"""
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        styles = batch['styles']
        tempos = batch['tempos']
        durations = batch['durations']
        
        # Use teacher forcing with probability
        use_teacher_forcing = torch.rand(1).item() < self.teacher_forcing_ratio
        
        if use_teacher_forcing:
            # Teacher forcing: use target as input (shifted)
            decoder_input = target_ids[:, :-1]
            decoder_target = target_ids[:, 1:]
        else:
            # No teacher forcing: use model's own predictions
            decoder_input = input_ids
            decoder_target = target_ids[:, 1:]
            
        # Forward pass
        logits = self.forward(decoder_input, styles, tempos, durations, target_ids)
        
        # Compute loss
        loss = self.criterion(
            logits.reshape(-1, self.vocab_size),
            decoder_target.reshape(-1)
        )
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('teacher_forcing_ratio', self.teacher_forcing_ratio, on_step=True)
        
        return loss
        
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        styles = batch['styles']
        tempos = batch['tempos']
        durations = batch['durations']
        
        # Always use teacher forcing for validation
        decoder_input = target_ids[:, :-1]
        decoder_target = target_ids[:, 1:]
        
        # Forward pass
        logits = self.forward(decoder_input, styles, tempos, durations, target_ids)
        
        # Compute loss
        loss = self.criterion(
            logits.reshape(-1, self.vocab_size),
            decoder_target.reshape(-1)
        )
        
        # Compute accuracy
        predictions = logits.argmax(dim=-1)
        mask = decoder_target != self.tokenizer.pad_token_id
        accuracy = ((predictions == decoder_target) * mask).sum().float() / mask.sum().float()
        
        # Logging
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
        
    def on_train_epoch_end(self):
        """Update teacher forcing ratio at end of epoch"""
        self.teacher_forcing_ratio = max(
            self.min_teacher_forcing,
            self.teacher_forcing_ratio * self.teacher_forcing_decay
        )
        
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config['training']['weight_decay']
        )
        
        scheduler_config = self.config['training']
        if scheduler_config['scheduler'] == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **scheduler_config['lr_scheduler_params']
            )
        elif scheduler_config['scheduler'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, **scheduler_config['lr_scheduler_params']
            )
        else:
            return optimizer
            
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
        
    def generate_arrangement(self, 
                           style: str,
                           tempo: int,
                           target_duration: int,
                           max_length: int = 32,
                           temperature: float = 1.0,
                           top_k: int = 50,
                           top_p: float = 0.9) -> List[Dict]:
        """
        Generate arrangement sequence
        
        Args:
            style: Style name ('rock_punk', 'rnb_ballad', 'country_pop')
            tempo: Target BPM
            target_duration: Target duration in bars
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            
        Returns:
            List of section dictionaries
        """
        self.eval()
        
        # Encode conditions
        style_id = self.style_to_id.get(style, 0)
        styles = torch.tensor([style_id], device=self.device)
        tempos = torch.tensor([tempo], device=self.device)
        durations = torch.tensor([target_duration], device=self.device)
        
        # Start with start token
        generated = torch.tensor([[self.tokenizer.start_token_id]], device=self.device)
        
        for _ in range(max_length):
            # Forward pass
            with torch.no_grad():
                logits = self.forward(generated, styles, tempos, durations)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Check for end token
                if next_token.item() == self.tokenizer.end_token_id:
                    break
                    
        # Decode to arrangement
        token_ids = generated[0].cpu().tolist()
        arrangement = self.tokenizer.decode_arrangement(token_ids)
        
        return arrangement


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)