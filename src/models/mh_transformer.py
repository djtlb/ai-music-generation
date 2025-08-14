"""
Melody & Harmony Transformer

A style-conditioned transformer for generating melody and chord progressions
with constraints for musicality and style consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import math
from typing import Dict, List, Optional, Tuple, Union
import json


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence modeling"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class StyleEmbedding(nn.Module):
    """Style conditioning module"""
    
    def __init__(self, d_model: int, style_vocab_size: int = 3):
        super().__init__()
        self.style_embedding = nn.Embedding(style_vocab_size, d_model)
        self.key_embedding = nn.Embedding(24, d_model)  # 12 keys * 2 (major/minor)
        self.section_embedding = nn.Embedding(5, d_model)  # INTRO, VERSE, CHORUS, BRIDGE, OUTRO
        
        # Drum groove conditioning (optional)
        self.groove_projection = nn.Linear(32, d_model)  # 32-dim groove features
        
    def forward(self, style_ids: torch.Tensor, key_ids: torch.Tensor, 
                section_ids: torch.Tensor, groove_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            style_ids: [batch_size] - style indices (0=rock_punk, 1=rnb_ballad, 2=country_pop)
            key_ids: [batch_size] - key indices 
            section_ids: [batch_size] - section indices
            groove_features: [batch_size, 32] - optional drum groove features
        """
        style_emb = self.style_embedding(style_ids)
        key_emb = self.key_embedding(key_ids)
        section_emb = self.section_embedding(section_ids)
        
        conditioning = style_emb + key_emb + section_emb
        
        if groove_features is not None:
            groove_emb = self.groove_projection(groove_features)
            conditioning = conditioning + groove_emb
            
        return conditioning


class MelodyHarmonyTransformer(nn.Module):
    """
    Transformer decoder for melody and harmony generation with style conditioning
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        style_vocab_size: int = 3,
        chord_vocab_size: int = 60,  # Number of possible chords
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.chord_vocab_size = chord_vocab_size
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Style conditioning
        self.style_conditioning = StyleEmbedding(d_model, style_vocab_size)
        
        # Transformer decoder layers
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerDecoder(decoder_layer, num_layers)
        
        # Output projections
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Auxiliary task heads for constraints
        self.chord_compatibility_head = nn.Linear(d_model, chord_vocab_size)
        self.scale_compatibility_head = nn.Linear(d_model, 12)  # 12 chromatic notes
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
    
    def forward(
        self,
        input_ids: torch.Tensor,
        style_ids: torch.Tensor,
        key_ids: torch.Tensor,
        section_ids: torch.Tensor,
        chord_sequence: Optional[torch.Tensor] = None,
        groove_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len] - input token sequence
            style_ids: [batch_size] - style conditioning
            key_ids: [batch_size] - key signature
            section_ids: [batch_size] - section type
            chord_sequence: [batch_size, seq_len] - chord progression (optional)
            groove_features: [batch_size, 32] - drum groove features (optional)
            attention_mask: [batch_size, seq_len] - attention mask
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        token_emb = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        token_emb = self.pos_encoding(token_emb.transpose(0, 1)).transpose(0, 1)
        
        # Style conditioning
        style_conditioning = self.style_conditioning(
            style_ids, key_ids, section_ids, groove_features
        )
        
        # Broadcast style conditioning to sequence length
        style_conditioning = style_conditioning.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine token embeddings with style conditioning
        hidden_states = token_emb + style_conditioning
        
        # Create causal mask
        causal_mask = self.create_causal_mask(seq_len, device)
        
        # Transformer decoder
        memory = torch.zeros(batch_size, 1, self.d_model, device=device)  # Empty memory
        hidden_states = self.transformer(
            hidden_states,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=attention_mask
        )
        
        # Output projections
        logits = self.output_projection(hidden_states)
        
        # Auxiliary outputs for constraint losses
        chord_compat_logits = self.chord_compatibility_head(hidden_states)
        scale_compat_logits = self.scale_compatibility_head(hidden_states)
        
        return {
            'logits': logits,
            'chord_compatibility': chord_compat_logits,
            'scale_compatibility': scale_compat_logits,
            'hidden_states': hidden_states
        }
    
    def generate(
        self,
        prompt_ids: torch.Tensor,
        style_ids: torch.Tensor,
        key_ids: torch.Tensor,
        section_ids: torch.Tensor,
        max_length: int = 512,
        temperature: float = 0.8,
        nucleus_p: float = 0.9,
        chord_sequence: Optional[torch.Tensor] = None,
        groove_features: Optional[torch.Tensor] = None,
        constraint_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate melody and harmony sequence
        
        Args:
            prompt_ids: [batch_size, prompt_len] - initial sequence
            style_ids: [batch_size] - style conditioning
            key_ids: [batch_size] - key signature
            section_ids: [batch_size] - section type
            max_length: Maximum generation length
            temperature: Sampling temperature
            nucleus_p: Nucleus sampling threshold
            chord_sequence: [batch_size, max_length] - chord progression constraint
            groove_features: [batch_size, 32] - drum groove features
            constraint_mask: [batch_size, max_length, vocab_size] - constraint mask
        """
        self.eval()
        batch_size = prompt_ids.shape[0]
        device = prompt_ids.device
        
        generated = prompt_ids.clone()
        
        with torch.no_grad():
            for step in range(max_length - prompt_ids.shape[1]):
                # Forward pass
                outputs = self.forward(
                    generated,
                    style_ids,
                    key_ids,
                    section_ids,
                    chord_sequence,
                    groove_features
                )
                
                # Get next token logits
                next_token_logits = outputs['logits'][:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply constraint mask if provided
                if constraint_mask is not None:
                    current_step = generated.shape[1] - prompt_ids.shape[1]
                    if current_step < constraint_mask.shape[1]:
                        mask = constraint_mask[:, current_step, :]
                        next_token_logits = next_token_logits + mask
                
                # Nucleus sampling
                if nucleus_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > nucleus_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Apply the mask
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits.scatter_(1, indices_to_remove.unsqueeze(0), float('-inf'))
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for end-of-sequence
                if (next_token == 0).all():  # Assuming 0 is EOS token
                    break
        
        return generated


class MHTrainingLoss(nn.Module):
    """
    Training loss with auxiliary constraints for musicality
    """
    
    def __init__(
        self,
        vocab_size: int,
        chord_vocab_size: int = 60,
        scale_penalty_weight: float = 0.1,
        repetition_penalty_weight: float = 0.05,
        chord_compatibility_weight: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.chord_vocab_size = chord_vocab_size
        self.scale_penalty_weight = scale_penalty_weight
        self.repetition_penalty_weight = repetition_penalty_weight
        self.chord_compatibility_weight = chord_compatibility_weight
        
        # Main loss
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Auxiliary losses
        self.chord_compatibility_loss = nn.CrossEntropyLoss()
        self.scale_compatibility_loss = nn.BCEWithLogitsLoss()
    
    def compute_repetition_penalty(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Compute repetition penalty for generated sequences"""
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        penalty = torch.zeros_like(logits)
        
        for i in range(batch_size):
            sequence = input_ids[i]
            for pos in range(1, seq_len):
                # Look for repetitions in recent history
                window_size = min(8, pos)
                recent_tokens = sequence[pos-window_size:pos]
                
                # Count occurrences of each token
                for token_id in range(vocab_size):
                    count = (recent_tokens == token_id).sum().float()
                    if count > 0:
                        # Apply penalty proportional to repetition frequency
                        penalty[i, pos, token_id] = -count * 0.5
        
        return penalty
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target_ids: torch.Tensor,
        key_ids: torch.Tensor,
        chord_targets: Optional[torch.Tensor] = None,
        scale_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss with auxiliary constraints
        
        Args:
            outputs: Model outputs dictionary
            target_ids: [batch_size, seq_len] - target sequence
            key_ids: [batch_size] - key signature for scale penalty
            chord_targets: [batch_size, seq_len] - chord progression targets
            scale_targets: [batch_size, seq_len, 12] - scale compatibility targets
        """
        logits = outputs['logits']
        batch_size, seq_len, vocab_size = logits.shape
        
        # Main cross-entropy loss
        main_loss = self.cross_entropy(
            logits.reshape(-1, vocab_size),
            target_ids.reshape(-1)
        )
        
        total_loss = main_loss
        losses = {'main_loss': main_loss}
        
        # Repetition penalty
        if self.repetition_penalty_weight > 0:
            # Create dummy input_ids for repetition analysis (simplified)
            input_ids = torch.cat([
                torch.zeros(batch_size, 1, device=target_ids.device, dtype=target_ids.dtype),
                target_ids[:, :-1]
            ], dim=1)
            
            repetition_penalty = self.compute_repetition_penalty(input_ids, logits)
            rep_loss = -repetition_penalty.mean()  # Negative because penalty reduces probability
            
            total_loss = total_loss + self.repetition_penalty_weight * rep_loss
            losses['repetition_loss'] = rep_loss
        
        # Chord compatibility loss
        if self.chord_compatibility_weight > 0 and chord_targets is not None:
            chord_compat_logits = outputs['chord_compatibility']
            chord_loss = self.chord_compatibility_loss(
                chord_compat_logits.reshape(-1, self.chord_vocab_size),
                chord_targets.reshape(-1)
            )
            total_loss = total_loss + self.chord_compatibility_weight * chord_loss
            losses['chord_compatibility_loss'] = chord_loss
        
        # Scale compatibility loss
        if self.scale_penalty_weight > 0 and scale_targets is not None:
            scale_compat_logits = outputs['scale_compatibility']
            scale_loss = self.scale_compatibility_loss(
                scale_compat_logits.reshape(-1, 12),
                scale_targets.reshape(-1, 12)
            )
            total_loss = total_loss + self.scale_penalty_weight * scale_loss
            losses['scale_compatibility_loss'] = scale_loss
        
        losses['total_loss'] = total_loss
        return losses


def load_model_config(config_path: str) -> Dict:
    """Load model configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_mh_transformer(config: Dict) -> MelodyHarmonyTransformer:
    """Create MH Transformer from configuration"""
    return MelodyHarmonyTransformer(**config)


if __name__ == "__main__":
    # Test model creation
    model = MelodyHarmonyTransformer(
        vocab_size=1000,
        d_model=256,
        nhead=8,
        num_layers=4,
        style_vocab_size=3
    )
    
    # Test forward pass
    batch_size = 2
    seq_len = 64
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    style_ids = torch.randint(0, 3, (batch_size,))
    key_ids = torch.randint(0, 24, (batch_size,))
    section_ids = torch.randint(0, 5, (batch_size,))
    
    outputs = model(input_ids, style_ids, key_ids, section_ids)
    
    print(f"Model created successfully!")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Chord compatibility shape: {outputs['chord_compatibility'].shape}")
    print(f"Scale compatibility shape: {outputs['scale_compatibility'].shape}")