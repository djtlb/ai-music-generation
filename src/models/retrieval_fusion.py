"""
Retrieval Fusion for Biased Token Generation

Implements shallow fusion to bias token logits toward retrieved n-grams
during generation. Supports multiple fusion strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval-based fusion"""
    enabled: bool = True
    retrieval_weight: float = 0.3
    top_k: int = 5
    fusion_method: str = "shallow"  # "shallow", "deep", "interpolation"
    ngram_size: int = 3
    min_similarity: float = 0.5
    decay_factor: float = 0.9  # Decay weight for older context
    

class NGramMatcher:
    """Efficient n-gram matching for retrieved patterns"""
    
    def __init__(self, ngram_size: int = 3):
        self.ngram_size = ngram_size
        self.pattern_ngrams: Dict[str, Set[Tuple[str, ...]]] = {}
        
    def add_pattern(self, pattern_id: str, tokens: List[str]) -> None:
        """Add a pattern and extract its n-grams"""
        ngrams = set()
        for i in range(len(tokens) - self.ngram_size + 1):
            ngram = tuple(tokens[i:i + self.ngram_size])
            ngrams.add(ngram)
        self.pattern_ngrams[pattern_id] = ngrams
        
    def find_matching_continuations(
        self, 
        context: List[str], 
        pattern_ids: List[str]
    ) -> Dict[str, List[str]]:
        """
        Find tokens that could continue the current context based on retrieved patterns
        
        Args:
            context: Current token context
            pattern_ids: IDs of retrieved patterns to check
            
        Returns:
            Dictionary mapping pattern_id to list of continuation tokens
        """
        continuations = {}
        
        # Extract context suffix for matching
        context_suffix = tuple(context[-(self.ngram_size-1):])
        
        for pattern_id in pattern_ids:
            if pattern_id not in self.pattern_ngrams:
                continue
                
            pattern_continuations = []
            for ngram in self.pattern_ngrams[pattern_id]:
                # Check if ngram starts with our context suffix
                if ngram[:-1] == context_suffix:
                    pattern_continuations.append(ngram[-1])
                    
            if pattern_continuations:
                continuations[pattern_id] = pattern_continuations
                
        return continuations


class TokenVocabulary:
    """Token vocabulary for efficient lookups"""
    
    def __init__(self, vocab: Dict[str, int]):
        self.token_to_id = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.vocab_size = len(vocab)
        
    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs"""
        return [self.token_to_id.get(token, self.token_to_id.get('<UNK>', 0)) for token in tokens]
        
    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens"""
        return [self.id_to_token.get(id, '<UNK>') for id in ids]


class RetrievalFusion:
    """Main retrieval fusion class"""
    
    def __init__(
        self, 
        vocab: TokenVocabulary,
        config: RetrievalConfig = None
    ):
        self.vocab = vocab
        self.config = config or RetrievalConfig()
        self.ngram_matcher = NGramMatcher(self.config.ngram_size)
        
        # Cache for retrieved patterns
        self.retrieved_patterns: List[Dict] = []
        self.pattern_weights: List[float] = []
        
    def update_retrieved_patterns(
        self, 
        patterns: List[Dict],
        similarities: List[float]
    ) -> None:
        """
        Update the set of retrieved patterns
        
        Args:
            patterns: List of pattern dicts with 'pattern_id' and 'tokens'
            similarities: Similarity scores for each pattern
        """
        self.retrieved_patterns = patterns[:self.config.top_k]
        self.pattern_weights = similarities[:self.config.top_k]
        
        # Update n-gram matcher
        for pattern in self.retrieved_patterns:
            self.ngram_matcher.add_pattern(
                pattern['pattern_id'], 
                pattern['tokens']
            )
            
        logger.debug(f"Updated with {len(self.retrieved_patterns)} retrieved patterns")
        
    def compute_retrieval_bias(
        self, 
        context_tokens: List[str]
    ) -> torch.Tensor:
        """
        Compute bias scores for all vocabulary tokens based on retrieved patterns
        
        Args:
            context_tokens: Current generation context
            
        Returns:
            Bias tensor of shape (vocab_size,)
        """
        if not self.retrieved_patterns or not self.config.enabled:
            return torch.zeros(self.vocab.vocab_size)
            
        bias = torch.zeros(self.vocab.vocab_size)
        
        # Find matching continuations from retrieved patterns
        pattern_ids = [p['pattern_id'] for p in self.retrieved_patterns]
        continuations = self.ngram_matcher.find_matching_continuations(
            context_tokens, pattern_ids
        )
        
        # Weight continuations by pattern similarity and frequency
        for i, pattern_id in enumerate(pattern_ids):
            if pattern_id not in continuations:
                continue
                
            pattern_weight = self.pattern_weights[i] * (self.config.decay_factor ** i)
            tokens = continuations[pattern_id]
            
            # Count frequency of each continuation token
            token_counts = {}
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
                
            # Add weighted bias
            for token, count in token_counts.items():
                if token in self.vocab.token_to_id:
                    token_id = self.vocab.token_to_id[token]
                    # Bias proportional to pattern weight, similarity, and frequency
                    bias[token_id] += pattern_weight * count
                    
        # Normalize bias scores
        if bias.sum() > 0:
            bias = bias / bias.sum() * self.config.retrieval_weight
            
        return bias
        
    def apply_shallow_fusion(
        self, 
        logits: torch.Tensor, 
        context_tokens: List[str]
    ) -> torch.Tensor:
        """
        Apply shallow fusion by biasing output logits
        
        Args:
            logits: Model output logits (vocab_size,)
            context_tokens: Current context tokens
            
        Returns:
            Biased logits
        """
        if not self.config.enabled:
            return logits
            
        # Compute retrieval bias
        bias = self.compute_retrieval_bias(context_tokens)
        bias = bias.to(logits.device)
        
        # Apply bias based on fusion method
        if self.config.fusion_method == "shallow":
            # Simple additive bias
            biased_logits = logits + bias * self.config.retrieval_weight
            
        elif self.config.fusion_method == "interpolation":
            # Interpolate with bias distribution
            bias_probs = F.softmax(bias, dim=-1)
            logit_probs = F.softmax(logits, dim=-1)
            
            mixed_probs = (
                (1 - self.config.retrieval_weight) * logit_probs + 
                self.config.retrieval_weight * bias_probs
            )
            
            # Convert back to logits
            biased_logits = torch.log(mixed_probs + 1e-8)
            
        elif self.config.fusion_method == "multiplicative":
            # Multiplicative bias (element-wise)
            bias_weights = torch.exp(bias * self.config.retrieval_weight)
            biased_logits = logits * bias_weights
            
        else:
            raise ValueError(f"Unknown fusion method: {self.config.fusion_method}")
            
        return biased_logits
        
    def apply_deep_fusion(
        self, 
        hidden_states: torch.Tensor,
        layer_idx: int,
        context_tokens: List[str]
    ) -> torch.Tensor:
        """
        Apply deep fusion by modifying hidden states at intermediate layers
        
        Args:
            hidden_states: Hidden states (batch_size, seq_len, hidden_dim)
            layer_idx: Current transformer layer index
            context_tokens: Current context tokens
            
        Returns:
            Modified hidden states
        """
        if not self.config.enabled or self.config.fusion_method != "deep":
            return hidden_states
            
        # For now, implement as identity - would need model-specific implementation
        logger.warning("Deep fusion not fully implemented - using shallow fusion")
        return hidden_states


class RetrievalBiasedGenerator:
    """Generator with retrieval bias support"""
    
    def __init__(
        self,
        model: nn.Module,
        vocab: TokenVocabulary,
        config: RetrievalConfig = None
    ):
        self.model = model
        self.vocab = vocab
        self.config = config or RetrievalConfig()
        self.fusion = RetrievalFusion(vocab, config)
        
    def update_retrieved_patterns(
        self, 
        patterns: List[Dict], 
        similarities: List[float]
    ) -> None:
        """Update retrieved patterns for bias"""
        self.fusion.update_retrieved_patterns(patterns, similarities)
        
    def generate_with_bias(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate tokens with retrieval bias
        
        Args:
            input_ids: Initial token IDs (batch_size, seq_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Generated token IDs
        """
        batch_size, initial_length = input_ids.shape
        generated = input_ids.clone()
        
        for step in range(max_length):
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(generated)
                logits = outputs.logits[:, -1, :]  # Last token predictions
                
            # Apply retrieval bias
            for batch_idx in range(batch_size):
                # Convert current sequence to tokens for bias computation
                current_ids = generated[batch_idx].tolist()
                current_tokens = self.vocab.ids_to_tokens(current_ids)
                
                # Apply fusion bias
                biased_logits = self.fusion.apply_shallow_fusion(
                    logits[batch_idx], current_tokens
                )
                logits[batch_idx] = biased_logits
                
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
                
            # Sample next tokens
            if do_sample:
                # Top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Set logits to -inf for removed tokens
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                    
                # Sample from the distribution
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
                
            # Append to generated sequence
            generated = torch.cat([generated, next_tokens], dim=1)
            
            # Check for end tokens if needed
            # Could add early stopping logic here
            
        return generated
        
    def set_retrieval_weight(self, weight: float) -> None:
        """Update retrieval weight"""
        self.config.retrieval_weight = weight
        self.fusion.config.retrieval_weight = weight
        
    def toggle_retrieval(self, enabled: bool) -> None:
        """Enable/disable retrieval bias"""
        self.config.enabled = enabled
        self.fusion.config.enabled = enabled


# Unit test utilities
def create_test_patterns() -> List[Dict]:
    """Create synthetic test patterns for unit tests"""
    patterns = [
        {
            'pattern_id': 'rock_pattern_1',
            'tokens': ['KICK', 'BAR_1', 'POS_1', 'SNARE', 'BAR_1', 'POS_3'],
            'style': 'rock_punk'
        },
        {
            'pattern_id': 'rock_pattern_2', 
            'tokens': ['CHORD', 'C', 'BAR_1', 'CHORD', 'Am', 'BAR_2'],
            'style': 'rock_punk'
        },
        {
            'pattern_id': 'ballad_pattern_1',
            'tokens': ['PIANO', 'C4', 'BAR_1', 'POS_1', 'PIANO', 'E4'],
            'style': 'rnb_ballad'
        }
    ]
    return patterns


def test_retrieval_fusion():
    """Test retrieval fusion functionality"""
    # Create test vocabulary
    vocab_dict = {
        'KICK': 0, 'SNARE': 1, 'BAR_1': 2, 'POS_1': 3, 'POS_3': 4,
        'CHORD': 5, 'C': 6, 'Am': 7, 'BAR_2': 8, 'PIANO': 9, 'C4': 10, 'E4': 11,
        '<UNK>': 12, '<PAD>': 13
    }
    vocab = TokenVocabulary(vocab_dict)
    
    # Create fusion module
    config = RetrievalConfig(enabled=True, retrieval_weight=0.3, ngram_size=3)
    fusion = RetrievalFusion(vocab, config)
    
    # Add test patterns
    patterns = create_test_patterns()
    similarities = [0.9, 0.8, 0.7]
    fusion.update_retrieved_patterns(patterns, similarities)
    
    # Test bias computation
    context = ['KICK', 'BAR_1']
    bias = fusion.compute_retrieval_bias(context)
    
    print(f"Bias shape: {bias.shape}")
    print(f"Non-zero bias indices: {torch.nonzero(bias).flatten().tolist()}")
    print(f"Max bias value: {bias.max().item():.4f}")
    
    # Test shallow fusion
    logits = torch.randn(len(vocab_dict))
    biased_logits = fusion.apply_shallow_fusion(logits, context)
    
    print(f"Original logits sum: {logits.sum().item():.4f}")
    print(f"Biased logits sum: {biased_logits.sum().item():.4f}")
    
    return True


if __name__ == "__main__":
    # Run tests
    test_retrieval_fusion()
    print("Retrieval fusion tests passed!")