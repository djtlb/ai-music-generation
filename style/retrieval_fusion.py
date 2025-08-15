"""
Retrieval fusion for biasing token generation with hierarchical style patterns.
Integrates with transformer decoding to apply parent + child pattern bias.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import logging

from .faiss_index import HierarchicalFAISSIndex, StylePattern

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for retrieval-based fusion during decoding."""
    family_index: str  # Parent genre (e.g., "pop")
    child_bias: float = 0.0  # Bias weight for child patterns (0.0-1.0)
    child_genre: Optional[str] = None  # Specific child genre
    fusion_weight: float = 0.1  # How much to weight retrieval vs. model
    ngram_size: int = 3  # Size of n-grams to match
    top_k_patterns: int = 5  # Number of patterns to retrieve
    temperature_scale: float = 1.0  # Temperature scaling for fusion


class RetrievalFusion:
    """
    Applies retrieval bias during token generation using hierarchical style patterns.
    
    Workflow:
    1. Extract recent n-gram from generated sequence
    2. Retrieve similar patterns from parent index + child bias
    3. Extract next-token distributions from retrieved patterns
    4. Fuse with model logits using weighted interpolation
    """
    
    def __init__(self, faiss_index: HierarchicalFAISSIndex, 
                 vocab: Dict[str, int], config: RetrievalConfig):
        self.faiss_index = faiss_index
        self.vocab = vocab
        self.config = config
        self.token_to_id = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        
        # Cache for pattern n-grams
        self._pattern_ngrams_cache = {}
        self._build_pattern_ngrams()
        
    def _build_pattern_ngrams(self):
        """Pre-compute n-grams from all patterns for efficient lookup."""
        logger.info("Building pattern n-gram cache...")
        
        # Build for parent patterns
        if self.config.family_index in self.faiss_index.parent_patterns:
            patterns = self.faiss_index.parent_patterns[self.config.family_index]
            self._cache_pattern_ngrams(patterns, "parent")
            
        # Build for child patterns if specified
        if self.config.child_genre:
            child_key = f"{self.config.family_index}/{self.config.child_genre}"
            if child_key in self.faiss_index.child_patterns:
                patterns = self.faiss_index.child_patterns[child_key]
                self._cache_pattern_ngrams(patterns, "child")
                
        logger.info(f"Cached n-grams for {len(self._pattern_ngrams_cache)} pattern groups")
    
    def _cache_pattern_ngrams(self, patterns: List[StylePattern], pattern_type: str):
        """Extract and cache n-grams from patterns."""
        ngrams = {}
        
        for pattern in patterns:
            tokens = pattern.tokens
            
            # Extract n-grams
            for i in range(len(tokens) - self.config.ngram_size + 1):
                ngram = tuple(tokens[i:i + self.config.ngram_size])
                next_token = tokens[i + self.config.ngram_size] if i + self.config.ngram_size < len(tokens) else None
                
                if next_token and next_token in self.token_to_id:
                    if ngram not in ngrams:
                        ngrams[ngram] = []
                    
                    # Weight by pattern weight for child patterns
                    weight = pattern.weight if hasattr(pattern, 'weight') else 1.0
                    ngrams[ngram].append((self.token_to_id[next_token], weight))
        
        self._pattern_ngrams_cache[pattern_type] = ngrams
    
    def apply_retrieval_bias(self, logits: torch.Tensor, 
                           generated_tokens: List[int],
                           **kwargs) -> torch.Tensor:
        """
        Apply retrieval bias to model logits during generation.
        
        Args:
            logits: Model logits for next token [vocab_size]
            generated_tokens: Previously generated token IDs
            
        Returns:
            Biased logits incorporating retrieval patterns
        """
        if len(generated_tokens) < self.config.ngram_size:
            return logits
            
        # Extract recent n-gram
        recent_ngram = tuple([self.id_to_token.get(tid, "<unk>") 
                             for tid in generated_tokens[-self.config.ngram_size:]])
        
        # Get retrieval bias distribution
        retrieval_dist = self._get_retrieval_distribution(recent_ngram)
        
        if retrieval_dist is None:
            return logits
            
        # Convert to tensor
        device = logits.device
        retrieval_tensor = torch.zeros_like(logits)
        
        for token_id, prob in retrieval_dist.items():
            if token_id < len(retrieval_tensor):
                retrieval_tensor[token_id] = prob
                
        # Apply temperature scaling
        retrieval_tensor = retrieval_tensor / self.config.temperature_scale
        
        # Weighted fusion
        fusion_weight = self.config.fusion_weight
        fused_logits = (1.0 - fusion_weight) * logits + fusion_weight * retrieval_tensor
        
        return fused_logits
    
    def _get_retrieval_distribution(self, ngram: Tuple[str, ...]) -> Optional[Dict[int, float]]:
        """
        Get next-token distribution from retrieved patterns for given n-gram.
        
        Args:
            ngram: Recent n-gram tokens
            
        Returns:
            Dictionary mapping token IDs to probabilities
        """
        token_counts = {}
        total_weight = 0.0
        
        # Check parent patterns
        if "parent" in self._pattern_ngrams_cache:
            parent_ngrams = self._pattern_ngrams_cache["parent"]
            if ngram in parent_ngrams:
                for token_id, weight in parent_ngrams[ngram]:
                    token_counts[token_id] = token_counts.get(token_id, 0.0) + weight
                    total_weight += weight
        
        # Check child patterns with bias
        if "child" in self._pattern_ngrams_cache and self.config.child_bias > 0:
            child_ngrams = self._pattern_ngrams_cache["child"]
            if ngram in child_ngrams:
                child_bias_multiplier = 1.0 + self.config.child_bias
                for token_id, weight in child_ngrams[ngram]:
                    biased_weight = weight * child_bias_multiplier
                    token_counts[token_id] = token_counts.get(token_id, 0.0) + biased_weight
                    total_weight += biased_weight
        
        if total_weight == 0:
            return None
            
        # Normalize to probabilities
        distribution = {token_id: count / total_weight 
                       for token_id, count in token_counts.items()}
        
        return distribution
    
    def get_similar_patterns_context(self, query_tokens: List[str], 
                                   k: int = None) -> List[StylePattern]:
        """
        Get similar patterns for broader context (not just n-gram matching).
        
        Args:
            query_tokens: Input tokens to find patterns for
            k: Number of patterns to retrieve (defaults to config)
            
        Returns:
            List of similar patterns with child bias applied
        """
        if k is None:
            k = self.config.top_k_patterns
            
        patterns_with_scores = self.faiss_index.retrieve_similar_patterns(
            query_tokens=query_tokens,
            parent_genre=self.config.family_index,
            child_genre=self.config.child_genre,
            child_bias=self.config.child_bias,
            k=k
        )
        
        return [pattern for pattern, score in patterns_with_scores]


class RetrievalAugmentedSampler:
    """
    Sampling wrapper that integrates retrieval fusion into generation.
    """
    
    def __init__(self, model: Any, tokenizer: Any,
                 retrieval_fusion: RetrievalFusion):
        self.model = model
        self.tokenizer = tokenizer
        self.retrieval_fusion = retrieval_fusion
        
    def generate_with_retrieval(self, prompt_tokens: List[int],
                              max_length: int = 512,
                              temperature: float = 1.0,
                              top_p: float = 0.9,
                              **generation_kwargs) -> List[int]:
        """
        Generate tokens with retrieval-augmented sampling.
        
        Args:
            prompt_tokens: Initial prompt token IDs
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated token sequence
        """
        generated = prompt_tokens.copy()
        
        for _ in range(max_length - len(prompt_tokens)):
            # Get model logits
            input_ids = torch.tensor([generated]).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1]  # Last token logits
            
            # Apply retrieval bias
            biased_logits = self.retrieval_fusion.apply_retrieval_bias(
                logits, generated
            )
            
            # Apply temperature and sampling
            if temperature > 0:
                biased_logits = biased_logits / temperature
                
            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(biased_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                biased_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(biased_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            generated.append(next_token)
            
            # Check for end token
            if self.tokenizer and hasattr(self.tokenizer, 'eos_token_id'):
                if next_token == self.tokenizer.eos_token_id:
                    break
        
        return generated


def create_retrieval_fusion(faiss_index_dir: str,
                          vocab_file: str,
                          family_index: str,
                          child_bias: float = 0.0,
                          child_genre: Optional[str] = None,
                          **config_kwargs) -> RetrievalFusion:
    """
    Factory function to create RetrievalFusion with loaded indices.
    
    Args:
        faiss_index_dir: Directory containing saved FAISS indices
        vocab_file: Path to vocabulary JSON file
        family_index: Parent genre name
        child_bias: Child pattern bias weight
        child_genre: Optional child genre name
        **config_kwargs: Additional RetrievalConfig parameters
        
    Returns:
        Configured RetrievalFusion instance
    """
    import json
    
    # Load FAISS index
    faiss_index = HierarchicalFAISSIndex()
    faiss_index.load_indices(faiss_index_dir)
    
    # Load vocabulary
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    
    # Create config
    config = RetrievalConfig(
        family_index=family_index,
        child_bias=child_bias,
        child_genre=child_genre,
        **config_kwargs
    )
    
    return RetrievalFusion(faiss_index, vocab, config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would be called during model inference
    print("RetrievalFusion ready for integration with generation pipeline!")