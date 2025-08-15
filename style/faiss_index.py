"""
FAISS index builder for hierarchical style patterns.
Builds parent indices from refs_midi/tokenized bars with child bias support.
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import faiss
import logging

logger = logging.getLogger(__name__)


@dataclass
class StylePattern:
    """Represents a tokenized musical pattern with metadata."""
    tokens: List[str]
    embedding: np.ndarray
    parent_genre: str
    child_genre: Optional[str] = None
    bar_idx: int = 0
    source_file: str = ""
    weight: float = 1.0  # Higher weight for child patterns


class HierarchicalFAISSIndex:
    """
    Hierarchical FAISS index for style-aware pattern retrieval.
    
    Structure:
    - Parent indices: Built from all tokenized bars in parent/refs_midi
    - Child patterns: Extra patterns with higher fusion weights
    - Retrieval: Parent patterns + child bias during decoding
    """
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.parent_indices: Dict[str, faiss.IndexFlatIP] = {}
        self.parent_patterns: Dict[str, List[StylePattern]] = {}
        self.child_patterns: Dict[str, List[StylePattern]] = {}
        
    def build_parent_indices(self, style_packs_dir: str = "style_packs"):
        """
        Build FAISS indices for each parent genre from tokenized MIDI bars.
        
        Args:
            style_packs_dir: Root directory containing style packs
        """
        style_packs_path = Path(style_packs_dir)
        
        for parent_dir in style_packs_path.iterdir():
            if not parent_dir.is_dir():
                continue
                
            parent_genre = parent_dir.name
            logger.info(f"Building parent index for {parent_genre}")
            
            # Load tokenized bars from parent refs_midi
            patterns = self._load_parent_patterns(parent_dir, parent_genre)
            
            if not patterns:
                logger.warning(f"No patterns found for parent {parent_genre}")
                continue
                
            # Create embeddings for patterns
            embeddings = self._create_pattern_embeddings(patterns)
            
            # Build FAISS index
            index = faiss.IndexFlatIP(self.embedding_dim)
            index.add(embeddings.astype(np.float32))
            
            self.parent_indices[parent_genre] = index
            self.parent_patterns[parent_genre] = patterns
            
            logger.info(f"Built index for {parent_genre}: {len(patterns)} patterns")
    
    def register_child_patterns(self, parent_genre: str, child_genre: str, 
                              style_packs_dir: str = "style_packs", 
                              child_weight: float = 1.5):
        """
        Register additional child patterns with higher fusion weights.
        
        Args:
            parent_genre: Parent genre name
            child_genre: Child genre name  
            style_packs_dir: Root directory containing style packs
            child_weight: Higher weight for child pattern fusion
        """
        child_path = Path(style_packs_dir) / parent_genre / child_genre
        
        if not child_path.exists():
            logger.warning(f"Child path not found: {child_path}")
            return
            
        logger.info(f"Registering child patterns: {parent_genre}/{child_genre}")
        
        # Load child patterns
        patterns = self._load_child_patterns(child_path, parent_genre, child_genre, child_weight)
        
        if not patterns:
            logger.warning(f"No child patterns found for {child_genre}")
            return
            
        # Store child patterns separately for bias application
        child_key = f"{parent_genre}/{child_genre}"
        self.child_patterns[child_key] = patterns
        
        logger.info(f"Registered {len(patterns)} child patterns for {child_key}")
    
    def _load_parent_patterns(self, parent_dir: Path, parent_genre: str) -> List[StylePattern]:
        """Load tokenized patterns from parent refs_midi directory."""
        patterns = []
        refs_midi_dir = parent_dir / "refs_midi"
        
        if not refs_midi_dir.exists():
            return patterns
            
        # Look for tokenized files (assuming .tokens or .json format)
        for token_file in refs_midi_dir.glob("*.tokens"):
            try:
                with open(token_file, 'r') as f:
                    tokens = f.read().strip().split()
                    
                pattern = StylePattern(
                    tokens=tokens,
                    embedding=np.zeros(self.embedding_dim),  # Will be computed
                    parent_genre=parent_genre,
                    source_file=str(token_file),
                    bar_idx=0
                )
                patterns.append(pattern)
                
            except Exception as e:
                logger.error(f"Error loading {token_file}: {e}")
                
        # Also check for JSON format
        for json_file in refs_midi_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                if "tokenized_bars" in data:
                    for idx, bar_tokens in enumerate(data["tokenized_bars"]):
                        pattern = StylePattern(
                            tokens=bar_tokens,
                            embedding=np.zeros(self.embedding_dim),
                            parent_genre=parent_genre,
                            source_file=str(json_file),
                            bar_idx=idx
                        )
                        patterns.append(pattern)
                        
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                
        return patterns
    
    def _load_child_patterns(self, child_dir: Path, parent_genre: str, 
                           child_genre: str, weight: float) -> List[StylePattern]:
        """Load tokenized patterns from child refs_midi directory."""
        patterns = []
        refs_midi_dir = child_dir / "refs_midi"
        
        if not refs_midi_dir.exists():
            return patterns
            
        # Load child patterns similar to parent, but with higher weight
        for token_file in refs_midi_dir.glob("*.tokens"):
            try:
                with open(token_file, 'r') as f:
                    tokens = f.read().strip().split()
                    
                pattern = StylePattern(
                    tokens=tokens,
                    embedding=np.zeros(self.embedding_dim),
                    parent_genre=parent_genre,
                    child_genre=child_genre,
                    source_file=str(token_file),
                    bar_idx=0,
                    weight=weight
                )
                patterns.append(pattern)
                
            except Exception as e:
                logger.error(f"Error loading {token_file}: {e}")
                
        for json_file in refs_midi_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                if "tokenized_bars" in data:
                    for idx, bar_tokens in enumerate(data["tokenized_bars"]):
                        pattern = StylePattern(
                            tokens=bar_tokens,
                            embedding=np.zeros(self.embedding_dim),
                            parent_genre=parent_genre,
                            child_genre=child_genre,
                            source_file=str(json_file),
                            bar_idx=idx,
                            weight=weight
                        )
                        patterns.append(pattern)
                        
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
                
        return patterns
    
    def _create_pattern_embeddings(self, patterns: List[StylePattern]) -> np.ndarray:
        """
        Create embeddings for musical patterns.
        Simple implementation using token hashing - can be replaced with learned embeddings.
        """
        embeddings = []
        
        for pattern in patterns:
            # Simple token-based embedding (replace with learned encoder)
            embedding = self._tokens_to_embedding(pattern.tokens)
            pattern.embedding = embedding
            embeddings.append(embedding)
            
        return np.array(embeddings)
    
    def _tokens_to_embedding(self, tokens: List[str]) -> np.ndarray:
        """
        Convert tokens to embedding vector.
        Simple implementation - replace with neural encoder.
        """
        # Create a simple hash-based embedding
        embedding = np.zeros(self.embedding_dim)
        
        for i, token in enumerate(tokens[:100]):  # Limit token window
            # Simple hash mapping
            token_hash = hash(token) % self.embedding_dim
            embedding[token_hash] += 1.0
            
            # Add positional information
            pos_factor = 1.0 / (1.0 + i * 0.1)
            embedding[token_hash] *= pos_factor
            
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def retrieve_similar_patterns(self, query_tokens: List[str], 
                                parent_genre: str,
                                child_genre: Optional[str] = None,
                                child_bias: float = 0.0,
                                k: int = 5) -> List[Tuple[StylePattern, float]]:
        """
        Retrieve similar patterns with optional child bias.
        
        Args:
            query_tokens: Input tokens to find similar patterns for
            parent_genre: Parent genre to search in
            child_genre: Optional child genre for bias
            child_bias: Weight boost for child patterns (0.0-1.0)
            k: Number of patterns to retrieve
            
        Returns:
            List of (pattern, similarity_score) tuples
        """
        if parent_genre not in self.parent_indices:
            logger.error(f"No index found for parent genre: {parent_genre}")
            return []
            
        # Create query embedding
        query_embedding = self._tokens_to_embedding(query_tokens)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search parent index
        parent_index = self.parent_indices[parent_genre]
        parent_patterns = self.parent_patterns[parent_genre]
        
        scores, indices = parent_index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(parent_patterns):
                pattern = parent_patterns[idx]
                results.append((pattern, float(score)))
        
        # Apply child bias if specified
        if child_genre and child_bias > 0:
            results = self._apply_child_bias(results, query_embedding, 
                                           parent_genre, child_genre, child_bias, k)
        
        return results[:k]
    
    def _apply_child_bias(self, parent_results: List[Tuple[StylePattern, float]],
                         query_embedding: np.ndarray,
                         parent_genre: str, child_genre: str,
                         child_bias: float, k: int) -> List[Tuple[StylePattern, float]]:
        """Apply child pattern bias to retrieval results."""
        child_key = f"{parent_genre}/{child_genre}"
        
        if child_key not in self.child_patterns:
            return parent_results
            
        child_patterns = self.child_patterns[child_key]
        
        # Calculate similarities with child patterns
        child_results = []
        for pattern in child_patterns:
            # Compute similarity
            similarity = np.dot(query_embedding.flatten(), pattern.embedding)
            # Apply child bias and pattern weight
            biased_score = similarity * (1.0 + child_bias) * pattern.weight
            child_results.append((pattern, biased_score))
        
        # Merge and re-rank
        all_results = parent_results + child_results
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        return all_results
    
    def save_indices(self, save_dir: str):
        """Save FAISS indices and pattern metadata to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save parent indices
        for parent_genre, index in self.parent_indices.items():
            index_file = save_path / f"{parent_genre}.faiss"
            faiss.write_index(index, str(index_file))
            
        # Save pattern metadata
        patterns_file = save_path / "patterns.pkl"
        with open(patterns_file, 'wb') as f:
            pickle.dump({
                'parent_patterns': self.parent_patterns,
                'child_patterns': self.child_patterns,
                'embedding_dim': self.embedding_dim
            }, f)
            
        logger.info(f"Saved indices to {save_dir}")
    
    def load_indices(self, load_dir: str):
        """Load FAISS indices and pattern metadata from disk."""
        load_path = Path(load_dir)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Index directory not found: {load_dir}")
            
        # Load parent indices
        for index_file in load_path.glob("*.faiss"):
            parent_genre = index_file.stem
            index = faiss.read_index(str(index_file))
            self.parent_indices[parent_genre] = index
            
        # Load pattern metadata
        patterns_file = load_path / "patterns.pkl"
        if patterns_file.exists():
            with open(patterns_file, 'rb') as f:
                data = pickle.load(f)
                self.parent_patterns = data['parent_patterns']
                self.child_patterns = data['child_patterns']
                self.embedding_dim = data['embedding_dim']
                
        logger.info(f"Loaded indices from {load_dir}")


def build_hierarchical_indices(style_packs_dir: str = "style_packs",
                             output_dir: str = "indices",
                             embedding_dim: int = 512) -> HierarchicalFAISSIndex:
    """
    Build complete hierarchical FAISS indices for all parent genres.
    
    Args:
        style_packs_dir: Directory containing style packs
        output_dir: Directory to save indices
        embedding_dim: Dimension of pattern embeddings
        
    Returns:
        Built HierarchicalFAISSIndex instance
    """
    index = HierarchicalFAISSIndex(embedding_dim)
    
    # Build parent indices
    index.build_parent_indices(style_packs_dir)
    
    # Register child patterns for all available children
    style_packs_path = Path(style_packs_dir)
    for parent_dir in style_packs_path.iterdir():
        if not parent_dir.is_dir():
            continue
            
        parent_genre = parent_dir.name
        
        # Find child directories
        for child_dir in parent_dir.iterdir():
            if child_dir.is_dir() and child_dir.name not in ['refs_audio', 'refs_midi']:
                child_genre = child_dir.name
                index.register_child_patterns(parent_genre, child_genre, 
                                             style_packs_dir, child_weight=1.5)
    
    # Save indices
    index.save_indices(output_dir)
    
    return index


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Build indices
    index = build_hierarchical_indices()
    
    print("Hierarchical FAISS indices built successfully!")
    print(f"Parent genres: {list(index.parent_indices.keys())}")
    print(f"Child patterns: {list(index.child_patterns.keys())}")