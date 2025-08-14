"""
FAISS Index for Style-based Pattern Retrieval

Builds and queries a FAISS index of musical patterns organized by style.
Supports similarity search for retrieval-based generation bias.
"""

import numpy as np
import torch
import pickle
import json
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu")

logger = logging.getLogger(__name__)


class PatternEmbedding:
    """Represents a musical pattern with its embedding and metadata"""
    
    def __init__(
        self,
        pattern_id: str,
        pattern_tokens: List[str],
        embedding: np.ndarray,
        style: str,
        bars: int,
        metadata: Optional[Dict] = None
    ):
        self.pattern_id = pattern_id
        self.pattern_tokens = pattern_tokens
        self.embedding = embedding.astype(np.float32)
        self.style = style
        self.bars = bars
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict:
        """Convert to serializable dictionary"""
        return {
            'pattern_id': self.pattern_id,
            'pattern_tokens': self.pattern_tokens,
            'embedding': self.embedding.tolist(),
            'style': self.style,
            'bars': self.bars,
            'metadata': self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'PatternEmbedding':
        """Create from dictionary"""
        return cls(
            pattern_id=data['pattern_id'],
            pattern_tokens=data['pattern_tokens'],
            embedding=np.array(data['embedding'], dtype=np.float32),
            style=data['style'],
            bars=data['bars'],
            metadata=data.get('metadata', {})
        )


class StylePatternIndex:
    """FAISS-based index for style-conditioned pattern retrieval"""
    
    def __init__(
        self,
        embedding_dim: int = 512,
        index_type: str = "IVF",
        nlist: int = 100,
        nprobe: int = 10
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        
        # Style-specific indices
        self.indices: Dict[str, faiss.Index] = {}
        self.patterns: Dict[str, List[PatternEmbedding]] = {}
        self.styles = ["rock_punk", "rnb_ballad", "country_pop"]
        
        # Initialize indices for each style
        for style in self.styles:
            self.indices[style] = self._create_index()
            self.patterns[style] = []
            
        self.is_trained = {style: False for style in self.styles}
        
    def _create_index(self) -> faiss.Index:
        """Create a FAISS index based on configuration"""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required but not available")
            
        if self.index_type == "Flat":
            # Exact search (slower but accurate)
            index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product
        elif self.index_type == "IVF":
            # Approximate search with inverted file
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist)
        elif self.index_type == "HNSW":
            # Hierarchical navigable small world
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
            
        return index
        
    def add_pattern(self, pattern: PatternEmbedding) -> None:
        """Add a pattern to the appropriate style index"""
        style = pattern.style
        if style not in self.styles:
            logger.warning(f"Unknown style: {style}")
            return
            
        # Store pattern metadata
        self.patterns[style].append(pattern)
        
        # Add to FAISS index (will train if needed)
        if self.index_type != "Flat" and not self.is_trained[style]:
            # Need enough patterns to train IVF index
            if len(self.patterns[style]) >= self.nlist:
                self._train_index(style)
        
        if self.is_trained[style] or self.index_type == "Flat":
            # Normalize embedding for cosine similarity
            embedding = pattern.embedding.copy()
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            self.indices[style].add(embedding.reshape(1, -1))
            
    def _train_index(self, style: str) -> None:
        """Train the index for a specific style"""
        if self.index_type == "Flat":
            self.is_trained[style] = True
            return
            
        patterns = self.patterns[style]
        if len(patterns) < self.nlist:
            logger.warning(f"Not enough patterns to train {style} index: {len(patterns)} < {self.nlist}")
            return
            
        # Prepare training data
        embeddings = np.stack([p.embedding for p in patterns])
        # Normalize for cosine similarity
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Train the index
        self.indices[style].train(embeddings)
        self.is_trained[style] = True
        
        # Add all existing patterns
        self.indices[style].add(embeddings)
        
        logger.info(f"Trained {style} index with {len(patterns)} patterns")
        
    def search(
        self,
        query_embedding: np.ndarray,
        style: str,
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[Tuple[PatternEmbedding, float]]:
        """
        Search for similar patterns in a specific style
        
        Args:
            query_embedding: Query vector
            style: Target style to search in
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (pattern, similarity_score) tuples
        """
        if style not in self.styles:
            logger.warning(f"Unknown style: {style}")
            return []
            
        if not self.patterns[style]:
            logger.warning(f"No patterns available for style: {style}")
            return []
            
        index = self.indices[style]
        patterns = self.patterns[style]
        
        if not self.is_trained[style] and self.index_type != "Flat":
            # Fallback to linear search
            return self._linear_search(query_embedding, patterns, top_k, min_similarity)
            
        # Normalize query for cosine similarity
        query = query_embedding.copy()
        query = query / (np.linalg.norm(query) + 1e-8)
        
        # Set nprobe for IVF indices
        if hasattr(index, 'nprobe'):
            index.nprobe = self.nprobe
            
        # Search
        similarities, indices = index.search(query.reshape(1, -1), min(top_k, len(patterns)))
        
        # Filter and return results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= 0 and sim >= min_similarity:
                results.append((patterns[idx], float(sim)))
                
        return results
        
    def _linear_search(
        self,
        query_embedding: np.ndarray,
        patterns: List[PatternEmbedding],
        top_k: int,
        min_similarity: float
    ) -> List[Tuple[PatternEmbedding, float]]:
        """Fallback linear search when index is not trained"""
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        similarities = []
        for pattern in patterns:
            pattern_norm = pattern.embedding / (np.linalg.norm(pattern.embedding) + 1e-8)
            sim = np.dot(query_norm, pattern_norm)
            if sim >= min_similarity:
                similarities.append((pattern, float(sim)))
                
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
        
    def multi_style_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> Dict[str, List[Tuple[PatternEmbedding, float]]]:
        """Search across all styles"""
        results = {}
        for style in self.styles:
            results[style] = self.search(
                query_embedding, style, top_k, min_similarity
            )
        return results
        
    def get_style_statistics(self) -> Dict[str, Dict]:
        """Get statistics about patterns per style"""
        stats = {}
        for style in self.styles:
            patterns = self.patterns[style]
            stats[style] = {
                'count': len(patterns),
                'trained': self.is_trained[style],
                'avg_bars': np.mean([p.bars for p in patterns]) if patterns else 0,
                'token_diversity': len(set(
                    token for p in patterns for token in p.pattern_tokens
                )) if patterns else 0
            }
        return stats
        
    def save(self, save_dir: Union[str, Path]) -> None:
        """Save index and patterns to disk"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS indices
        for style in self.styles:
            index_path = save_dir / f"{style}_index.faiss"
            faiss.write_index(self.indices[style], str(index_path))
            
        # Save patterns metadata
        patterns_data = {
            style: [p.to_dict() for p in patterns]
            for style, patterns in self.patterns.items()
        }
        
        patterns_path = save_dir / "patterns.json"
        with open(patterns_path, 'w') as f:
            json.dump(patterns_data, f, indent=2)
            
        # Save index metadata
        metadata = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'nlist': self.nlist,
            'nprobe': self.nprobe,
            'styles': self.styles,
            'is_trained': self.is_trained
        }
        
        metadata_path = save_dir / "index_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved style pattern index to {save_dir}")
        
    @classmethod
    def load(cls, load_dir: Union[str, Path]) -> 'StylePatternIndex':
        """Load index and patterns from disk"""
        load_dir = Path(load_dir)
        
        # Load metadata
        metadata_path = load_dir / "index_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Create instance
        index = cls(
            embedding_dim=metadata['embedding_dim'],
            index_type=metadata['index_type'],
            nlist=metadata['nlist'],
            nprobe=metadata['nprobe']
        )
        
        index.styles = metadata['styles']
        index.is_trained = metadata['is_trained']
        
        # Load FAISS indices
        for style in index.styles:
            index_path = load_dir / f"{style}_index.faiss"
            if index_path.exists():
                index.indices[style] = faiss.read_index(str(index_path))
                
        # Load patterns
        patterns_path = load_dir / "patterns.json"
        with open(patterns_path, 'r') as f:
            patterns_data = json.load(f)
            
        for style, pattern_list in patterns_data.items():
            index.patterns[style] = [
                PatternEmbedding.from_dict(p) for p in pattern_list
            ]
            
        logger.info(f"Loaded style pattern index from {load_dir}")
        return index


# Utility functions for pattern extraction
def extract_ngrams(tokens: List[str], n: int = 3) -> List[List[str]]:
    """Extract n-grams from token sequence"""
    if len(tokens) < n:
        return [tokens]
    return [tokens[i:i+n] for i in range(len(tokens) - n + 1)]


def tokenize_pattern(pattern_str: str) -> List[str]:
    """Convert pattern string to token list"""
    # Simple tokenization - can be enhanced
    tokens = pattern_str.replace('|', ' | ').split()
    return [t.strip() for t in tokens if t.strip()]


def embed_pattern_tokens(
    tokens: List[str], 
    token_embeddings: Dict[str, np.ndarray],
    aggregation: str = "mean"
) -> np.ndarray:
    """
    Embed pattern tokens using token embeddings
    
    Args:
        tokens: List of pattern tokens
        token_embeddings: Mapping from tokens to embeddings
        aggregation: How to combine token embeddings ("mean", "sum", "max")
        
    Returns:
        Pattern embedding vector
    """
    embeddings = []
    for token in tokens:
        if token in token_embeddings:
            embeddings.append(token_embeddings[token])
        else:
            # Use random embedding for unknown tokens
            embeddings.append(np.random.randn(token_embeddings[list(token_embeddings.keys())[0]].shape[0]))
            
    if not embeddings:
        # Fallback to random embedding
        dim = token_embeddings[list(token_embeddings.keys())[0]].shape[0]
        return np.random.randn(dim)
        
    embeddings = np.stack(embeddings)
    
    if aggregation == "mean":
        return np.mean(embeddings, axis=0)
    elif aggregation == "sum":
        return np.sum(embeddings, axis=0)
    elif aggregation == "max":
        return np.max(embeddings, axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


if __name__ == "__main__":
    # Example usage
    if FAISS_AVAILABLE:
        # Create index
        index = StylePatternIndex(embedding_dim=128)
        
        # Add some example patterns
        for i in range(50):
            style = np.random.choice(["rock_punk", "rnb_ballad", "country_pop"])
            embedding = np.random.randn(128).astype(np.float32)
            tokens = [f"TOKEN_{j}" for j in range(5)]
            
            pattern = PatternEmbedding(
                pattern_id=f"pattern_{i}",
                pattern_tokens=tokens,
                embedding=embedding,
                style=style,
                bars=2
            )
            
            index.add_pattern(pattern)
            
        # Search for similar patterns
        query = np.random.randn(128).astype(np.float32)
        results = index.search(query, "rock_punk", top_k=3)
        
        print(f"Found {len(results)} similar patterns")
        for pattern, similarity in results:
            print(f"  {pattern.pattern_id}: {similarity:.3f}")
            
        # Print statistics
        stats = index.get_style_statistics()
        print("\nStyle statistics:")
        for style, stat in stats.items():
            print(f"  {style}: {stat}")
    else:
        print("FAISS not available - skipping example")