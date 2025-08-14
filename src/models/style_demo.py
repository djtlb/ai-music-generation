"""
Style Embeddings + Retrieval Demo

Demonstrates the complete pipeline:
1. Audio encoding to style vectors
2. Building FAISS index with musical patterns
3. Retrieval-biased token generation

This script can be run independently to test the system.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from style_encoder import StyleEncoder, create_style_encoder
from retrieval_fusion import RetrievalFusion, RetrievalConfig, TokenVocabulary, RetrievalBiasedGenerator
from tokenizer import MIDITokenizer, load_vocab  # Assuming we have this from previous implementation

try:
    from style_index import StylePatternIndex, PatternEmbedding
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available - using mock index")


class MockStyleIndex:
    """Mock index when FAISS is not available"""
    
    def __init__(self):
        self.patterns = []
        
    def add_pattern(self, pattern):
        self.patterns.append(pattern)
        
    def search(self, query, style, top_k=5, min_similarity=0.5):
        # Return random patterns for demo
        import random
        n_results = min(top_k, len(self.patterns))
        if n_results == 0:
            return []
        
        selected = random.sample(self.patterns, n_results)
        return [(p, random.uniform(0.6, 0.95)) for p in selected]


def create_demo_patterns() -> List[Dict]:
    """Create demonstration musical patterns for different styles"""
    patterns = {
        "rock_punk": [
            "STYLE=rock_punk TEMPO=140 KEY=C SECTION=VERSE KICK BAR_1 POS_1 SNARE BAR_1 POS_3 BASS_PICK C2 BAR_1 POS_1",
            "CHORD C BAR_1 CHORD Am BAR_2 CHORD F BAR_3 CHORD G BAR_4 ACOUSTIC_STRUM Em BAR_2 POS_1",
            "NOTE_ON E4 VEL_90 DUR_8 BAR_1 POS_1 NOTE_ON G4 VEL_85 DUR_4 BAR_1 POS_3 KICK BAR_2 POS_1",
            "KICK BAR_1 POS_1 KICK BAR_1 POS_2 SNARE BAR_1 POS_3 BASS_PICK C2 BAR_1 POS_1 BASS_PICK G2 BAR_1 POS_3",
        ],
        "rnb_ballad": [
            "STYLE=rnb_ballad TEMPO=70 KEY=Bb SECTION=VERSE PIANO Bb4 BAR_1 POS_1 PIANO D5 BAR_1 POS_2",
            "CHORD Bb BAR_1 CHORD Gm BAR_2 CHORD Eb BAR_3 CHORD F BAR_4 PIANO Bb3 BAR_1 POS_1",
            "NOTE_ON Bb4 VEL_60 DUR_16 BAR_1 POS_1 NOTE_ON D5 VEL_65 DUR_8 BAR_1 POS_2 SNARE BAR_2 POS_3",
            "PIANO Bb3 BAR_1 POS_1 PIANO D4 BAR_1 POS_2 PIANO F4 BAR_1 POS_3 CHORD Bb BAR_1",
        ],
        "country_pop": [
            "STYLE=country_pop TEMPO=120 KEY=G SECTION=CHORUS ACOUSTIC_STRUM G BAR_1 POS_1 ACOUSTIC_STRUM C BAR_2 POS_1",
            "CHORD G BAR_1 CHORD C BAR_2 CHORD D BAR_3 CHORD G BAR_4 KICK BAR_1 POS_1 SNARE BAR_1 POS_3",
            "NOTE_ON G4 VEL_75 DUR_8 BAR_1 POS_1 NOTE_ON B4 VEL_70 DUR_4 BAR_1 POS_2 ACOUSTIC_STRUM G BAR_1 POS_3",
            "ACOUSTIC_STRUM G BAR_1 POS_1 KICK BAR_1 POS_1 ACOUSTIC_STRUM C BAR_2 POS_1 SNARE BAR_2 POS_3",
        ]
    }
    
    demo_patterns = []
    for style, pattern_strings in patterns.items():
        for i, pattern_str in enumerate(pattern_strings):
            tokens = pattern_str.split()
            demo_patterns.append({
                'pattern_id': f"{style}_pattern_{i}",
                'tokens': tokens,
                'pattern_string': pattern_str,
                'style': style,
                'bars': 4  # Assume 4 bars for demo
            })
    
    return demo_patterns


def create_demo_vocab() -> TokenVocabulary:
    """Create a vocabulary for the demo"""
    # Collect all unique tokens from demo patterns
    patterns = create_demo_patterns()
    all_tokens = set()
    
    for pattern in patterns:
        all_tokens.update(pattern['tokens'])
    
    # Add common tokens
    common_tokens = {
        '<PAD>', '<UNK>', '<START>', '<END>',
        'STYLE=rock_punk', 'STYLE=rnb_ballad', 'STYLE=country_pop',
        'TEMPO=60', 'TEMPO=70', 'TEMPO=80', 'TEMPO=90', 'TEMPO=100', 
        'TEMPO=110', 'TEMPO=120', 'TEMPO=130', 'TEMPO=140', 'TEMPO=150',
        'KEY=C', 'KEY=G', 'KEY=Bb', 'KEY=F', 'KEY=D', 'KEY=A', 'KEY=E',
        'SECTION=INTRO', 'SECTION=VERSE', 'SECTION=CHORUS', 'SECTION=BRIDGE', 'SECTION=OUTRO'
    }
    
    all_tokens.update(common_tokens)
    
    # Create vocab mapping
    vocab_dict = {token: i for i, token in enumerate(sorted(all_tokens))}
    
    return TokenVocabulary(vocab_dict)


def encode_patterns_with_style_vectors(patterns: List[Dict], embedding_dim: int = 512) -> List[Dict]:
    """Add synthetic style embeddings to patterns"""
    # Create style-specific base vectors
    style_bases = {
        'rock_punk': np.random.randn(embedding_dim) * 0.5,
        'rnb_ballad': np.random.randn(embedding_dim) * 0.5,
        'country_pop': np.random.randn(embedding_dim) * 0.5
    }
    
    encoded_patterns = []
    for pattern in patterns:
        style = pattern['style']
        
        # Create pattern-specific embedding based on style + random variation
        pattern_embedding = (
            style_bases[style] + 
            np.random.randn(embedding_dim) * 0.2
        )
        # Normalize
        pattern_embedding = pattern_embedding / (np.linalg.norm(pattern_embedding) + 1e-8)
        
        pattern_with_embedding = pattern.copy()
        pattern_with_embedding['embedding'] = pattern_embedding
        encoded_patterns.append(pattern_with_embedding)
    
    return encoded_patterns


def build_pattern_index(encoded_patterns: List[Dict]) -> object:
    """Build the pattern index (FAISS or mock)"""
    embedding_dim = len(encoded_patterns[0]['embedding'])
    
    if FAISS_AVAILABLE:
        index = StylePatternIndex(embedding_dim=embedding_dim, index_type="Flat")
    else:
        index = MockStyleIndex()
    
    # Add patterns to index
    for pattern in encoded_patterns:
        if FAISS_AVAILABLE:
            pattern_embedding = PatternEmbedding(
                pattern_id=pattern['pattern_id'],
                pattern_tokens=pattern['tokens'],
                embedding=pattern['embedding'],
                style=pattern['style'],
                bars=pattern['bars']
            )
            index.add_pattern(pattern_embedding)
        else:
            # For mock index, just store the pattern dict
            index.add_pattern(pattern)
    
    logger.info(f"Built pattern index with {len(encoded_patterns)} patterns")
    return index


def demo_style_encoder():
    """Demonstrate the style encoder"""
    logger.info("=== Style Encoder Demo ===")
    
    # Create style encoder
    config = {
        'n_mels': 128,
        'embedding_dim': 512,
        'n_classes': 3,
        'dropout': 0.1
    }
    
    encoder = create_style_encoder(config)
    logger.info(f"Created style encoder with {sum(p.numel() for p in encoder.parameters())} parameters")
    
    # Create dummy audio clips (10 seconds each)
    batch_size = 3
    audio_clips = torch.randn(batch_size, 220500)  # 10s at 22050 Hz
    
    # Encode audio
    with torch.no_grad():
        outputs = encoder(audio_clips)
        
    logger.info(f"Encoded {batch_size} audio clips:")
    logger.info(f"  Embeddings shape: {outputs['embeddings'].shape}")
    logger.info(f"  Classification logits shape: {outputs['logits'].shape}")
    
    # Show style predictions
    style_probs = torch.softmax(outputs['logits'], dim=-1)
    styles = ['rock_punk', 'rnb_ballad', 'country_pop']
    
    for i in range(batch_size):
        predicted_style = styles[torch.argmax(style_probs[i]).item()]
        confidence = torch.max(style_probs[i]).item()
        logger.info(f"  Clip {i}: {predicted_style} (confidence: {confidence:.3f})")
        
    return outputs['embeddings'].numpy()


def demo_pattern_retrieval(query_embeddings: np.ndarray):
    """Demonstrate pattern retrieval"""
    logger.info("\n=== Pattern Retrieval Demo ===")
    
    # Create and encode patterns
    patterns = create_demo_patterns()
    encoded_patterns = encode_patterns_with_style_vectors(patterns)
    
    # Build index
    index = build_pattern_index(encoded_patterns)
    
    # Perform retrieval for each query
    for i, query_embedding in enumerate(query_embeddings):
        logger.info(f"\nQuery {i} retrieval:")
        
        for style in ['rock_punk', 'rnb_ballad', 'country_pop']:
            if hasattr(index, 'search'):
                # FAISS index
                results = index.search(query_embedding, style, top_k=3, min_similarity=0.5)
            else:
                # Mock index
                results = index.search(query_embedding, style, top_k=3)
                
            logger.info(f"  {style}: {len(results)} results")
            for j, (pattern, similarity) in enumerate(results[:2]):  # Show top 2
                if hasattr(pattern, 'pattern_tokens'):
                    tokens = pattern.pattern_tokens[:8]  # First 8 tokens
                else:
                    tokens = pattern['tokens'][:8]
                logger.info(f"    #{j+1}: {' '.join(tokens)}... (sim: {similarity:.3f})")
    
    return index, encoded_patterns


def demo_retrieval_fusion(index, encoded_patterns):
    """Demonstrate retrieval-biased generation"""
    logger.info("\n=== Retrieval Fusion Demo ===")
    
    # Create vocabulary and fusion module
    vocab = create_demo_vocab()
    config = RetrievalConfig(
        enabled=True,
        retrieval_weight=0.4,
        top_k=3,
        fusion_method="shallow",
        ngram_size=3
    )
    
    fusion = RetrievalFusion(vocab, config)
    logger.info(f"Created fusion module with vocabulary size: {vocab.vocab_size}")
    
    # Test different fusion scenarios
    test_contexts = [
        ['STYLE=rock_punk', 'TEMPO=140', 'KICK'],
        ['STYLE=rnb_ballad', 'PIANO', 'Bb4'],
        ['STYLE=country_pop', 'ACOUSTIC_STRUM', 'G']
    ]
    
    for context in test_contexts:
        logger.info(f"\nContext: {' '.join(context)}")
        
        # Get style from context
        style = 'rock_punk' if 'rock_punk' in context[0] else \
                'rnb_ballad' if 'rnb_ballad' in context[0] else 'country_pop'
        
        # Find relevant patterns for this style
        style_patterns = [p for p in encoded_patterns if p['style'] == style][:3]
        similarities = [0.9, 0.8, 0.7]  # Mock similarities
        
        # Update fusion with retrieved patterns
        fusion.update_retrieved_patterns(style_patterns, similarities)
        
        # Compute bias
        bias = fusion.compute_retrieval_bias(context)
        
        # Find tokens with highest bias
        top_bias_indices = torch.topk(bias, 5).indices.tolist()
        top_bias_tokens = [vocab.id_to_token[idx] for idx in top_bias_indices]
        top_bias_values = [bias[idx].item() for idx in top_bias_indices]
        
        logger.info("  Top biased tokens:")
        for token, value in zip(top_bias_tokens, top_bias_values):
            logger.info(f"    {token}: {value:.4f}")
        
        # Test fusion with dummy logits
        dummy_logits = torch.randn(vocab.vocab_size) * 0.1  # Small base logits
        biased_logits = fusion.apply_shallow_fusion(dummy_logits, context)
        
        # Show effect on token probabilities
        original_probs = torch.softmax(dummy_logits, dim=-1)
        biased_probs = torch.softmax(biased_logits, dim=-1)
        
        # Find tokens with biggest probability increase
        prob_increase = biased_probs - original_probs
        top_increased = torch.topk(prob_increase, 3).indices.tolist()
        
        logger.info("  Tokens with increased probability:")
        for idx in top_increased:
            token = vocab.id_to_token[idx]
            orig_prob = original_probs[idx].item()
            new_prob = biased_probs[idx].item()
            increase = new_prob - orig_prob
            logger.info(f"    {token}: {orig_prob:.4f} → {new_prob:.4f} (+{increase:.4f})")


def demo_generation_with_bias():
    """Demonstrate complete generation with bias"""
    logger.info("\n=== Biased Generation Demo ===")
    
    # This would normally use a real transformer model
    # For demo, we'll simulate the process
    
    vocab = create_demo_vocab()
    config = RetrievalConfig(enabled=True, retrieval_weight=0.3)
    
    # Mock generation process
    initial_context = ['STYLE=rock_punk', 'TEMPO=140', 'KEY=C', 'SECTION=VERSE']
    generated_tokens = initial_context.copy()
    
    logger.info(f"Starting generation with: {' '.join(initial_context)}")
    logger.info("Generated sequence:")
    
    # Simulate token generation
    for step in range(8):  # Generate 8 tokens
        # In real implementation, this would use the actual model + fusion
        # For demo, just show the process
        
        # Simulate token selection (biased toward retrieved patterns)
        likely_continuations = [
            'KICK', 'SNARE', 'BASS_PICK', 'CHORD', 'NOTE_ON', 
            'BAR_1', 'POS_1', 'POS_3', 'C2', 'G2'
        ]
        
        next_token = np.random.choice(likely_continuations)
        generated_tokens.append(next_token)
        
        logger.info(f"  Step {step+1}: {' '.join(generated_tokens[-6:])}")  # Show last 6 tokens
    
    final_sequence = ' '.join(generated_tokens)
    logger.info(f"\nFinal generated sequence:")
    logger.info(f"  {final_sequence}")


def main():
    """Run the complete style embeddings + retrieval demo"""
    logger.info("🎵 Style Embeddings + Retrieval Bias Demo 🎵")
    logger.info("=" * 50)
    
    try:
        # 1. Style encoding
        style_embeddings = demo_style_encoder()
        
        # 2. Pattern retrieval
        index, patterns = demo_pattern_retrieval(style_embeddings)
        
        # 3. Retrieval fusion
        demo_retrieval_fusion(index, patterns)
        
        # 4. Generation demo
        demo_generation_with_bias()
        
        logger.info("\n✅ Demo completed successfully!")
        logger.info("\nKey capabilities demonstrated:")
        logger.info("  • Audio encoding to style vectors")
        logger.info("  • FAISS indexing of musical patterns")
        logger.info("  • N-gram matching for continuations")
        logger.info("  • Shallow fusion for bias application")
        logger.info("  • Configurable retrieval weights")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()