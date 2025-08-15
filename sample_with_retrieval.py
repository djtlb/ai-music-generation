#!/usr/bin/env python3
"""
Enhanced sampling script with hierarchical retrieval fusion support.
Supports --family_index and --child_bias flags for style-aware generation.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from style.retrieval_fusion import create_retrieval_fusion, RetrievalAugmentedSampler


def main():
    parser = argparse.ArgumentParser(
        description="Generate music with hierarchical style retrieval fusion"
    )
    
    # Model and generation parameters
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    
    parser.add_argument(
        "--vocab_file",
        type=str,
        default="models/vocab.json",
        help="Path to vocabulary JSON file"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="STYLE=pop TEMPO=120 KEY=C SECTION=VERSE",
        help="Generation prompt"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum generation length"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold"
    )
    
    # Retrieval fusion parameters
    parser.add_argument(
        "--family_index",
        type=str,
        help="Parent genre for retrieval (e.g., 'pop', 'rock')"
    )
    
    parser.add_argument(
        "--child_bias",
        type=float,
        default=0.0,
        help="Bias weight for child patterns (0.0-1.0)"
    )
    
    parser.add_argument(
        "--child_genre",
        type=str,
        help="Specific child genre for bias (e.g., 'dance_pop')"
    )
    
    parser.add_argument(
        "--faiss_index_dir",
        type=str,
        default="indices",
        help="Directory containing FAISS indices"
    )
    
    parser.add_argument(
        "--fusion_weight",
        type=float,
        default=0.1,
        help="Weight for retrieval fusion (0.0-1.0)"
    )
    
    parser.add_argument(
        "--ngram_size",
        type=int,
        default=3,
        help="Size of n-grams for pattern matching"
    )
    
    parser.add_argument(
        "--top_k_patterns",
        type=int,
        default=5,
        help="Number of similar patterns to retrieve"
    )
    
    # Output parameters
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file for generated tokens"
    )
    
    parser.add_argument(
        "--output_midi",
        type=str,
        help="Output MIDI file (if tokenizer supports conversion)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Load model and tokenizer (placeholder - replace with actual model loading)
    logger.info(f"Loading model from {args.model_path}")
    
    # For now, create a mock model interface
    class MockModel:
        def __init__(self):
            self.device = "cpu"
            
        def __call__(self, input_ids):
            # Mock outputs - replace with actual model inference
            import torch
            batch_size, seq_len = input_ids.shape
            vocab_size = 10000  # Replace with actual vocab size
            
            class MockOutput:
                def __init__(self):
                    # Random logits for demonstration
                    self.logits = torch.randn(batch_size, seq_len, vocab_size)
            
            return MockOutput()
    
    class MockTokenizer:
        def __init__(self, vocab_file):
            with open(vocab_file, 'r') as f:
                self.vocab = json.load(f)
            self.eos_token_id = self.vocab.get('<EOS>', 1)
            
        def encode(self, text):
            # Simple tokenization - replace with actual tokenizer
            tokens = text.split()
            return [self.vocab.get(token, 0) for token in tokens]
            
        def decode(self, token_ids):
            id_to_token = {v: k for k, v in self.vocab.items()}
            return ' '.join(id_to_token.get(tid, '<UNK>') for tid in token_ids)
    
    # Load components
    try:
        model = MockModel()  # Replace with actual model loading
        tokenizer = MockTokenizer(args.vocab_file)
    except Exception as e:
        logger.error(f"Error loading model/tokenizer: {e}")
        return
    
    # Setup retrieval fusion if family_index is provided
    retrieval_sampler = None
    
    if args.family_index:
        logger.info(f"Setting up retrieval fusion for family: {args.family_index}")
        
        try:
            retrieval_fusion = create_retrieval_fusion(
                faiss_index_dir=args.faiss_index_dir,
                vocab_file=args.vocab_file,
                family_index=args.family_index,
                child_bias=args.child_bias,
                child_genre=args.child_genre,
                fusion_weight=args.fusion_weight,
                ngram_size=args.ngram_size,
                top_k_patterns=args.top_k_patterns
            )
            
            retrieval_sampler = RetrievalAugmentedSampler(
                model=model,
                tokenizer=tokenizer,
                retrieval_fusion=retrieval_fusion
            )
            
            logger.info("Retrieval fusion enabled")
            if args.child_genre and args.child_bias > 0:
                logger.info(f"Child bias: {args.child_genre} (weight: {args.child_bias})")
                
        except Exception as e:
            logger.error(f"Error setting up retrieval fusion: {e}")
            logger.info("Falling back to standard generation")
    
    # Encode prompt
    prompt_tokens = tokenizer.encode(args.prompt)
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Prompt tokens: {prompt_tokens}")
    
    # Generate
    logger.info("Generating...")
    
    if retrieval_sampler:
        # Generate with retrieval fusion
        generated_tokens = retrieval_sampler.generate_with_retrieval(
            prompt_tokens=prompt_tokens,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p
        )
    else:
        # Standard generation (mock implementation)
        import torch
        generated_tokens = prompt_tokens.copy()
        
        for _ in range(min(50, args.max_length - len(prompt_tokens))):
            # Mock generation - replace with actual model inference
            next_token = torch.randint(0, len(tokenizer.vocab), (1,)).item()
            generated_tokens.append(next_token)
            
            if next_token == tokenizer.eos_token_id:
                break
    
    # Decode results
    generated_text = tokenizer.decode(generated_tokens)
    
    logger.info("Generation complete!")
    logger.info(f"Generated {len(generated_tokens)} tokens")
    
    # Output results
    print("\n" + "="*50)
    print("GENERATED SEQUENCE:")
    print("="*50)
    print(generated_text)
    print("="*50)
    
    # Save to file if specified
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(generated_text)
        
        logger.info(f"Saved generated text to: {args.output_file}")
    
    # Save metadata
    metadata = {
        "prompt": args.prompt,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "generated_text": generated_text,
        "generation_params": {
            "max_length": args.max_length,
            "temperature": args.temperature,
            "top_p": args.top_p
        },
        "retrieval_params": {
            "family_index": args.family_index,
            "child_bias": args.child_bias,
            "child_genre": args.child_genre,
            "fusion_weight": args.fusion_weight,
            "enabled": args.family_index is not None
        }
    }
    
    if args.output_file:
        metadata_file = output_path.with_suffix('.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to: {metadata_file}")


if __name__ == "__main__":
    main()