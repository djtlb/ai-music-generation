#!/usr/bin/env python3
"""
Sampling script for Melody & Harmony Transformer

Generate melody and harmony sequences using trained model with various
sampling strategies and constraints.
"""

import os
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

import torch
import torch.nn.functional as F
import numpy as np

# Local imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.mh_transformer import MelodyHarmonyTransformer
from models.tokenizer import MIDITokenizer
from utils.constraints import (
    ConstraintMaskGenerator, 
    RepetitionController,
    create_combined_constraint_mask
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MHSampler:
    """Melody & Harmony sampler with constraint support"""
    
    def __init__(
        self,
        model: MelodyHarmonyTransformer,
        tokenizer: MIDITokenizer,
        device: torch.device
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Initialize constraint utilities
        self.constraint_generator = ConstraintMaskGenerator(tokenizer.vocab)
        
        # Style and section mappings
        self.style_mapping = {
            'rock_punk': 0,
            'rnb_ballad': 1,
            'country_pop': 2
        }
        
        self.section_mapping = {
            'intro': 0,
            'verse': 1,
            'chorus': 2,
            'bridge': 3,
            'outro': 4
        }
        
        # Section-specific temperature settings
        self.section_temperatures = {
            'intro': 0.8,
            'verse': 0.9,
            'chorus': 0.7,
            'bridge': 0.85,
            'outro': 0.8
        }
    
    def sample_with_constraints(
        self,
        style: str,
        key: str,
        section: str,
        chord_progression: List[str],
        target_length: int = 256,
        nucleus_p: float = 0.9,
        use_constraints: bool = True,
        prompt_tokens: Optional[List[int]] = None,
        groove_features: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Sample melody and harmony with musical constraints
        
        Args:
            style: Musical style ('rock_punk', 'rnb_ballad', 'country_pop')
            key: Musical key (e.g., 'C_major', 'A_minor')
            section: Song section ('intro', 'verse', 'chorus', 'bridge', 'outro')
            chord_progression: List of chord names
            target_length: Target sequence length
            nucleus_p: Nucleus sampling threshold
            use_constraints: Whether to apply musical constraints
            prompt_tokens: Optional prompt tokens to start generation
            groove_features: Optional drum groove features
            
        Returns:
            Dictionary with generated sequence and metadata
        """
        # Parse inputs
        style_id = self.style_mapping.get(style, 0)
        section_id = self.section_mapping.get(section, 1)
        
        # Parse key signature
        key_parts = key.split('_')
        key_root = self._key_to_int(key_parts[0])
        is_major = len(key_parts) > 1 and key_parts[1] == 'major'
        key_id = key_root * 2 + (0 if is_major else 1)
        
        # Get temperature for section
        temperature = self.section_temperatures.get(section, 0.8)
        
        # Create input tensors
        style_tensor = torch.tensor([style_id], device=self.device)
        key_tensor = torch.tensor([key_id], device=self.device)
        section_tensor = torch.tensor([section_id], device=self.device)
        
        # Initialize sequence with prompt or default start tokens
        if prompt_tokens:
            prompt = torch.tensor([prompt_tokens], device=self.device)
        else:
            # Start with style, key, section tokens
            prompt = self._create_default_prompt(style, key, section)
        
        # Initialize repetition controller
        repetition_controller = RepetitionController(len(self.tokenizer.vocab))
        
        # Generation loop
        generated = prompt.clone()
        
        with torch.no_grad():
            for step in range(target_length - prompt.shape[1]):
                # Forward pass
                outputs = self.model(
                    input_ids=generated,
                    style_ids=style_tensor,
                    key_ids=key_tensor,
                    section_ids=section_tensor,
                    groove_features=groove_features
                )
                
                # Get next token logits
                next_token_logits = outputs['logits'][0, -1, :].clone()
                
                # Apply constraints if enabled
                if use_constraints:
                    constraint_mask = create_combined_constraint_mask(
                        mask_generator=self.constraint_generator,
                        repetition_controller=repetition_controller,
                        key=key_root,
                        chord_sequence=chord_progression,
                        style=style,
                        section=section,
                        generated_sequence=generated[0],
                        current_position=generated.shape[1],
                        seq_len=target_length
                    )
                    
                    next_token_logits = next_token_logits + constraint_mask
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Nucleus sampling
                next_token = self._nucleus_sample(next_token_logits, nucleus_p)
                
                # Update repetition controller
                repetition_controller.update(
                    next_token.item(),
                    is_phrase_boundary=self._is_phrase_boundary(next_token.item())
                )
                
                # Append to sequence
                generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                # Check for early stopping
                if next_token.item() == self.tokenizer.vocab.get('EOS', 1):
                    break
        
        # Decode generated sequence
        generated_tokens = generated[0].cpu().numpy().tolist()
        
        try:
            generated_events = self.tokenizer.decode_tokens(generated_tokens)
        except Exception as e:
            logger.warning(f"Failed to decode tokens: {e}")
            generated_events = []
        
        return {
            'generated_tokens': generated_tokens,
            'generated_events': generated_events,
            'style': style,
            'key': key,
            'section': section,
            'chord_progression': chord_progression,
            'target_length': target_length,
            'actual_length': len(generated_tokens),
            'temperature': temperature,
            'nucleus_p': nucleus_p,
            'used_constraints': use_constraints
        }
    
    def _key_to_int(self, key_str: str) -> int:
        """Convert key string to integer (0-11, C=0)"""
        key_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }
        return key_map.get(key_str, 0)
    
    def _create_default_prompt(self, style: str, key: str, section: str) -> torch.Tensor:
        """Create default prompt tokens"""
        prompt_tokens = []
        
        # Add style token
        style_token = f"STYLE_{style}"
        if style_token in self.tokenizer.vocab:
            prompt_tokens.append(self.tokenizer.vocab[style_token])
        
        # Add section token
        section_token = f"SECTION_{section.upper()}"
        if section_token in self.tokenizer.vocab:
            prompt_tokens.append(self.tokenizer.vocab[section_token])
        
        # Add bar marker
        if 'BAR' in self.tokenizer.vocab:
            prompt_tokens.append(self.tokenizer.vocab['BAR'])
        
        # Ensure we have at least some tokens
        if not prompt_tokens:
            prompt_tokens = [self.tokenizer.vocab.get('PAD', 0)]
        
        return torch.tensor([prompt_tokens], device=self.device)
    
    def _nucleus_sample(self, logits: torch.Tensor, nucleus_p: float = 0.9) -> torch.Tensor:
        """Nucleus (top-p) sampling"""
        if nucleus_p >= 1.0:
            return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > nucleus_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        
        # Apply mask
        logits = logits.clone()
        logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')
        
        # Sample from remaining tokens
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    def _is_phrase_boundary(self, token_id: int) -> bool:
        """Check if token represents a phrase boundary"""
        token_name = self.tokenizer.reverse_vocab.get(token_id, '')
        return token_name in ['BAR', 'SECTION_VERSE', 'SECTION_CHORUS', 'SECTION_BRIDGE']
    
    def generate_multiple_variations(
        self,
        style: str,
        key: str,
        section: str,
        chord_progression: List[str],
        num_variations: int = 5,
        **kwargs
    ) -> List[Dict]:
        """Generate multiple variations of a melody/harmony"""
        variations = []
        
        for i in range(num_variations):
            logger.info(f"Generating variation {i+1}/{num_variations}")
            
            # Slightly vary temperature for diversity
            base_temp = self.section_temperatures.get(section, 0.8)
            temp_variation = np.random.uniform(0.9, 1.1)
            kwargs['temperature'] = base_temp * temp_variation
            
            variation = self.sample_with_constraints(
                style=style,
                key=key,
                section=section,
                chord_progression=chord_progression,
                **kwargs
            )
            
            variation['variation_id'] = i + 1
            variations.append(variation)
        
        return variations


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> MelodyHarmonyTransformer:
    """Load model from checkpoint"""
    logger.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']['model']
    
    model = MelodyHarmonyTransformer(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully (epoch {checkpoint['epoch']})")
    return model


def main():
    parser = argparse.ArgumentParser(description='Sample from Melody & Harmony Transformer')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, help='Path to sampling config file')
    parser.add_argument('--output-dir', type=str, default='./outputs/mh_samples', help='Output directory')
    
    # Generation parameters
    parser.add_argument('--style', type=str, default='rock_punk', 
                       choices=['rock_punk', 'rnb_ballad', 'country_pop'], help='Musical style')
    parser.add_argument('--key', type=str, default='C_major', help='Musical key (e.g., C_major, A_minor)')
    parser.add_argument('--section', type=str, default='verse',
                       choices=['intro', 'verse', 'chorus', 'bridge', 'outro'], help='Song section')
    parser.add_argument('--chord-progression', type=str, nargs='+', 
                       default=['C_maj', 'F_maj', 'G_maj', 'C_maj'], help='Chord progression')
    parser.add_argument('--length', type=int, default=256, help='Target sequence length')
    parser.add_argument('--nucleus-p', type=float, default=0.9, help='Nucleus sampling threshold')
    parser.add_argument('--num-variations', type=int, default=1, help='Number of variations to generate')
    parser.add_argument('--disable-constraints', action='store_true', help='Disable musical constraints')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load tokenizer
    tokenizer = MIDITokenizer()
    vocab_path = 'src/models/vocab.json'
    if os.path.exists(vocab_path):
        tokenizer.load_vocab(vocab_path)
    else:
        logger.error(f'Tokenizer vocabulary not found at {vocab_path}')
        return
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, device)
    
    # Create sampler
    sampler = MHSampler(model, tokenizer, device)
    
    # Load sampling config if provided
    if args.config:
        with open(args.config, 'r') as f:
            sampling_config = yaml.safe_load(f)
    else:
        sampling_config = {}
    
    # Override config with command line arguments
    sampling_params = {
        'target_length': args.length,
        'nucleus_p': args.nucleus_p,
        'use_constraints': not args.disable_constraints,
        **sampling_config
    }
    
    # Generate samples
    logger.info(f"Generating {args.num_variations} variations...")
    logger.info(f"Style: {args.style}, Key: {args.key}, Section: {args.section}")
    logger.info(f"Chord progression: {args.chord_progression}")
    
    if args.num_variations > 1:
        results = sampler.generate_multiple_variations(
            style=args.style,
            key=args.key,
            section=args.section,
            chord_progression=args.chord_progression,
            num_variations=args.num_variations,
            **sampling_params
        )
    else:
        result = sampler.sample_with_constraints(
            style=args.style,
            key=args.key,
            section=args.section,
            chord_progression=args.chord_progression,
            **sampling_params
        )
        results = [result]
    
    # Save results
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    for i, result in enumerate(results):
        # Save JSON
        if args.num_variations > 1:
            filename = f'mh_sample_{args.style}_{args.section}_var{i+1}_{timestamp}.json'
        else:
            filename = f'mh_sample_{args.style}_{args.section}_{timestamp}.json'
        
        output_path = os.path.join(args.output_dir, filename)
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved sample to {output_path}")
        
        # Print summary
        logger.info(f"Variation {i+1}:")
        logger.info(f"  Generated {result['actual_length']} tokens")
        logger.info(f"  Generated {len(result['generated_events'])} MIDI events")
        logger.info(f"  Used constraints: {result['used_constraints']}")
    
    logger.info("Sampling completed!")


if __name__ == '__main__':
    # Add pandas import for timestamp
    import pandas as pd
    main()