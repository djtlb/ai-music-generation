#!/usr/bin/env python3
"""
Sampling script for Arrangement Transformer

Usage:
    python scripts/sample_arrangement.py --checkpoint path/to/model.ckpt --style rock_punk --tempo 140 --duration 64
    python scripts/sample_arrangement.py --config configs/arrangement/default.yaml --style rnb_ballad --tempo 80 --duration 96
"""

import argparse
import yaml
import torch
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.arrangement_transformer import ArrangementTransformer, load_config


def print_arrangement(arrangement: List[Dict], style: str, tempo: int, duration: int):
    """Pretty print the generated arrangement"""
    print(f"\n=== Generated Arrangement ===")
    print(f"Style: {style}")
    print(f"Tempo: {tempo} BPM")
    print(f"Target Duration: {duration} bars")
    print(f"Generated Sections: {len(arrangement)}")
    print()
    
    total_bars = 0
    for i, section in enumerate(arrangement):
        section_type = section['type']
        start_bar = section['start_bar']
        length_bars = section['length_bars']
        end_bar = start_bar + length_bars
        total_bars = max(total_bars, end_bar)
        
        print(f"{i+1:2d}. {section_type:8s} | Bars {start_bar:3d}-{end_bar:3d} ({length_bars:2d} bars)")
    
    print(f"\nTotal length: {total_bars} bars")
    
    # Calculate section distribution
    section_counts = {}
    for section in arrangement:
        section_type = section['type']
        section_counts[section_type] = section_counts.get(section_type, 0) + 1
    
    print("\nSection distribution:")
    for section_type, count in section_counts.items():
        print(f"  {section_type}: {count}")


def generate_arrangement(model: ArrangementTransformer,
                        style: str,
                        tempo: int,
                        duration: int,
                        num_samples: int = 1,
                        temperature: float = 0.9,
                        top_k: int = 50,
                        top_p: float = 0.9,
                        max_length: int = 32) -> List[List[Dict]]:
    """Generate multiple arrangement samples"""
    
    print(f"Generating {num_samples} arrangement(s)...")
    print(f"Parameters: style={style}, tempo={tempo}, duration={duration}")
    print(f"Sampling: temp={temperature}, top_k={top_k}, top_p={top_p}")
    print()
    
    arrangements = []
    
    for i in range(num_samples):
        print(f"Generating sample {i+1}/{num_samples}...")
        
        arrangement = model.generate_arrangement(
            style=style,
            tempo=tempo,
            target_duration=duration,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        arrangements.append(arrangement)
        
        if num_samples == 1:
            print_arrangement(arrangement, style, tempo, duration)
        else:
            print(f"  Generated {len(arrangement)} sections, {arrangement[-1]['start_bar'] + arrangement[-1]['length_bars'] if arrangement else 0} total bars")
    
    return arrangements


def main():
    parser = argparse.ArgumentParser(description="Generate arrangement with trained model")
    
    # Model loading options (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--checkpoint", type=str,
                           help="Path to model checkpoint (.ckpt file)")
    model_group.add_argument("--config", type=str,
                           help="Path to config file (will use random weights)")
    
    # Generation parameters
    parser.add_argument("--style", type=str, required=True,
                       choices=['rock_punk', 'rnb_ballad', 'country_pop'],
                       help="Music style")
    parser.add_argument("--tempo", type=int, required=True,
                       help="Target tempo in BPM")
    parser.add_argument("--duration", type=int, required=True,
                       help="Target duration in bars")
    
    # Sampling parameters
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of samples to generate")
    parser.add_argument("--temperature", type=float, default=0.9,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--max_length", type=int, default=32,
                       help="Maximum sequence length")
    
    # Output options
    parser.add_argument("--output", type=str,
                       help="Save arrangements to JSON file")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Load model
    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model = ArrangementTransformer.load_from_checkpoint(args.checkpoint)
    else:
        print(f"Loading model from config: {args.config}")
        config = load_config(args.config)
        model = ArrangementTransformer(config)
        print("Warning: Using randomly initialized weights!")
    
    # Set to evaluation mode
    model.eval()
    
    # Generate arrangements
    arrangements = generate_arrangement(
        model=model,
        style=args.style,
        tempo=args.tempo,
        duration=args.duration,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_length=args.max_length
    )
    
    # Print all arrangements if multiple samples
    if args.num_samples > 1:
        for i, arrangement in enumerate(arrangements):
            print(f"\n--- Sample {i+1} ---")
            print_arrangement(arrangement, args.style, args.tempo, args.duration)
    
    # Save to file if requested
    if args.output:
        output_data = []
        for arrangement in arrangements:
            output_data.append({
                'style': args.style,
                'tempo': args.tempo,
                'duration_bars': args.duration,
                'sections': arrangement,
                'generation_params': {
                    'temperature': args.temperature,
                    'top_k': args.top_k,
                    'top_p': args.top_p,
                    'seed': args.seed
                }
            })
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nSaved {len(arrangements)} arrangement(s) to {args.output}")


if __name__ == "__main__":
    main()