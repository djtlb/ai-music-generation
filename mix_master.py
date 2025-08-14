#!/usr/bin/env python3
"""
CLI tool for automatic mixing and mastering using the differentiable mixing chain.

This script processes audio stems through the auto-mixing system to achieve
target LUFS, spectral characteristics, and stereo imaging for different styles.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
import torch
import torchaudio
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from mix import AutoMixChain, load_style_targets
from mix.utils import (
    compute_lufs, 
    compute_spectral_centroid, 
    compute_stereo_ms_ratio,
    compute_mix_quality_score,
    export_mix_analysis,
    create_white_noise_stems
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_audio_stems(stem_paths: List[str], target_sample_rate: int = 48000) -> List[torch.Tensor]:
    """
    Load audio stems from file paths.
    
    Args:
        stem_paths: List of paths to audio files
        target_sample_rate: Target sample rate for processing
        
    Returns:
        List of audio tensors
    """
    stems = []
    logger = logging.getLogger(__name__)
    
    for stem_path in stem_paths:
        try:
            audio, sample_rate = torchaudio.load(stem_path)
            
            # Resample if necessary
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
                audio = resampler(audio)
                
            # Convert to stereo if mono
            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)
            elif audio.shape[0] > 2:
                # Mix down to stereo
                audio = audio[:2]
                
            stems.append(audio)
            logger.info(f"Loaded stem: {stem_path} ({audio.shape})")
            
        except Exception as e:
            logger.error(f"Failed to load stem {stem_path}: {e}")
            raise
            
    return stems


def create_test_stems(n_stems: int = 4, duration: float = 10.0, 
                     sample_rate: int = 48000) -> List[torch.Tensor]:
    """Create test stems for demonstration/validation."""
    return create_white_noise_stems(n_stems, duration, sample_rate)


def process_stems(stems: List[torch.Tensor], style: str, 
                 model_path: Optional[str] = None,
                 style_targets: Optional[Dict] = None) -> tuple:
    """
    Process stems through auto-mixing chain.
    
    Args:
        stems: List of audio tensors
        style: Target mixing style
        model_path: Path to trained model (optional)
        style_targets: Style targets dictionary
        
    Returns:
        (mixed_audio, analysis_results)
    """
    logger = logging.getLogger(__name__)
    
    # Load style targets
    if style_targets is None:
        style_targets = load_style_targets()
        
    # Create auto-mix chain
    auto_mix = AutoMixChain(
        n_stems=len(stems),
        sample_rate=48000,
        style_targets=style_targets
    )
    
    # Load trained model if provided
    if model_path and Path(model_path).exists():
        logger.info(f"Loading trained model from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        auto_mix.load_state_dict(checkpoint['model_state_dict'])
        auto_mix.eval()
    else:
        logger.info("Using randomly initialized model (no trained weights)")
        
    # Process stems
    with torch.no_grad():
        mixed_audio, analysis = auto_mix(stems, style)
        
    logger.info(f"Mixed audio shape: {mixed_audio.shape}")
    logger.info(f"Analysis results: {analysis}")
    
    return mixed_audio, analysis


def save_output(mixed_audio: torch.Tensor, output_path: str, 
               sample_rate: int = 48000):
    """Save mixed audio to file."""
    logger = logging.getLogger(__name__)
    
    # Ensure audio is in valid range
    mixed_audio = torch.clamp(mixed_audio, -1.0, 1.0)
    
    torchaudio.save(output_path, mixed_audio, sample_rate)
    logger.info(f"Saved mixed audio to: {output_path}")


def validate_targets(analysis: Dict, style: str, targets: Dict) -> Dict:
    """
    Validate mix against targets and compute accuracy metrics.
    
    Args:
        analysis: Mix analysis results
        style: Style name
        targets: Style targets
        
    Returns:
        Validation results
    """
    logger = logging.getLogger(__name__)
    
    if style not in targets:
        logger.warning(f"No targets defined for style: {style}")
        return {"status": "no_targets"}
        
    style_targets = targets[style]
    validation = {
        "style": style,
        "targets": style_targets,
        "actual": analysis,
        "errors": {},
        "within_tolerance": {}
    }
    
    # LUFS validation
    if 'lufs' in style_targets:
        target_lufs = style_targets['lufs']
        actual_lufs = analysis['lufs']
        error = abs(actual_lufs - target_lufs)
        
        validation["errors"]["lufs"] = error
        validation["within_tolerance"]["lufs"] = error <= 1.0  # 1dB tolerance
        
        logger.info(f"LUFS - Target: {target_lufs:.1f}, Actual: {actual_lufs:.1f}, Error: {error:.1f}dB")
        
    # Spectral centroid validation
    if 'spectral_centroid_hz' in style_targets:
        target_centroid = style_targets['spectral_centroid_hz']
        actual_centroid = analysis['spectral_centroid']
        error = abs(actual_centroid - target_centroid)
        
        validation["errors"]["spectral_centroid"] = error
        validation["within_tolerance"]["spectral_centroid"] = error <= 200.0  # 200Hz tolerance
        
        logger.info(f"Spectral Centroid - Target: {target_centroid:.0f}Hz, Actual: {actual_centroid:.0f}Hz, Error: {error:.0f}Hz")
        
    # Stereo ratio validation
    if 'stereo_ms_ratio' in style_targets:
        target_ratio = style_targets['stereo_ms_ratio']
        actual_ratio = analysis['stereo_ms_ratio']
        error = abs(actual_ratio - target_ratio)
        
        validation["errors"]["stereo_ms_ratio"] = error
        validation["within_tolerance"]["stereo_ms_ratio"] = error <= 0.2  # 0.2 ratio tolerance
        
        logger.info(f"M/S Ratio - Target: {target_ratio:.2f}, Actual: {actual_ratio:.2f}, Error: {error:.2f}")
        
    # Overall quality score
    validation["quality_score"] = compute_mix_quality_score(analysis, style, targets)
    logger.info(f"Overall quality score: {validation['quality_score']:.2f}")
    
    return validation


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Automatic mixing and mastering with target LUFS and spectral control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mix stems with rock_punk style
  python mix_master.py --stems track1.wav track2.wav track3.wav --style rock_punk --output mixed.wav
  
  # Create test mix with validation
  python mix_master.py --test-stems 4 --style rnb_ballad --output test_mix.wav --validate
  
  # Use custom style targets
  python mix_master.py --stems *.wav --style country_pop --targets custom_targets.yaml --output final.wav
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--stems', 
        nargs='+', 
        help='Paths to audio stem files'
    )
    input_group.add_argument(
        '--test-stems',
        type=int,
        metavar='N',
        help='Create N test stems with white noise'
    )
    
    # Processing options
    parser.add_argument(
        '--style',
        required=True,
        choices=['rock_punk', 'rnb_ballad', 'country_pop'],
        help='Target mixing style'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output audio file path'
    )
    parser.add_argument(
        '--model',
        help='Path to trained model weights (optional)'
    )
    parser.add_argument(
        '--targets',
        help='Path to custom style targets YAML file'
    )
    
    # Analysis options
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate output against targets'
    )
    parser.add_argument(
        '--analysis-output',
        help='Path to save analysis results JSON'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=48000,
        help='Sample rate for processing (default: 48000)'
    )
    
    # Other options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load stems
        if args.stems:
            logger.info(f"Loading {len(args.stems)} stems...")
            stems = load_audio_stems(args.stems, args.sample_rate)
        else:
            logger.info(f"Creating {args.test_stems} test stems...")
            stems = create_test_stems(args.test_stems, duration=10.0, sample_rate=args.sample_rate)
            
        # Load style targets
        style_targets = None
        if args.targets:
            logger.info(f"Loading custom targets from {args.targets}")
            with open(args.targets, 'r') as f:
                custom_targets = yaml.safe_load(f)
                # Extract style_targets section if it exists
                if 'style_targets' in custom_targets:
                    style_targets = custom_targets['style_targets']
                else:
                    style_targets = custom_targets
        else:
            style_targets = load_style_targets()
            
        # Process stems
        logger.info(f"Processing stems with style: {args.style}")
        mixed_audio, analysis = process_stems(
            stems, 
            args.style, 
            args.model,
            style_targets
        )
        
        # Save output
        save_output(mixed_audio, args.output, args.sample_rate)
        
        # Validation
        if args.validate:
            logger.info("Validating output against targets...")
            validation = validate_targets(analysis, args.style, style_targets)
            
            # Print validation summary
            print("\n" + "="*50)
            print("VALIDATION RESULTS")
            print("="*50)
            print(f"Style: {args.style}")
            print(f"Quality Score: {validation.get('quality_score', 0):.2f}/1.00")
            
            if 'within_tolerance' in validation:
                for metric, within_tol in validation['within_tolerance'].items():
                    status = "✓ PASS" if within_tol else "✗ FAIL"
                    error = validation['errors'].get(metric, 0)
                    print(f"{metric}: {status} (error: {error:.2f})")
            print("="*50)
            
        # Save analysis
        if args.analysis_output:
            export_mix_analysis(analysis, args.analysis_output)
            logger.info(f"Analysis saved to: {args.analysis_output}")
            
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()