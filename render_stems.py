#!/usr/bin/env python3
"""
CLI script for rendering MIDI files to audio stems using the sampler-based renderer.

Usage:
    python render_stems.py --midi path/to/file.mid --style rock_punk --output stems/
    python render_stems.py --midi composition.mid --style rnb_ballad --song-id "my_song_001"
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add the audio module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from audio.render import render_midi_to_stems, RenderConfig

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

def validate_args(args):
    """Validate command line arguments."""
    # Check if MIDI file exists
    if not os.path.exists(args.midi):
        print(f"Error: MIDI file not found: {args.midi}")
        sys.exit(1)
    
    # Check if output directory exists, create if not
    output_path = Path(args.output)
    if not output_path.exists():
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"Created output directory: {output_path}")
        except Exception as e:
            print(f"Error creating output directory: {e}")
            sys.exit(1)
    
    # Validate style
    valid_styles = ['rock_punk', 'rnb_ballad', 'country_pop']
    if args.style not in valid_styles:
        print(f"Error: Invalid style '{args.style}'. Valid styles: {', '.join(valid_styles)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Render MIDI files to audio stems using style-specific instrument presets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --midi song.mid --style rock_punk
  %(prog)s --midi ballad.mid --style rnb_ballad --output /path/to/stems --song-id "ballad_v1"
  %(prog)s --midi track.mid --style country_pop --sample-rate 44100 --normalize
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--midi", 
        required=True,
        help="Path to input MIDI file"
    )
    
    parser.add_argument(
        "--style", 
        required=True,
        choices=['rock_punk', 'rnb_ballad', 'country_pop'],
        help="Music style for instrument selection"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output", 
        default="stems",
        help="Output directory for stems (default: stems)"
    )
    
    parser.add_argument(
        "--song-id",
        help="Unique identifier for the song (used as subfolder name)"
    )
    
    # Audio configuration
    parser.add_argument(
        "--sample-rate", 
        type=int,
        default=48000,
        choices=[44100, 48000, 96000],
        help="Sample rate for output audio (default: 48000)"
    )
    
    parser.add_argument(
        "--bit-depth",
        type=int,
        default=24,
        choices=[16, 24, 32],
        help="Bit depth for output audio (default: 24)"
    )
    
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize stems to target LUFS"
    )
    
    parser.add_argument(
        "--lufs-target",
        type=float,
        default=-18.0,
        help="Target LUFS for normalization (default: -18.0)"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        help="Force specific duration in seconds (default: auto-detect from MIDI)"
    )
    
    parser.add_argument(
        "--fade-out",
        type=float,
        default=0.1,
        help="Fade out duration in seconds (default: 0.1)"
    )
    
    # Processing options
    parser.add_argument(
        "--no-latency-compensation",
        action="store_true",
        help="Disable latency compensation"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Development/debugging options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse arguments and validate setup without rendering"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    validate_args(args)
    
    # Create render configuration
    config = RenderConfig(
        sample_rate=args.sample_rate,
        bit_depth=args.bit_depth,
        normalize_stems=args.normalize,
        normalize_target_lufs=args.lufs_target,
        apply_latency_compensation=not args.no_latency_compensation,
        render_length_seconds=args.duration,
        fade_out_seconds=args.fade_out
    )
    
    # Print configuration
    logger.info("Render Configuration:")
    logger.info(f"  MIDI file: {args.midi}")
    logger.info(f"  Style: {args.style}")
    logger.info(f"  Output directory: {args.output}")
    logger.info(f"  Song ID: {args.song_id or 'auto-generated'}")
    logger.info(f"  Sample rate: {config.sample_rate} Hz")
    logger.info(f"  Bit depth: {config.bit_depth} bits")
    logger.info(f"  Normalize stems: {config.normalize_stems}")
    if config.normalize_stems:
        logger.info(f"  Target LUFS: {config.normalize_target_lufs}")
    logger.info(f"  Latency compensation: {config.apply_latency_compensation}")
    if config.render_length_seconds:
        logger.info(f"  Forced duration: {config.render_length_seconds} seconds")
    
    # Dry run mode
    if args.dry_run:
        logger.info("Dry run mode - configuration validated successfully")
        return 0
    
    # Perform rendering
    try:
        logger.info("Starting stem rendering...")
        
        result = render_midi_to_stems(
            midi_path=args.midi,
            style=args.style,
            output_dir=args.output,
            song_id=args.song_id,
            config=config
        )
        
        # Print results
        print("\n" + "="*60)
        print("RENDERING COMPLETE!")
        print("="*60)
        print(f"Style: {args.style}")
        print(f"Generated {len(result)} stems:")
        
        for instrument_role, file_path in result.items():
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  {instrument_role:15} -> {file_path} ({file_size:.1f} MB)")
        
        # Calculate total size
        total_size = sum(os.path.getsize(path) for path in result.values()) / (1024 * 1024)
        print(f"\nTotal size: {total_size:.1f} MB")
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Rendering interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Rendering failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())