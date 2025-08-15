#!/usr/bin/env python3
"""
Demo script showing various usage examples of the AI Music Composition Pipeline
"""

import subprocess
import sys
from pathlib import Path

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🎵 {title}")
    print(f"{'='*60}")

def run_example(description: str, cmd: list):
    """Run an example command"""
    print(f"\n📝 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 40)
    
    # Note: We're not actually running these in the demo to avoid long execution times
    print("(This would execute the pipeline - skipped in demo)")
    return True

def main():
    """Show usage examples"""
    print_header("AI Music Composition Pipeline - Usage Examples")
    
    print("""
The pipeline orchestrates 7 steps to create a complete song:
1. Data Ingestion → 2. MIDI Tokenization → 3. Arrangement Generation
4. Melody/Harmony → 5. Stem Rendering → 6. Mix/Master → 7. Export & Report
    """)
    
    # Example 1: Basic usage
    run_example(
        "Basic rock punk song with default settings",
        ["python", "run_pipeline.py", "--style", "rock_punk"]
    )
    
    # Example 2: Custom parameters
    run_example(
        "R&B ballad with custom duration and tempo",
        ["python", "run_pipeline.py", "--style", "rnb_ballad", "--duration_bars", "96", "--bpm", "65", "--key", "Bb"]
    )
    
    # Example 3: Using config file
    run_example(
        "Country pop with style-specific config",
        ["python", "run_pipeline.py", "--style", "country_pop", "--config", "configs/country_pop.yaml"]
    )
    
    # Example 4: Verbose output
    run_example(
        "Rock punk with detailed logging",
        ["python", "run_pipeline.py", "--style", "rock_punk", "--duration_bars", "32", "--verbose"]
    )
    
    print_header("Output Structure")
    print("""
The pipeline creates timestamped output directories:

exports/20241203_143022/
├── final.wav                 # Final mastered audio
├── report.json              # Analysis and pipeline metadata  
├── arrangement.json         # Generated song structure
├── composition_metadata.json # Input parameters
└── stems/                   # Individual instrument tracks
    ├── drums.wav
    ├── bass.wav
    ├── guitar.wav
    └── ...
    """)
    
    print_header("Configuration Files")
    print("""
Style-specific configs are in the configs/ directory:

configs/
├── default.yaml             # Base configuration
├── rock_punk.yaml          # Rock/punk specific settings
├── rnb_ballad.yaml         # R&B ballad specific settings
└── country_pop.yaml        # Country pop specific settings

Each config defines:
- Default BPM, key, duration
- Instrument selection
- Mix/master targets (LUFS, spectral tilt, stereo width)
- Arrangement preferences
- Sound design parameters
    """)
    
    print_header("Pipeline Steps Detail")
    print("""
1. 📥 Data Ingestion
   - Creates composition metadata
   - Sets up style-specific parameters

2. 🔤 MIDI Tokenization  
   - Converts musical concepts to tokens
   - Handles STYLE, TEMPO, KEY, SECTION markers

3. 🏗️  Arrangement Generation
   - Creates song structure (INTRO, VERSE, CHORUS, etc.)
   - Uses Transformer decoder with coverage penalty

4. 🎼 Melody & Harmony
   - Generates MIDI for all instruments
   - Style-conditioned with constraint masks

5. 🎚️  Stem Rendering
   - Maps MIDI to audio using sample libraries
   - Style-specific instrument presets

6. 🎛️  Mix & Master
   - Auto-mixing with style targets
   - LUFS normalization, EQ, compression

7. 📊 Export & Report
   - Saves final WAV and analysis data
   - Validates against quality metrics
    """)
    
    print_header("Testing")
    print("""
To test the pipeline:

# Run tests for all styles
python test_pipeline.py

# Manual test with specific style
python run_pipeline.py --style rock_punk --duration_bars 8

# Check pipeline health
python run_pipeline.py --help
    """)
    
    print_header("Dependencies")
    print("""
Install pipeline requirements:

pip install -r requirements-pipeline.txt

Key dependencies:
- hydra-core (configuration management)
- torch + pytorch-lightning (ML models)
- librosa (audio analysis)
- mido (MIDI processing)
    """)
    
    print(f"\n✨ Ready to create AI music! Try running:")
    print(f"   python run_pipeline.py --style rock_punk")
    print()

if __name__ == "__main__":
    main()