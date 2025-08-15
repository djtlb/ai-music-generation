# AI Music Composition Pipeline

An end-to-end system that generates complete, mastered songs from style inputs using AI. Goes from style selection to finished WAV in one command.

## Overview

The pipeline orchestrates 7 interconnected steps to create professional-quality music:

```
Style Input → Tokenization → Arrangement → Melody/Harmony → Stem Rendering → Mix/Master → Final WAV
```

### Supported Styles
- **rock_punk**: High-energy, aggressive, distorted guitars
- **rnb_ballad**: Smooth, warm, gradual builds  
- **country_pop**: Balanced, acoustic-electric blend

## Quick Start

```bash
# Install dependencies
pip install -r requirements-pipeline.txt

# Generate a rock punk song
python run_pipeline.py --style rock_punk

# Custom R&B ballad
python run_pipeline.py --style rnb_ballad --duration_bars 96 --bpm 65 --key Bb

# Country pop with config
python run_pipeline.py --style country_pop --config configs/country_pop.yaml
```

## Pipeline Steps

### 1. Data Ingestion 📥
Creates composition metadata and style-specific parameters.

**Inputs**: Style, duration, BPM, key  
**Outputs**: `composition_metadata.json`

### 2. MIDI Tokenization 🔤
Converts musical concepts to discrete tokens for AI processing.

**Tokens**: `STYLE={style}`, `TEMPO={bpm}`, `KEY={key}`, `SECTION={type}`, `BAR`, `POS`, `INST`, `CHORD`, `NOTE_ON/OFF`  
**Outputs**: Token sequence

### 3. Arrangement Generation 🏗️
Generates song structure using Transformer decoder.

**Model**: PyTorch + Lightning with teacher forcing + coverage penalty  
**Outputs**: Section sequence with bar counts (`arrangement.json`)

### 4. Melody & Harmony Generation 🎼
Creates multi-track MIDI with style conditioning.

**Features**: 
- Cross-entropy on events
- In-scale penalty, repetition penalty, chord compatibility
- Temperature sampling by section
- Constraint masks

**Outputs**: Multi-track MIDI file

### 5. Stem Rendering 🎚️
Renders MIDI to audio using style-specific sample libraries.

**Process**: 
- Maps instrument roles to SFZ/SF2 presets
- Style-specific drum kits, bass tones, guitar sounds
- Normalization and latency compensation

**Outputs**: Individual stems (`stems/{instrument}.wav`)

### 6. Mixing & Mastering 🎛️
Auto-mixing chain targeting style-specific metrics.

**Chain**: Per-stem EQ + compression + saturation → Bus compressor → Stereo widener → Limiter  
**Targets**: 
- LUFS: Rock (-9.5), R&B (-12.0), Country (-10.5)
- Spectral tilt and stereo width per style

**Outputs**: Balanced, mastered mix

### 7. Export & Analysis 📊
Saves final audio and generates quality report.

**Analysis**: LUFS, spectral centroid, stereo characteristics  
**Outputs**: `final.wav`, `report.json`

## Configuration

### Style Configs
Each style has dedicated configuration in `configs/`:

```yaml
# configs/rock_punk.yaml
style: rock_punk
duration_bars: 64
bpm: 140
key: E

instruments:
  - drums
  - bass_pick  
  - guitar_distorted
  - guitar_power_chords

mixing:
  lufs_target: -9.5
  spectral_tilt: "bright"
  stereo_width: 0.7
```

### Hydra Integration
Uses Hydra for configuration management with override support:

```bash
python run_pipeline.py --style rock_punk --config configs/rock_punk.yaml duration_bars=32 bpm=150
```

## Output Structure

```
exports/20241203_143022/
├── final.wav                 # Final mastered audio
├── report.json              # Analysis and pipeline metadata
├── arrangement.json         # Generated song structure  
├── composition_metadata.json # Input parameters
└── stems/                   # Individual instrument tracks
    ├── drums.wav
    ├── bass.wav
    ├── guitar.wav
    └── keys.wav
```

## Command Line Options

```bash
python run_pipeline.py [OPTIONS]

Required:
  --style {rock_punk|rnb_ballad|country_pop}    Music style

Optional:
  --duration_bars INT    Song length in bars
  --bpm INT             Beats per minute  
  --key STR             Musical key (C, D, E, F, G, A, B)
  --config PATH         YAML configuration file
  --verbose, -v         Detailed logging
```

## Dependencies

Core requirements in `requirements-pipeline.txt`:

```
hydra-core>=1.3.0       # Configuration management
torch>=2.0.0            # ML models
pytorch-lightning>=2.0.0 # Training framework
librosa>=0.10.0         # Audio analysis
mido>=1.2.10           # MIDI processing
omegaconf>=2.3.0       # Config merging
```

## Testing

```bash
# Test all styles
python test_pipeline.py

# Quick manual test
python run_pipeline.py --style rock_punk --duration_bars 8

# View examples and documentation
python demo_pipeline.py
```

## Architecture

### Modular Design
Each step is independent and swappable:
- Tokenizer: Converts MIDI ↔ discrete events
- Arrangement Transformer: Generates section sequences  
- Melody/Harmony Transformer: Creates multi-track MIDI
- Sound Design: MIDI → Audio with sample libraries
- Mixing: Differentiable audio processing chain

### Style Consistency
Style tokens propagate through all stages ensuring coherent output.

### Loose Coupling
Data flows via standardized JSON/MIDI formats. Modules can be developed and tested independently.

## Error Handling

- Graceful fallbacks for missing models/scripts
- Comprehensive error reporting in `error_report.json`
- Pipeline state tracking for debugging
- Temporary file cleanup

## Example Reports

```json
{
  "audio_analysis": {
    "duration_seconds": 85.3,
    "estimated_lufs": -9.7,
    "spectral_centroid_hz": 2547.3
  },
  "pipeline": {
    "success": true,
    "steps_completed": ["data_ingestion", "tokenization", "arrangement_generation", "melody_harmony_generation", "stem_rendering", "mixing_mastering", "report_generation"],
    "timing": {
      "arrangement_generation": 2.4,
      "melody_harmony_generation": 15.7,
      "stem_rendering": 8.3,
      "mixing_mastering": 3.1
    }
  }
}
```

## Integration

This pipeline integrates with the broader AI Music Composer system:
- Uses existing tokenizer and model implementations
- Leverages trained Transformer models
- Connects to sample libraries and audio processors
- Outputs compatible with DAW workflows

## Performance

Typical generation times on modern hardware:
- Rock Punk (64 bars): ~45s
- R&B Ballad (96 bars): ~65s  
- Country Pop (80 bars): ~55s

Times include arrangement generation, MIDI synthesis, audio rendering, and mixing.

## Troubleshooting

**Pipeline fails early**: Check style name and config file paths  
**No audio output**: Verify sample libraries and audio dependencies  
**LUFS targets not met**: Check mixing configuration and stem levels  
**Slow generation**: Use shorter durations for testing, check GPU availability

For detailed logs: `python run_pipeline.py --verbose --style {style}`