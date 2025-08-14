# Auto-Mixing & Mastering Engine

A differentiable mixing chain that automatically predicts optimal mixing parameters to achieve target LUFS, spectral characteristics, and stereo imaging for different musical styles.

## Overview

This system implements a complete PyTorch-based auto-mixing pipeline that:

1. **Extracts features** from audio stems (RMS, crest factor, spectral centroid, dynamic range)
2. **Predicts mixing parameters** using an MLP trained on style-specific targets
3. **Processes audio** through differentiable EQ, compression, and mastering chains
4. **Validates output** against target LUFS, spectral centroid, and M/S ratio specifications

## Architecture

```
Audio Stems → Feature Extraction → Parameter Prediction → Processing Chain → Analysis
              (RMS, Crest, etc.)   (MLP Network)        (Diff. DSP)     (LUFS, etc.)
```

### Core Components

- **`StemFeatureExtractor`**: Analyzes audio characteristics
- **`MixingParameterPredictor`**: MLP that predicts optimal parameters
- **`ChannelStrip`**: Per-stem EQ, compression, and saturation
- **`MasteringChain`**: Bus compression, stereo enhancement, limiting
- **`AutoMixChain`**: Complete end-to-end processing pipeline

### Style Targets

| Style | LUFS | Spectral Centroid | M/S Ratio | Characteristics |
|-------|------|------------------|-----------|-----------------|
| `rock_punk` | -9.5 dB | 2800 Hz | 0.6 | Aggressive, bright, compressed |
| `rnb_ballad` | -12.0 dB | 1800 Hz | 0.8 | Warm, dynamic, wide stereo |
| `country_pop` | -10.5 dB | 2200 Hz | 0.7 | Clear, balanced, radio-ready |

## Usage

### CLI Tool

```bash
# Mix stems with target style
python mix_master.py --stems track1.wav track2.wav track3.wav --style rock_punk --output mixed.wav

# Generate test mix with validation
python mix_master.py --test-stems 4 --style rnb_ballad --output test_mix.wav --validate

# Use custom style targets
python mix_master.py --stems *.wav --style country_pop --targets custom_targets.yaml --output final.wav
```

### Python API

```python
from mix import AutoMixChain, load_style_targets
import torch

# Load stems
stems = [torch.load(f"stem_{i}.pt") for i in range(4)]

# Create auto-mix chain
targets = load_style_targets()
auto_mix = AutoMixChain(n_stems=4, style_targets=targets)

# Process stems
mixed_audio, analysis = auto_mix(stems, style='rock_punk')

print(f"Target LUFS: {targets['rock_punk']['lufs']}")
print(f"Actual LUFS: {analysis['lufs']:.1f}")
print(f"LUFS Error: {analysis['lufs_error']:.1f} dB")
```

### React Integration

The mixing engine integrates with the React UI to provide:
- AI-generated mixing parameters
- Real-time target compliance visualization  
- Style-specific parameter optimization
- Export functionality for DAW integration

## Validation

### Test Suite

Run comprehensive validation:

```bash
python validate_mix.py --verbose
```

This runs tests with:
- White noise stems (basic functionality)
- Frequency-specific content (bass-heavy, bright)
- Dynamic range variations (quiet, loud)
- Cross-style validation

### Unit Tests

```bash
python test_auto_mix.py
```

Tests individual components:
- Feature extraction accuracy
- Parameter prediction ranges
- Audio processing integrity
- Target compliance validation

## File Structure

```
mix/
├── __init__.py              # Package exports
├── auto_mix.py              # Core mixing components
└── utils.py                 # Analysis and validation utilities

configs/
└── style_targets.yaml       # Style-specific targets and processing chains

# CLI Tools
mix_master.py                # Main mixing CLI
validate_mix.py              # Validation suite
test_auto_mix.py             # Unit tests
```

## Technical Details

### Differentiable Processing

All audio processing uses PyTorch tensors for end-to-end differentiability:

- **EQ**: Spectral domain processing with frequency masks
- **Compression**: Simplified envelope following with attack/release
- **Saturation**: Hyperbolic tangent soft clipping
- **Limiting**: Adaptive gain reduction
- **Stereo Processing**: M/S matrix operations

### Parameter Prediction

The MLP predictor uses:
- **Input**: Concatenated features from all stems (4 features × 8 stems = 32D)
- **Architecture**: 256→256→128→output fully connected layers
- **Output**: Structured parameters for all processing stages
- **Constraints**: Parameters scaled to realistic ranges (dB, ratios, times)

### Loss Functions

Training targets minimize:
- **LUFS Error**: MSE between predicted and target loudness
- **Spectral Error**: MSE between predicted and target centroid (scaled)
- **Stereo Error**: MSE between predicted and target M/S ratio

## Style Configuration

Styles are defined in `configs/style_targets.yaml` with:

```yaml
rock_punk:
  lufs: -9.5
  spectral_centroid_hz: 2800
  stereo_ms_ratio: 0.6
  frequency_balance:
    low_energy: 0.3
    mid_energy: 0.5  
    high_energy: 0.4
  processing_chains:
    compression:
      threshold: -12
      ratio: 4.0
      attack: 3
      release: 100
```

## Performance

- **Processing Speed**: ~2-3x real-time for 8-stem mixes
- **Memory Usage**: ~500MB for 10-second, 8-stem processing
- **Accuracy**: Typical LUFS error <1dB, spectral error <200Hz

## Integration with Music Pipeline

The auto-mixing engine fits into the complete music production pipeline:

```
Arrangement → Melody/Harmony → Sound Design → **Auto-Mixing** → Final Track
```

It receives:
- **Sound Design**: Instrument patches and synthesis parameters
- **Composition**: MIDI tracks with arrangement structure

And produces:
- **Mixed Audio**: Professional-quality stereo mix
- **Analysis**: Target compliance metrics and quality scores
- **Parameters**: Exportable settings for DAW integration

## Future Enhancements

- **Neural Audio Codecs**: Integration with RAVE/DDSP for higher-quality processing
- **Style Transfer**: Cross-style mixing with interpolated targets  
- **Real-time Processing**: Optimized inference for live mixing
- **Advanced Modeling**: Transformer-based parameter prediction
- **Multi-objective Optimization**: Pareto-optimal parameter selection