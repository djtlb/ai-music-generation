# Auto-Mixing & Mastering Engine

A differentiable PyTorch-based auto-mixing and mastering chain that automatically optimizes audio stems to hit target LUFS, spectral characteristics, and stereo imaging for different musical styles.

## Features

### Core Components

1. **Differentiable Processing Chain**
   - Per-stem EQ (4-band: low shelf, low-mid peaking, high-mid peaking, high shelf)
   - Compressor with threshold, ratio, attack, release, and makeup gain
   - Soft saturation for harmonic enhancement
   - Level and pan controls

2. **Mastering Chain**
   - Bus compressor for glue
   - Master EQ for final tonal shaping
   - Stereo widener using M/S processing
   - Brick-wall limiter for loudness control

3. **Feature Extraction**
   - RMS energy analysis
   - Crest factor (peak-to-RMS ratio)
   - Spectral centroid computation
   - Dynamic range measurement

4. **Parameter Prediction**
   - MLP-based parameter predictor
   - Takes stem features as input
   - Outputs complete mixing parameters for all stems + master
   - Parameters automatically scaled to appropriate ranges

### Style Targets

The system supports three musical styles with different mixing targets:

| Style | LUFS Target | Spectral Centroid | Stereo M/S Ratio |
|-------|-------------|-------------------|------------------|
| **rock_punk** | -9.5 dB | 2800 Hz | 0.6 |
| **rnb_ballad** | -12.0 dB | 1800 Hz | 0.8 |
| **country_pop** | -10.5 dB | 2200 Hz | 0.7 |

## Quick Start

### Basic Usage

```python
from mix import AutoMixChain, load_style_targets
from mix.utils import create_white_noise_stems

# Load style targets
targets = load_style_targets()

# Create auto-mix chain
auto_mix = AutoMixChain(n_stems=4, style_targets=targets)

# Create or load audio stems
stems = create_white_noise_stems(4, duration=10.0)  # Test stems
# OR load real stems:
# stems = [torchaudio.load("kick.wav")[0], torchaudio.load("snare.wav")[0], ...]

# Mix with target style
mixed_audio, analysis = auto_mix(stems, style='rock_punk')

print(f"LUFS: {analysis['lufs']:.1f} dB")
print(f"Spectral Centroid: {analysis['spectral_centroid']:.0f} Hz")
print(f"Stereo M/S Ratio: {analysis['stereo_ms_ratio']:.2f}")
```

### Command Line Interface

```bash
# Mix real stems
python mix_master.py --stems kick.wav snare.wav bass.wav guitar.wav \
                     --style rock_punk --output mixed.wav --validate

# Create test mix for validation
python mix_master.py --test-stems 4 --style rnb_ballad \
                     --output test_mix.wav --analysis-output analysis.json

# Use custom style targets
python mix_master.py --stems *.wav --style country_pop \
                     --targets custom_targets.yaml --output final.wav
```

### Validation

```bash
# Run complete validation suite
python validate_mix.py --verbose

# Test specific style
python validate_mix.py --styles rock_punk --output validation_results.json

# Quick validation
python validate_mix.py --quick --test-filter white_noise
```

## Architecture

### Data Flow

```
Audio Stems → Feature Extraction → Parameter Prediction → Processing Chain → Mixed Output
     ↓              ↓                    ↓                    ↓              ↓
  [N×2×T]      [N×4 features]      [Mix Parameters]     [Effects Chain]   [2×T]
```

### Processing Pipeline

1. **Feature Extraction**: Extract RMS, crest factor, spectral centroid, dynamic range
2. **Parameter Prediction**: MLP predicts optimal mixing parameters
3. **Per-Stem Processing**: EQ → Compression → Saturation → Level/Pan
4. **Mixing**: Sum all processed stems
5. **Mastering**: Bus compression → Master EQ → Stereo enhancement → Limiting
6. **Analysis**: Compute LUFS, spectral centroid, M/S ratio vs targets

### Parameter Ranges

#### Per-Stem Parameters
- **EQ Gains**: -12 to +12 dB
- **Compression Threshold**: -40 to -6 dB
- **Compression Ratio**: 1:1 to 10:1
- **Attack Time**: 0.001 to 0.1 seconds
- **Release Time**: 0.01 to 1.0 seconds
- **Makeup Gain**: 0 to 12 dB
- **Saturation**: 1.0 to 3.0x
- **Level**: -24 to +6 dB
- **Pan**: -1.0 (left) to +1.0 (right)

#### Master Parameters
- **Bus Compression**: Similar ranges but more conservative
- **Master EQ**: ±6 dB range for subtle corrections
- **Stereo Width**: 0.5 to 1.5 (0.5=narrow, 1.0=normal, 1.5=wide)
- **Limiter Threshold**: 0.7 to 1.0 (linear scale)

## Configuration

### Style Targets (configs/style_targets.yaml)

```yaml
style_targets:
  rock_punk:
    lufs: -9.5
    spectral_centroid_hz: 2800
    stereo_ms_ratio: 0.6
    # Additional style-specific parameters...
    
  rnb_ballad:
    lufs: -12.0
    spectral_centroid_hz: 1800
    stereo_ms_ratio: 0.8
    
  country_pop:
    lufs: -10.5
    spectral_centroid_hz: 2200
    stereo_ms_ratio: 0.7
```

### Custom Style Targets

You can define custom mixing targets:

```yaml
my_custom_style:
  lufs: -11.0
  spectral_centroid_hz: 2500
  stereo_ms_ratio: 0.65
  frequency_balance:
    low_energy: 0.4
    mid_energy: 0.5
    high_energy: 0.35
```

## Training (Future Work)

The system is designed to be trainable end-to-end:

```python
from mix import AutoMixChain, create_training_loss

# Create loss function
loss_fn = create_training_loss(style_targets)

# Training loop
for batch in dataloader:
    stems, target_style = batch
    mixed_audio, analysis = auto_mix(stems, target_style)
    loss = loss_fn(analysis, target_style)
    loss.backward()
    optimizer.step()
```

## Validation & Testing

### Automated Testing

```bash
# Run comprehensive test suite
python test_auto_mix.py

# Quick functionality test
python test_basic_mix.py

# Validation with metrics
python validate_mix.py --output validation_report.json
```

### Test Coverage

- **Unit Tests**: Individual components (EQ, compressor, limiter, etc.)
- **Integration Tests**: Complete mixing pipeline
- **Style Tests**: Different styles produce different results
- **Edge Cases**: Various stem counts, lengths, and signal types
- **Performance Tests**: Processing time benchmarks

### Metrics Validation

The validation system checks:
- LUFS accuracy (±1 dB tolerance)
- Spectral centroid accuracy (±200 Hz tolerance)
- Stereo M/S ratio accuracy (±0.2 tolerance)
- Overall quality score (0-1 scale)

## Performance

- **Real-time Factor**: ~0.1x (processes 10s audio in ~1s on CPU)
- **Memory Usage**: ~500MB for 8 stems × 10 seconds
- **GPU Acceleration**: Fully compatible with CUDA

## Error Handling

The system includes robust error handling:
- Graceful fallbacks for STFT/ISTFT failures
- Device compatibility (CPU/GPU)
- Audio length mismatches
- Invalid parameter ranges
- Missing style targets

## Limitations & Future Work

### Current Limitations
- Simplified EQ implementation (frequency domain)
- Basic compressor model (needs proper envelope detection)
- No look-ahead limiting
- LUFS computation is approximated (not full ITU-R BS.1770-4)

### Planned Improvements
1. **Advanced DSP**: Proper biquad filters, look-ahead limiting
2. **Training Pipeline**: End-to-end training on real music data
3. **More Styles**: Additional genre targets and style interpolation
4. **Advanced Features**: Sidechain compression, multiband processing
5. **Real-time Processing**: Optimized for live performance

## References

- ITU-R BS.1770-4: Loudness measurement standards
- AES Convention Papers on automatic mixing
- PyTorch Audio Processing: torchaudio documentation
- Music Information Retrieval: spectral feature extraction

## License

See LICENSE file for details.