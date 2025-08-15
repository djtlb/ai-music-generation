# Genre Taxonomy Implementation

## Quick Start

The hierarchical genre taxonomy provides a comprehensive framework for AI music generation with 13 parent genres and 3-6 sub-genres each.

### Basic Usage

```python
# Load a parent genre configuration
from pathlib import Path
import yaml

def load_genre_config(genre_name):
    config_path = Path(f"configs/genres/{genre_name}.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Load a sub-genre configuration (inherits from parent)
def load_style_config(parent_genre, sub_genre):
    parent_config = load_genre_config(parent_genre)
    
    style_path = Path(f"configs/styles/{parent_genre}/{sub_genre}.yaml")
    if style_path.exists():
        with open(style_path, 'r') as f:
            style_overrides = yaml.safe_load(f)
        
        # Merge parent config with sub-genre overrides
        merged_config = {**parent_config}
        for key, value in style_overrides.items():
            if key != 'parent':  # Skip inheritance marker
                if isinstance(value, dict) and key in merged_config:
                    merged_config[key].update(value)
                else:
                    merged_config[key] = value
        
        return merged_config
    else:
        return parent_config

# Example usage
pop_config = load_genre_config("pop")
pop_rock_config = load_style_config("pop", "pop_rock")
```

## Configuration Structure

### Parent Genre Configuration

Each parent genre file (`/configs/genres/<genre>.yaml`) contains:

```yaml
name: "pop"
display_name: "Pop"
description: "Mainstream pop music with catchy melodies..."

# Tempo characteristics
bpm:
  min: 85
  max: 135
  preferred: 120

# Musical elements
keys:
  major: ["C", "G", "D", "A", "F", "Bb", "Eb"]
  minor: ["Am", "Em", "Bm", "F#m", "Dm", "Gm", "Cm"]
  preferred: ["C", "G", "Am", "F"]

# Instrumentation
instruments:
  required: ["KICK", "SNARE", "BASS_PICK", "PIANO", "LEAD"]
  optional: ["ACOUSTIC_STRUM", "PAD", "STRINGS", "BRASS"]

# Production targets
mix_targets:
  lufs: -11.0
  true_peak_db: -1.0
  spectral_centroid_hz: 2200

# Quality thresholds
qa_thresholds:
  min_hook_strength: 0.7
  min_harmonic_stability: 0.8
  min_mix_quality: 0.8

# Available sub-genres
sub_genres:
  - "pop_rock"
  - "synth_pop"
  - "dance_pop"
  - "indie_pop"
  - "electro_pop"
```

### Sub-Genre Configuration

Sub-genre files (`/configs/styles/<parent>/<child>.yaml`) inherit from parent and override specific parameters:

```yaml
parent: "pop"
name: "pop_rock"
display_name: "Pop Rock"
description: "Pop music with rock instrumentation..."

# Override specific parameters
bpm:
  min: 100      # More energetic than base pop
  max: 140
  preferred: 125

mix_targets:
  lufs: -10.0   # Louder than base pop (-11.0)
  spectral_centroid_hz: 2400  # Brighter than base pop (2200 Hz)

instruments:
  required: ["KICK", "SNARE", "BASS_PICK", "ACOUSTIC_STRUM", "LEAD"]
```

### Style Pack Metadata

Each style pack includes a `meta.json` file with training data information:

```json
{
  "genre": "pop",
  "display_name": "Pop",
  "description": "Mainstream pop music...",
  "reference_count": {
    "audio": 50,
    "midi": 100
  },
  "characteristics": {
    "tempo_range": "85-135 BPM",
    "key_preference": ["C", "G", "Am", "F"],
    "typical_instruments": ["vocals", "piano", "guitar", "bass", "drums"]
  },
  "training_data": {
    "audio_sources": ["Billboard Hot 100 (2015-2023)", "Radio singles"],
    "midi_sources": ["Professional MIDI transcriptions", "Hook-focused arrangements"]
  },
  "quality_metrics": {
    "min_hook_strength": 0.7,
    "target_lufs": -11.0
  },
  "sub_genres": ["pop_rock", "synth_pop", "dance_pop", "indie_pop", "electro_pop"]
}
```

## Implemented Genres

### Complete Parent Genres (13)
- ✅ Pop (pop)
- ✅ Hip-Hop/Rap (hiphop_rap)  
- ✅ R&B/Soul (rnb_soul)
- ✅ Rock (rock)
- ✅ Country (country)
- ✅ Dance/EDM (dance_edm)
- ✅ Latin (latin)
- ✅ Afro (afro)
- ✅ Reggae/Dancehall (reggae_dancehall)
- ✅ K-Pop/J-Pop (kpop_jpop)
- ✅ Singer-Songwriter (singer_songwriter)
- ✅ Jazz-Influenced (jazz_influenced)
- ✅ Christian/Gospel (christian_gospel)

### Example Sub-Genre Implementations
- ✅ Pop Rock (pop/pop_rock)
- ✅ Synth Pop (pop/synth_pop)
- ✅ Dance Pop (pop/dance_pop)
- ✅ Indie Pop (pop/indie_pop)
- ✅ Electro Pop (pop/electro_pop)
- ✅ Trap (hiphop_rap/trap)
- ✅ Boom Bap (hiphop_rap/boom_bap)
- ✅ Contemporary R&B (rnb_soul/contemporary_rnb)
- ✅ House (dance_edm/house)
- ✅ Alternative Rock (rock/alternative_rock)
- ✅ Country Pop (country/country_pop)
- ✅ Reggaeton (latin/reggaeton)
- ✅ K-Pop Dance (kpop_jpop/kpop_dance)

## Key Parameters

### Tempo & Rhythm
- **BPM Range**: Minimum, maximum, and preferred tempos
- **Groove**: Swing amount, pocket feel, energy level
- **Time Signatures**: Primary and secondary options

### Harmony & Structure
- **Key Signatures**: Preferred major/minor keys
- **Chord Progressions**: Common progressions and complexity
- **Song Structure**: Typical arrangements and section lengths

### Production & Mix
- **Instrument Roles**: Required and optional instruments
- **Mix Targets**: LUFS, spectral centroid, stereo imaging
- **Frequency Balance**: Low/mid/high energy distribution
- **Dynamic Range**: Crest factor and RMS targets

### Quality Control
- **Hook Strength**: Catchiness requirements
- **Harmonic Stability**: Chord progression quality
- **Mix Quality**: Technical production standards
- **Style Matching**: Genre consistency requirements

## Usage in AI Pipeline

1. **Style Selection**: User chooses parent genre or specific sub-genre
2. **Config Loading**: System loads and merges parent + sub-genre configs
3. **Parameter Application**: AI models use style parameters for generation
4. **Reference Retrieval**: Relevant training data loaded from style packs
5. **Quality Assessment**: Output evaluated against genre-specific thresholds

## Extending the Taxonomy

### Adding New Parent Genres

1. Create `/configs/genres/<new_genre>.yaml` with all required parameters
2. Create `/style_packs/<new_genre>/` directory structure
3. Add reference materials and `meta.json`
4. Define 3-6 sub-genres in the parent config

### Adding New Sub-Genres

1. Create `/configs/styles/<parent>/<new_subgenre>.yaml`
2. Set `parent: "<parent_genre>"`
3. Override specific parameters as needed
4. Create corresponding style pack directory
5. Add to parent genre's `sub_genres` list

### Validation

Run the validation script to ensure proper configuration:

```bash
python3 scripts/validate_taxonomy.py
```

This checks for:
- YAML/JSON syntax errors
- Required field presence
- Proper inheritance structure
- Style pack completeness

## Benefits

- **Consistency**: Standardized parameters ensure authentic style generation
- **Flexibility**: Easy customization through inheritance system
- **Scalability**: New genres/styles easily added with minimal configuration
- **Quality**: Built-in QA thresholds maintain output standards
- **Efficiency**: Inheritance reduces configuration redundancy