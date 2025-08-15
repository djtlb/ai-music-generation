# Music Planner Module

The `planner.py` module converts free-text lyrics and genre descriptions into structured control JSON that drives the entire music generation pipeline.

## Overview

**Input**: 
- `lyrics_text`: Free-text lyrics or lyrics description
- `genre_text`: Free-text genre description (e.g., "pop dance, 124 bpm, bright")

**Output**: 
Control JSON with fields:
- `style`: Detected music style (e.g., "pop/dance_pop")
- `bpm`: Tempo in beats per minute
- `time_feel`: Rhythm feel (straight, halftime, swing, etc.)
- `key`: Key signature (e.g., "C", "Am") 
- `arrangement`: Song structure and section lengths
- `drum_template`: Drum pattern template name
- `hook_type`: Type of musical hook (chorus_hook, prechorus_hook, etc.)
- `mix_targets`: Mixing/mastering targets (LUFS, spectral centroid, etc.)
- `lyrics_sections`: Parsed lyrics with section markers

## Key Features

### Intelligent Text Parsing
- **Genre Detection**: Uses regex patterns to detect styles from natural language
- **BPM Extraction**: Finds explicit BPM mentions in genre text
- **Key Signature**: Parses key signatures (e.g., "in D minor")
- **Time Feel**: Detects rhythm patterns (halftime, swing, etc.)

### Lyrics Structure Analysis
- **Section Detection**: Identifies VERSE, CHORUS, BRIDGE, etc. from lyrics
- **Structure Inference**: Generates song structure when not explicitly marked
- **Hook Type**: Determines the primary hook strategy
- **Content Parsing**: Extracts and organizes lyrical content by section

### Configuration System
- **Hierarchical Configs**: Parent genres with child style overrides
- **Style Inheritance**: Child styles inherit from parent with selective overrides
- **Default Values**: Sensible defaults for all musical parameters
- **Mix Targets**: Style-specific mixing/mastering targets

### T5 Integration (Optional)
- **Missing Field Prediction**: Uses T5 model to predict missing parameters
- **Fallback Heuristics**: Rule-based fallbacks when T5 unavailable
- **Smart Defaults**: Context-aware default value selection

## Usage

### Basic Usage

```python
from planner import MusicPlanner

planner = MusicPlanner("configs")

lyrics = """
VERSE: Walking down the street tonight
CHORUS: We can fly, we can touch the sky
"""

genre = "dance pop, 128 bpm, four on the floor"

control_json = planner.plan(lyrics, genre)
print(control_json['style'])  # "pop/dance_pop"
print(control_json['bpm'])    # 128
print(control_json['drum_template'])  # "four_on_floor"
```

### Command Line Interface

```bash
python planner.py --lyrics "VERSE: test lyrics" --genre "pop rock, 120 bpm" --output control.json
```

### Integration with Pipeline

```python
# Step 1: Plan
control = planner.plan(user_lyrics, user_genre)

# Step 2: Feed to other modules
tokenizer.encode(control['style'], control['bpm'], control['key'])
arranger.generate(control['arrangement']['structure'])
melody_gen.create(control['key'], control['chord_progressions'])
mixer.apply_targets(control['mix_targets'])
```

## Configuration Format

### Parent Genre Config (`configs/genres/pop.yaml`)

```yaml
name: "pop"
display_name: "Pop"
description: "Mainstream pop music"

bpm:
  min: 85
  max: 135
  preferred: 120

keys:
  major: ["C", "G", "D", "A", "F"]
  minor: ["Am", "Em", "Bm", "F#m", "Dm"]
  preferred: ["C", "G", "Am", "F"]

instruments:
  required: ["KICK", "SNARE", "BASS_PICK", "PIANO", "LEAD"]
  optional: ["ACOUSTIC_STRUM", "PAD", "STRINGS"]

structure:
  common_forms:
    - ["INTRO", "VERSE", "CHORUS", "VERSE", "CHORUS", "OUTRO"]
  section_lengths:
    INTRO: 8
    VERSE: 16
    CHORUS: 16
    OUTRO: 8

mix_targets:
  lufs: -11.0
  spectral_centroid_hz: 2200
  stereo_ms_ratio: 0.7

groove:
  swing: 0.0
  pocket: "tight"
  energy: "high"
```

### Child Style Config (`configs/styles/pop/dance_pop.yaml`)

```yaml
parent: "pop"
name: "dance_pop"
display_name: "Dance Pop"

# Override BPM for dance floor
bpm:
  min: 110
  max: 135
  preferred: 128

# Override mix targets for club sound
mix_targets:
  lufs: -8.5  # Louder for clubs
  spectral_centroid_hz: 2500
```

## Pattern Recognition

### Genre Detection Patterns

The planner uses regex patterns to detect styles:

```python
style_patterns = {
    'pop/dance_pop': r'\b(dance[\s\-]?pop|edm[\s\-]?pop|club[\s\-]?pop)\b',
    'rock/punk': r'\b(punk|rock[\s\-]?punk|punk[\s\-]?rock)\b',
    'hiphop_rap/drill': r'\b(drill|trap[\s\-]?drill|uk[\s\-]?drill)\b',
    # ... more patterns
}
```

### Lyrics Section Patterns

```python
section_patterns = {
    'INTRO': r'\b(intro|introduction)\b[:\-]?',
    'VERSE': r'\b(verse|v\d+)\b[:\-]?',
    'CHORUS': r'\b(chorus|hook|refrain)\b[:\-]?',
    'BRIDGE': r'\b(bridge|middle[\s\-]?8)\b[:\-]?',
    # ... more patterns
}
```

## Test Examples

### Pop Dance Track
**Input**: 
- Lyrics: "VERSE: Dancing tonight CHORUS: We can fly"
- Genre: "dance pop, 128 bpm, four on the floor"

**Output**:
```json
{
  "style": "pop/dance_pop",
  "bpm": 128,
  "time_feel": "straight",
  "key": "C",
  "drum_template": "four_on_floor",
  "hook_type": "chorus_hook",
  "arrangement": {
    "structure": ["VERSE", "CHORUS"],
    "total_bars": 32
  }
}
```

### Rock Anthem
**Input**:
- Lyrics: "VERSE: Standing strong CHORUS: We are champions"  
- Genre: "rock anthem, 140 bpm, powerful"

**Output**:
```json
{
  "style": "rock",
  "bpm": 140,
  "key": "E",
  "drum_template": "rock_basic",
  "hook_type": "chorus_hook",
  "mix_targets": {
    "lufs": -9.5,
    "spectral_centroid_hz": 2800
  }
}
```

## Error Handling & Validation

### BPM Constraints
- Minimum: 60 BPM
- Maximum: 200 BPM
- Auto-correction for out-of-range values

### Required Fields
- All outputs guaranteed to have essential fields
- Fallback defaults for missing configurations
- Graceful degradation when T5 unavailable

### Input Validation
- Handles empty inputs gracefully
- Truncates overly long inputs
- Sanitizes special characters

## Testing

### Unit Tests
```bash
python test_planner.py
```

Includes tests for:
- Basic genre detection
- BPM and key extraction
- Lyrics structure parsing
- Configuration loading
- Edge cases and error handling
- Output stability across runs
- JSON serializability

### Integration Tests
```bash
python test_planner_cli.py
```

### Demo Scripts
```bash
python planner_integration_demo.py
```

## Dependencies

### Required
- `PyYAML>=6.0`: Configuration file parsing
- `numpy>=1.21.0`: Numerical operations

### Optional  
- `transformers>=4.30.0`: T5 model for field prediction
- `torch>=2.0.0`: PyTorch backend for T5

### Development
- `pytest>=7.0.0`: Testing framework

## Architecture

The planner follows a multi-stage processing pipeline:

1. **Text Parsing**: Extract explicit parameters from text
2. **Configuration Loading**: Load hierarchical style configs  
3. **Intelligent Merging**: Combine parsed data with config defaults
4. **Missing Field Prediction**: Use T5 or heuristics for gaps
5. **Validation & Constraints**: Ensure output validity
6. **JSON Generation**: Produce final control structure

This design provides:
- **Flexibility**: Handles varied text input styles
- **Consistency**: Stable outputs for same inputs
- **Extensibility**: Easy to add new genres/patterns
- **Reliability**: Fallbacks for all failure modes
- **Integration**: Clean interface with pipeline modules