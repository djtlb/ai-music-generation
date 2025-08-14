# Sample Library Structure

This directory contains the sample library organized by instrument type and musical style.

## Directory Structure

```
samples/
├── drums/
│   ├── rock_punk/
│   │   ├── kick_punk.wav
│   │   ├── snare_punk.wav
│   │   └── hihat_punk.wav
│   ├── rnb_ballad/
│   │   ├── kick_soft.wav
│   │   ├── snare_brush.wav
│   │   └── hihat_closed.wav
│   └── country_pop/
│       ├── kick_country.wav
│       ├── snare_country.wav
│       └── hihat_open.wav
├── bass/
│   ├── rock_punk/
│   │   └── bass_pick_distorted.wav
│   ├── rnb_ballad/
│   │   └── bass_finger.wav
│   └── country_pop/
│       └── bass_country.wav
├── guitar/
│   ├── rock_punk/
│   │   ├── power_chord_e.wav
│   │   └── lead_guitar.wav
│   ├── rnb_ballad/
│   │   └── acoustic_fingerpick.wav
│   └── country_pop/
│       ├── acoustic_strum_bright.wav
│       └── lead_guitar_country.wav
├── piano/
│   ├── rock_punk/
│   │   └── piano_bright.wav
│   ├── rnb_ballad/
│   │   └── electric_piano_warm.wav
│   └── country_pop/
│       └── piano_bright_country.wav
└── synth/
    ├── rock_punk/
    │   └── lead_synth_aggressive.wav
    ├── rnb_ballad/
    │   └── lead_pad_warm.wav
    └── country_pop/
        └── synth_lead_bright.wav
```

## Sample Requirements

### Format Specifications
- **Sample Rate**: 48kHz (or higher)
- **Bit Depth**: 24-bit minimum
- **Format**: WAV (uncompressed)
- **Channels**: Mono or stereo (as appropriate for instrument)

### Naming Convention
- `{instrument}_{style_descriptor}.wav`
- Use lowercase with underscores
- Keep names descriptive but concise

### Content Guidelines

#### Drums
- Clean, dry samples without reverb/effects
- Multiple velocity layers recommended
- Root note mapping: Kick=C1 (36), Snare=D1 (38), Hi-hat=F#1 (42)

#### Bass
- Single note samples at multiple pitches or multi-sampling
- Clean DI signal preferred for maximum flexibility
- Root note typically E0 (28) for 4-string bass

#### Guitar
- Clean DI or lightly processed signals
- Chord samples should be recorded at specific voicings
- Lead samples should be single notes for pitch-shifting

#### Piano
- High-quality acoustic or electric piano samples
- Consider recording at multiple velocity layers
- Root note typically C3 (60)

#### Synth
- Clean oscillator sounds or light processing
- Good for lead lines and pad sounds
- Multiple octaves recommended

## Style Characteristics

### Rock/Punk
- **Drums**: Punchy, tight, aggressive attack
- **Bass**: Distorted, picked bass with attack
- **Guitar**: Power chords, overdriven lead tones
- **Piano**: Bright, percussive attack
- **Synth**: Aggressive lead sounds

### R&B Ballad
- **Drums**: Soft attack, brushed snares, subtle
- **Bass**: Fingered bass, warm and smooth
- **Guitar**: Clean acoustic fingerpicking
- **Piano**: Warm electric piano tones
- **Synth**: Lush pads and warm leads

### Country Pop
- **Drums**: Natural, open sound with good transients
- **Bass**: Clean, present but not aggressive
- **Guitar**: Bright acoustic strumming, twangy leads
- **Piano**: Bright acoustic piano
- **Synth**: Clean, supportive lead sounds

## Implementation Notes

1. **Sample Loading**: The render engine will cache samples in memory for performance
2. **Pitch Shifting**: Samples will be pitch-shifted using time-domain methods
3. **Velocity Mapping**: Multiple velocity layers can be defined in instrument configs
4. **Effects Processing**: Effects are applied post-sample in the rendering pipeline
5. **Latency Compensation**: All samples should be trimmed to minimize pre-delay

## Adding New Samples

1. Place sample files in appropriate directory structure
2. Update instrument configuration YAML files
3. Ensure sample paths match configuration
4. Test rendering with sample MIDI files
5. Validate audio quality and timing

## Licensing

Ensure all samples are either:
- Original recordings
- Licensed for commercial use
- Public domain
- Creative Commons with appropriate attribution

Document licensing information for each sample set.