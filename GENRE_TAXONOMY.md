# Hierarchical Genre Taxonomy

## Overview

This document describes the hierarchical genre taxonomy implemented for the AI Music Composer system. The taxonomy consists of 13 parent genres, each with 3-6 sub-genres, providing comprehensive coverage of popular music styles.

## Structure

### Directory Layout
```
/configs/genres/<parent>.yaml          # Parent genre defaults
/configs/styles/<parent>/<child>.yaml  # Sub-genre overrides
/style_packs/<parent>/                 # Parent reference materials
├── refs_audio/                        # Audio reference files
├── refs_midi/                         # MIDI reference files
└── meta.json                          # Metadata and training info
/style_packs/<parent>/<child>/         # Sub-genre reference materials
├── refs_audio/                        # Audio reference files
├── refs_midi/                         # MIDI reference files
└── meta.json                          # Metadata and training info
```

## Parent Genres

### 1. Pop (pop)
**Description**: Mainstream pop music with catchy melodies, polished production, and radio-friendly appeal
- **BPM Range**: 85-135 (preferred: 120)
- **LUFS Target**: -11.0
- **Spectral Centroid**: 2200 Hz
- **Sub-genres**: pop_rock, synth_pop, dance_pop, indie_pop, electro_pop

### 2. Hip-Hop/Rap (hiphop_rap)
**Description**: Rhythm-centric urban music featuring rap vocals, strong beats, and characteristic production
- **BPM Range**: 70-140 (preferred: 95)
- **LUFS Target**: -12.0
- **Spectral Centroid**: 1600 Hz
- **Sub-genres**: trap, boom_bap, conscious_rap, drill, mumble_rap, old_school

### 3. R&B/Soul (rnb_soul)
**Description**: Smooth, soulful music featuring rich harmonies, expressive vocals, and sophisticated production
- **BPM Range**: 65-130 (preferred: 85)
- **LUFS Target**: -14.0
- **Spectral Centroid**: 1800 Hz
- **Sub-genres**: contemporary_rnb, neo_soul, classic_soul, funk_soul, gospel_influenced

### 4. Rock (rock)
**Description**: Guitar-driven music with strong rhythms, dynamic arrangements, and energetic performances
- **BPM Range**: 80-180 (preferred: 120)
- **LUFS Target**: -10.0
- **Spectral Centroid**: 2600 Hz
- **Sub-genres**: alternative_rock, hard_rock, punk_rock, indie_rock, classic_rock, progressive_rock

### 5. Country (country)
**Description**: Storytelling music featuring acoustic instruments, traditional American roots, and narrative vocals
- **BPM Range**: 70-140 (preferred: 100)
- **LUFS Target**: -12.0
- **Spectral Centroid**: 2000 Hz
- **Sub-genres**: country_pop, country_rock, bluegrass, outlaw_country, modern_country

### 6. Dance/EDM (dance_edm)
**Description**: Electronic dance music featuring synthesized sounds, strong beats, and club-oriented production
- **BPM Range**: 110-150 (preferred: 128)
- **LUFS Target**: -8.0
- **Spectral Centroid**: 2400 Hz
- **Sub-genres**: house, techno, trance, dubstep, future_bass, progressive_house

### 7. Latin (latin)
**Description**: Rhythmic music featuring Latin percussion, melodic traditions, and diverse regional influences
- **BPM Range**: 90-160 (preferred: 120)
- **LUFS Target**: -11.0
- **Spectral Centroid**: 2100 Hz
- **Sub-genres**: salsa, bachata, reggaeton, merengue, cumbia, latin_pop

### 8. Afro (afro)
**Description**: African-influenced music featuring polyrhythmic elements, traditional instruments, and contemporary fusion
- **BPM Range**: 85-130 (preferred: 105)
- **LUFS Target**: -12.0
- **Spectral Centroid**: 1900 Hz
- **Sub-genres**: afrobeat, afro_pop, highlife, amapiano, afro_house

### 9. Reggae/Dancehall (reggae_dancehall)
**Description**: Jamaican music featuring distinctive rhythm patterns, heavy bass, and island-influenced production
- **BPM Range**: 60-110 (preferred: 80)
- **LUFS Target**: -11.0
- **Spectral Centroid**: 1700 Hz
- **Sub-genres**: roots_reggae, dancehall, dub, ska, lovers_rock

### 10. K-Pop/J-Pop (kpop_jpop)
**Description**: Asian pop music featuring polished production, varied musical influences, and dynamic arrangements
- **BPM Range**: 90-140 (preferred: 115)
- **LUFS Target**: -10.0
- **Spectral Centroid**: 2300 Hz
- **Sub-genres**: kpop_dance, kpop_ballad, jpop_idol, jpop_rock, city_pop

### 11. Singer-Songwriter (singer_songwriter)
**Description**: Intimate, personal music featuring acoustic instruments, lyrical storytelling, and vocal-focused arrangements
- **BPM Range**: 60-120 (preferred: 85)
- **LUFS Target**: -16.0
- **Spectral Centroid**: 1600 Hz
- **Sub-genres**: folk_acoustic, indie_folk, contemporary_folk, coffee_house, acoustic_pop

### 12. Jazz-Influenced (jazz_influenced)
**Description**: Music featuring jazz harmony, improvisation elements, and sophisticated musical arrangements
- **BPM Range**: 60-160 (preferred: 110)
- **LUFS Target**: -18.0
- **Spectral Centroid**: 1900 Hz
- **Sub-genres**: smooth_jazz, jazz_fusion, contemporary_jazz, neo_soul_jazz, acid_jazz

### 13. Christian/Gospel (christian_gospel)
**Description**: Sacred music featuring spiritual themes, gospel influences, and uplifting arrangements
- **BPM Range**: 70-140 (preferred: 100)
- **LUFS Target**: -13.0
- **Spectral Centroid**: 2000 Hz
- **Sub-genres**: contemporary_christian, traditional_gospel, praise_worship, southern_gospel, urban_gospel

## Configuration Parameters

Each parent and sub-genre configuration includes:

### Musical Characteristics
- **BPM Range**: Minimum, maximum, and preferred tempos
- **Key Signatures**: Preferred major and minor keys
- **Time Signatures**: Primary and secondary time signatures
- **Chord Progressions**: Common progressions and complexity level
- **Song Structure**: Typical arrangement templates and section lengths

### Production Characteristics
- **Instrument Roles**: Required and optional instruments
- **Groove**: Swing amount, pocket feel, energy level, dynamics
- **Mix Targets**: LUFS, spectral centroid, stereo imaging
- **Frequency Balance**: Low/mid/high energy distribution
- **Mix Style**: Compression, saturation, enhancement levels

### Quality Assurance
- **Hook Strength**: Minimum catchiness threshold
- **Harmonic Stability**: Chord progression quality
- **Arrangement Contrast**: Dynamic variation requirements
- **Mix Quality**: Technical production standards
- **Style Matching**: Genre consistency requirements
- **Repetition Limits**: Maximum allowable repetition
- **Melodic Interest**: Minimum melody engagement
- **Timing Variance**: Acceptable timing flexibility

## Inheritance System

Sub-genres inherit all parameters from their parent genre and can override specific values:

```yaml
# Example: Pop Rock inherits from Pop
parent: "pop"
name: "pop_rock"

# Override only specific parameters
mix_targets:
  lufs: -10.0          # Louder than base pop (-11.0)
  spectral_centroid_hz: 2400  # Brighter than base pop (2200 Hz)

instruments:
  required: ["KICK", "SNARE", "BASS_PICK", "ACOUSTIC_STRUM", "LEAD"]
```

## Usage in AI Pipeline

1. **Style Selection**: User selects parent genre or specific sub-genre
2. **Parameter Loading**: System loads parent config + sub-genre overrides
3. **Reference Retrieval**: Relevant audio/MIDI references loaded from style packs
4. **Generation**: AI models use style parameters to guide composition
5. **Quality Assessment**: Generated content evaluated against QA thresholds

## Benefits

- **Consistency**: Standardized parameters ensure style authenticity
- **Flexibility**: Easy to add new genres or modify existing ones
- **Scalability**: Hierarchical structure reduces configuration redundancy
- **Quality Control**: Built-in QA thresholds maintain output standards
- **Extensibility**: New sub-genres inherit sensible defaults from parents