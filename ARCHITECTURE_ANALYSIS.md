# AI Music Composer - Senior ML/Audio Engineering Analysis

## 1. Current Architecture Summary

### Directory Structure & Key Modules
```
/workspaces/spark-template/
├── src/
│   ├── components/music/          # Core production modules
│   │   ├── DataFlowPipeline.tsx   # Pipeline visualization
│   │   ├── LyricGenerator.tsx     # NLP-based lyric generation
│   │   ├── SongStructurePlanner.tsx  # Arrangement generator (Markov/rule-based)
│   │   ├── ChordProgressionBuilder.tsx  # Harmonic progression AI
│   │   ├── MelodyHarmonyGenerator.tsx   # MIDI composition engine
│   │   ├── LyricAlignment.tsx     # Phoneme-to-melody alignment
│   │   ├── SoundDesignEngine.tsx  # Audio synthesis/sample selection
│   │   ├── MixingMasteringEngine.tsx  # DSP processing & mastering
│   │   └── CompositionHistory.tsx # Version control & persistence
│   ├── hooks/useKV.ts            # Persistent storage abstraction
│   ├── types/spark.d.ts          # Runtime API definitions
│   └── lib/utils.ts              # Utility functions
├── **MISSING: /data/**           # Training data, samples, presets
├── **MISSING: /configs/**        # Style-specific configurations
└── package.json                  # Dependencies (D3, Three.js, React ecosystem)
```

### Current Data Formats
- **KV Storage Keys**: `generated-lyrics`, `song-structures`, `melody-harmony-compositions`, `sound-designs`, `mix-masters`
- **Arrangement Output**: JSON with `{name, duration, bars, description, bpm, startTime}`
- **Mixing Engine**: Advanced DSP structures (EQ, compression, mastering chain)
- **Pipeline State**: React-based state management with `useKV` persistence

### Technology Stack
- **Frontend**: React 19 + TypeScript + Tailwind CSS
- **Visualization**: D3.js + Three.js
- **Audio**: Web Audio API capabilities
- **AI Integration**: Spark LLM API (`spark.llm()` with prompt templating)
- **Storage**: KV-based persistence (`spark.kv`)

## 2. Missing Pieces for Multi-Style Support

### Critical Gaps for {rock_punk, rnb_ballad, country_pop}

#### A. Configuration Management System
```
**MISSING: /src/configs/**
├── styles/
│   ├── rock_punk.json           # Tempo: 150-180 BPM, LUFS: -8 to -6
│   ├── rnb_ballad.json          # Tempo: 60-80 BPM, LUFS: -12 to -10
│   └── country_pop.json         # Tempo: 100-130 BPM, LUFS: -10 to -8
├── mastering/
│   ├── rock_punk_master.json    # Spectral tilt: +2dB/octave above 1kHz
│   ├── rnb_ballad_master.json   # Spectral tilt: flat response, wide stereo
│   └── country_pop_master.json  # Spectral tilt: +1dB/octave, natural width
└── instruments/
    ├── rock_punk_kit.json       # Distorted guitars, aggressive drums
    ├── rnb_ballad_kit.json      # Smooth synths, acoustic elements
    └── country_pop_kit.json     # Acoustic guitars, country percussion
```

#### B. Style-Aware Training Data
```
**MISSING: /src/data/**
├── chord_progressions/
│   ├── rock_punk_patterns.json  # i-VI-III-VII, power chords
│   ├── rnb_ballad_patterns.json # ii-V-I, extended chords
│   └── country_pop_patterns.json # I-V-vi-IV, open voicings
├── song_structures/
│   ├── rock_punk_templates.json # Verse-Chorus-Bridge, shorter songs
│   ├── rnb_ballad_templates.json # Extended intros, longer bridges
│   └── country_pop_templates.json # Story-driven verse structure
└── mixing_references/
    ├── lufs_targets.json        # Per-style loudness standards
    ├── spectral_curves.json     # EQ reference curves
    └── stereo_imaging.json      # MS ratio targets per style
```

#### C. Style Token Integration
- **Current State**: No style propagation between modules
- **Needed**: Style context passed through entire pipeline
- **Missing Integration**: Style-aware AI prompts, parameter selection

#### D. Audio Processing Extensions
- **Current**: Basic DSP parameter structures
- **Missing**: Style-specific plugin chains, reference mastering, spectral analysis

## 3. Data Flow Analysis

### Current Pipeline
```
[Style Selection] ❌ NOT IMPLEMENTED
        ↓
[Arrangement Generator] ✅ BASIC (no style awareness)
        ↓
[Melody/Harmony Gen] ✅ BASIC (generic MIDI output)
        ↓
[Lyric Alignment] ✅ PLACEHOLDER
        ↓
[Sound Design] ✅ PLACEHOLDER (no style-specific synthesis)
        ↓
[Mixing/Mastering] ✅ ADVANCED DSP (no style presets)
        ↓
[Final Track] ❌ NO AUDIO EXPORT
```

### Required Dependency Graph (ASCII)
```
Style Config ──┬─── Arrangement Generator
               │         │
               │         ▼
               ├─── Chord Progression ──┐
               │         │               │
               │         ▼               ▼
               ├─── Melody Generator ────┼─── Lyric Alignment
               │         │               │         │
               │         ▼               │         ▼
               ├─── Sound Design ────────┼─────────┼─── Mix Engine
               │         │               │         │         │
               │         ▼               │         │         ▼
               └─── Mastering Preset ────┴─────────┴─── Audio Export
                         │                               │
                         ▼                               ▼
                  Style Validation ──────────── Quality Check
```

## 4. Two-Week Development Roadmap

### Week 1: Foundation & Configuration System

#### **Ticket 1.1: Style Configuration Framework** (3 days)
**Priority**: P0 - Critical Path
**Acceptance Criteria**:
- [ ] Create `/src/configs/` directory structure
- [ ] Implement `StyleConfig` TypeScript interfaces
- [ ] Build style selection UI component with 3 target styles
- [ ] Add style context provider to React app
- [ ] Test style propagation through pipeline

**Technical Requirements**:
```typescript
interface StyleConfig {
  name: string;
  tempo: { min: number; max: number; default: number };
  lufs: { target: number; tolerance: number };
  spectralTilt: { slope: number; pivot: number };
  stereoWidth: { ms_ratio: number; bass_mono: boolean };
  chordTendencies: ChordProgression[];
  arrangementTemplates: ArrangementTemplate[];
}
```

#### **Ticket 1.2: Data Pipeline Refactoring** (2 days)
**Priority**: P0 - Critical Path
**Acceptance Criteria**:
- [ ] Modify all pipeline components to accept `StyleConfig`
- [ ] Update `useKV` storage to include style metadata
- [ ] Implement style consistency validation
- [ ] Add pipeline state debugging tools

#### **Ticket 1.3: Arrangement Generator Style Integration** (2 days)
**Priority**: P1 - High Impact
**Acceptance Criteria**:
- [ ] Style-aware song structure generation
- [ ] BPM selection based on style config
- [ ] Genre-appropriate section naming/descriptions
- [ ] Integration with LLM prompts for style-specific arrangements

### Week 2: Audio Processing & Export

#### **Ticket 2.1: Mixing Engine Style Presets** (3 days)
**Priority**: P0 - Critical Path
**Acceptance Criteria**:
- [ ] Implement style-specific EQ curves
- [ ] Add LUFS targeting per style
- [ ] Create spectral tilt processing
- [ ] Build MS stereo width control
- [ ] Validate against reference tracks

**Technical Implementation**:
```typescript
interface MasteringPreset {
  eq: { freq: number; gain: number; q: number }[];
  lufs_target: number;
  spectral_tilt: { slope: number; pivot_freq: number };
  stereo: { width: number; bass_mono_freq: number };
  limiter: { threshold: number; release: number };
}
```

#### **Ticket 2.2: Sound Design Style Engine** (2 days)
**Priority**: P1 - Feature Complete
**Acceptance Criteria**:
- [ ] Style-specific instrument selection
- [ ] Synthesizer parameter mapping per genre
- [ ] Audio sample library integration
- [ ] MIDI-to-audio rendering with style context

#### **Ticket 2.3: Audio Export & Quality Validation** (2 days)
**Priority**: P0 - Production Ready
**Acceptance Criteria**:
- [ ] WAV/MP3 export functionality
- [ ] LUFS measurement validation
- [ ] Spectral analysis display
- [ ] A/B comparison with reference tracks
- [ ] Automated quality checks per style

### Dependencies & Blockers

#### **External Dependencies**:
- Web Audio API sample loading (browser limitations)
- LLM prompt engineering for style-specific content
- Reference track licensing for quality validation

#### **Technical Risks**:
- Browser audio processing performance for real-time DSP
- LLM consistency for style-appropriate musical decisions
- Storage limitations for audio samples and presets

#### **Mitigation Strategies**:
- Implement Web Workers for heavy audio processing
- Create fallback static configurations if LLM fails
- Use compressed audio formats and lazy loading

### Success Metrics

#### **Week 1 Success Criteria**:
- [ ] 3 styles (rock_punk, rnb_ballad, country_pop) fully configured
- [ ] Style selection affects all pipeline components
- [ ] Data flows with style context preservation
- [ ] No regression in existing functionality

#### **Week 2 Success Criteria**:
- [ ] Audio export with measurable LUFS targets per style
- [ ] Spectral analysis shows appropriate tilt characteristics
- [ ] Stereo imaging meets MS ratio specifications
- [ ] End-to-end pipeline produces style-appropriate tracks

### Technical Debt & Future Considerations

#### **Immediate Technical Debt**:
- No error handling for LLM failures
- Missing audio processing optimization
- No unit tests for audio DSP functions
- Hard-coded configuration values

#### **Post-2-Week Roadmap**:
- ML model training on collected user data
- Advanced synthesis parameter learning
- Multi-track stem export functionality
- Collaborative features and sharing

## 5. Risk Assessment

### **High Risk Items**:
1. **Audio Processing Performance**: Browser limitations may require Web Worker implementation
2. **LLM Prompt Consistency**: Style-specific generation quality depends on prompt engineering
3. **Storage Scaling**: Audio data storage may exceed browser limits

### **Medium Risk Items**:
1. **User Experience**: Complex pipeline may overwhelm non-technical users
2. **Cross-Browser Compatibility**: Web Audio API variations across browsers
3. **Quality Validation**: Subjective style appropriateness difficult to measure

### **Mitigation Strategies**:
- Progressive enhancement with fallbacks
- Extensive user testing with target demographics
- Performance monitoring and optimization tooling