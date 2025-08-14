# Melody & Harmony Transformer

A style-conditioned Transformer model for generating melody and harmony sequences with musical constraints and quality control.

## Overview

This implementation provides a complete solution for generating musically coherent melody and harmony sequences using a Transformer decoder architecture with:

- **Style Conditioning**: Supports rock_punk, rnb_ballad, and country_pop styles
- **Musical Constraints**: Scale compatibility, chord constraints, and repetition control
- **Auxiliary Losses**: In-scale penalty, chord compatibility, and repetition penalty
- **Advanced Sampling**: Temperature control by section and nucleus sampling

## Architecture

### Core Components

#### 1. MelodyHarmonyTransformer (`src/models/mh_transformer.py`)
- **Main Model**: Transformer decoder with style conditioning
- **Style Embedding**: Combines style, key, section, and optional groove features
- **Positional Encoding**: Standard sinusoidal position embeddings
- **Auxiliary Heads**: Chord compatibility and scale compatibility prediction
- **Generation**: Supports constrained generation with nucleus sampling

#### 2. Constraint System (`src/utils/constraints.py`)
- **Musical Constraints**: Scale notes, chord intervals, style preferences
- **Constraint Masks**: Dynamic masking for scale, chord, and style compliance
- **Repetition Controller**: Token and phrase-level repetition prevention
- **Combined Masking**: Unified constraint application during generation

#### 3. Training Loss (`src/models/mh_transformer.py`)
- **Cross-Entropy**: Main language modeling loss
- **Scale Penalty**: Encourages in-scale note generation
- **Chord Compatibility**: Auxiliary loss for chord-tone alignment
- **Repetition Penalty**: Reduces excessive pattern repetition

## Key Features

### Musical Intelligence
- **Scale Awareness**: Enforces key signature compliance
- **Chord Compatibility**: Generates notes that fit harmonic context
- **Style Conditioning**: Adapts to rock, R&B, and country characteristics
- **Repetition Control**: Prevents monotonous patterns

### Generation Control
- **Section-Specific Temperature**: Different creativity levels per song section
- **Nucleus Sampling**: Top-p sampling for quality control
- **Constraint Masking**: Real-time musical rule enforcement
- **Groove Conditioning**: Optional drum pattern influence

### Training Features
- **Teacher Forcing**: Standard autoregressive training
- **Coverage Penalty**: Prevents attention collapse
- **Multi-Task Learning**: Joint melody and harmony optimization
- **Data Augmentation**: Key transposition and tempo variation

## Usage

### Training

```bash
# Train the model
python train_mh.py \
    --config configs/mh_transformer.yaml \
    --output-dir ./outputs/mh_training \
    --wandb-project melody-harmony
```

### Sampling

```bash
# Generate samples
python sample_mh.py \
    --checkpoint ./outputs/mh_training/best_model.pt \
    --style rock_punk \
    --key C_major \
    --section verse \
    --chord-progression C_maj F_maj G_maj C_maj \
    --length 256 \
    --num-variations 5
```

### Programmatic Usage

```python
from models.mh_transformer import MelodyHarmonyTransformer
from utils.constraints import ConstraintMaskGenerator

# Create model
model = MelodyHarmonyTransformer(
    vocab_size=2000,
    d_model=512,
    nhead=8,
    num_layers=6,
    style_vocab_size=3
)

# Generate with constraints
generated = model.generate(
    prompt_ids=prompt_tokens,
    style_ids=torch.tensor([0]),  # rock_punk
    key_ids=torch.tensor([0]),    # C major
    section_ids=torch.tensor([1]), # verse
    temperature=0.9,
    nucleus_p=0.9
)
```

## Configuration

### Model Configuration (`configs/mh_transformer.yaml`)
- Model architecture parameters
- Training hyperparameters
- Loss function weights
- Data paths and preprocessing

### Sampling Configuration (`configs/mh_sampling.yaml`)
- Generation parameters
- Style-specific settings
- Constraint configurations
- Output formatting

## Musical Constraints

### Scale Constraints
- Enforces key signature compliance
- Penalizes out-of-scale notes
- Supports major and minor modes

### Chord Constraints
- Aligns melody with chord progressions
- Supports extended harmonies
- Style-specific chord preferences

### Style Constraints
- **Rock/Punk**: Pentatonic preference, power chords
- **R&B/Ballad**: Extended chords, chromatic passing tones
- **Country/Pop**: Major scale preference, simple triads

### Repetition Control
- Token-level repetition tracking
- Phrase-level pattern detection
- Configurable penalty weights

## Data Format

### Input Tokens
Required token types for training:
- `STYLE_{rock_punk|rnb_ballad|country_pop}`
- `TEMPO`, `KEY`
- `SECTION_{INTRO|VERSE|CHORUS|BRIDGE|OUTRO}`
- `BAR`, `POS_{1/16 grid positions}`
- `INST_{KICK|SNARE|BASS_PICK|ACOUSTIC_STRUM|PIANO|LEAD}`
- `CHORD_{chord_name}`
- `NOTE_ON`, `NOTE_OFF`, `VEL_{bucketed}`, `DUR_{bucketed}`

### Training Data Structure
```
data/processed/
├── train/
│   ├── rock_punk/
│   │   └── song_001/
│   │       ├── melody_harmony.json
│   │       └── metadata.json
│   ├── rnb_ballad/
│   └── country_pop/
└── val/
    └── ... (similar structure)
```

## Testing

### Constraint Tests (`tests/test_constraints.py`)
- Musical constraint validation
- Mask generation verification
- Repetition control testing

### Model Tests (`tests/test_mh_transformer.py`)
- Architecture validation
- Forward pass testing
- Generation functionality

### Running Tests
```bash
# Run all tests
bash run_mh_tests.sh

# Individual test files
python tests/test_constraints.py
python tests/test_mh_transformer.py
```

## Implementation Details

### Style Conditioning
The model uses learnable embeddings for:
- Style ID (3 classes: rock_punk, rnb_ballad, country_pop)
- Key signature (24 classes: 12 keys × 2 modes)
- Section type (5 classes: intro, verse, chorus, bridge, outro)
- Optional groove features (32-dimensional vector)

### Constraint Application
Constraints are applied during generation as additive log-probability penalties:
```python
# Scale constraint
if note not in scale:
    logits[note_token] += scale_penalty

# Chord constraint  
if note not in current_chord:
    logits[note_token] += chord_penalty

# Repetition constraint
if token_count > max_repetitions:
    logits[token] += repetition_penalty
```

### Sampling Strategy
- **Temperature**: Section-specific values (verse: 0.9, chorus: 0.7)
- **Nucleus**: Top-p sampling with p=0.9
- **Constraints**: Real-time mask application
- **Early Stopping**: EOS token detection

## Performance Considerations

### Training Efficiency
- Mixed precision training support
- Gradient accumulation for large batches
- Model compilation for faster inference
- Checkpointing with automatic resumption

### Generation Speed
- Constraint mask pre-computation
- Efficient nucleus sampling
- Batch generation support
- GPU acceleration

### Memory Usage
- Configurable sequence lengths
- Attention mask optimization
- Gradient checkpointing option
- Dynamic vocabulary sizing

## Future Enhancements

### Model Architecture
- [ ] Multi-head attention visualization
- [ ] Hierarchical section modeling
- [ ] Cross-attention to arrangement
- [ ] Adaptive computation time

### Musical Features
- [ ] Micro-timing variations
- [ ] Expressive dynamics
- [ ] Articulation modeling
- [ ] Multi-instrument interaction

### Training Improvements
- [ ] Curriculum learning
- [ ] Self-supervised pre-training
- [ ] Style transfer capabilities
- [ ] Few-shot adaptation

## Dependencies

### Core Requirements
- PyTorch >= 1.12.0
- torch-audio >= 0.12.0
- transformers >= 4.20.0
- numpy >= 1.21.0

### Training & Evaluation
- pytorch-lightning >= 1.6.0
- wandb >= 0.12.0
- tensorboard >= 2.8.0
- pytest >= 7.0.0

### Data Processing
- pretty_midi >= 0.2.9
- librosa >= 0.9.0
- scipy >= 1.8.0
- pandas >= 1.4.0

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{melody_harmony_transformer,
  title={Style-Conditioned Melody and Harmony Generation with Musical Constraints},
  author={AI Music Composer Team},
  year={2024},
  url={https://github.com/your-repo/melody-harmony-transformer}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.