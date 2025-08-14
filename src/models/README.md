# Critic Reward Model & DPO Finetuning

This directory contains the implementation of a critic reward model for music quality assessment and Direct Preference Optimization (DPO) finetuning for preference-aligned music generation.

## Architecture Overview

```
critic/
├── model.py          # Critic neural network architecture
├── dataset.py        # Data loading and preprocessing
├── train.py          # Training loop and utilities
└── evaluate.py       # Comprehensive evaluation suite

finetune/
└── dpo.py            # DPO training implementation

scripts/
├── train_critic.py   # CLI for critic training
├── finetune_dpo.py   # CLI for DPO finetuning
└── evaluate_model.py # CLI for evaluation
```

## Quality Dimensions

The critic model scores audio clips on:

- **hook_strength**: Memorability and catchiness of melodic content
- **harmonic_stability**: Quality and coherence of chord progressions  
- **arrangement_contrast**: Dynamic variation and structural interest
- **mix_quality**: Technical audio quality (LUFS, spectral balance)
- **style_match**: Consistency with target musical style

## Usage

### 1. Train Critic Model

```bash
# Create mock training data
python scripts/train_critic.py --mock_data --data_csv mock_preferences.csv

# Train on real data
python scripts/train_critic.py \
  --data_csv preference_data.csv \
  --audio_dir ./audio_clips \
  --epochs 100 \
  --batch_size 16 \
  --learning_rate 1e-3
```

### 2. DPO Finetuning

```bash
# Create mock DPO data
python scripts/finetune_dpo.py --mock_data --data_path ./mock_dpo_data/

# Run DPO finetuning
python scripts/finetune_dpo.py \
  --data_path dpo_pairs.json \
  --critic_checkpoint logs/critic_training/checkpoint_best.pth \
  --epochs 5 \
  --batch_size 8 \
  --beta 0.1
```

### 3. Evaluation

```bash
# Create validation playlist
python scripts/evaluate_model.py --create_playlist --test_data validation.csv

# Evaluate critic performance
python scripts/evaluate_model.py \
  --critic_checkpoint logs/critic_training/checkpoint_best.pth \
  --test_data validation.csv \
  --audio_dir ./validation_audio

# Compare before/after DPO
python scripts/evaluate_model.py \
  --critic_checkpoint logs/critic_training/checkpoint_best.pth \
  --test_data validation.csv \
  --audio_dir ./validation_audio \
  --before_dpo before_dpo.pth \
  --after_dpo after_dpo.pth
```

## Data Format

### Preference CSV Format

```csv
clip_id,audio_file,style,preference_rank,hook_strength,harmonic_stability,arrangement_contrast,mix_quality,style_match,overall_score
clip_0001,audio_0001.wav,rock_punk,1,0.85,0.78,0.92,0.76,0.88,0.84
clip_0002,audio_0002.wav,rnb_ballad,2,0.72,0.89,0.65,0.91,0.83,0.80
```

### DPO Pairs JSON Format

```json
[
  {
    "chosen_sequences": [1, 45, 123, ...],
    "rejected_sequences": [1, 67, 89, ...], 
    "style_ids": 0
  }
]
```

## Target Metrics

### Critic Performance
- **Overall Accuracy**: >85% within 0.1 threshold
- **Quality Correlation**: >0.8 for each dimension
- **Preference Alignment**: >80% pairwise accuracy

### DPO Results
- **Win Rate**: >80% preference alignment
- **Reward Improvement**: +15% average score increase
- **KL Divergence**: <0.1 (maintains model coherence)

## Integration with Pipeline

The critic model integrates with the broader music generation pipeline:

1. **Arrangement Generator** → sequences
2. **Melody/Harmony Generator** → MIDI tracks  
3. **Sound Design** → audio stems
4. **Mixing/Mastering** → final audio
5. **Critic Evaluation** → quality scores
6. **DPO Finetuning** → preference alignment

This creates a complete feedback loop for improving generation quality through human preference optimization.