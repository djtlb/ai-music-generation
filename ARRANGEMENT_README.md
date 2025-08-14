# Arrangement Transformer

A PyTorch Lightning-based Transformer decoder for generating song arrangement sequences. Takes style, BPM, and target duration as inputs and outputs sequences of section tokens with bar counts.

## Features

- **Style-Conditioned Generation**: Supports rock_punk, rnb_ballad, and country_pop styles
- **Teacher Forcing**: Configurable teacher forcing with decay schedule
- **Coverage Penalty**: Prevents repetitive sequences and loops
- **Flexible Sampling**: Supports temperature, top-k, and top-p sampling
- **Data Augmentation**: Tempo and duration augmentation for training robustness

## Architecture

```
Input: [STYLE, TARGET_BPM, TARGET_DURATION] -> Transformer Decoder -> [SECTION_TOKENS]
```

The model uses:
- Style embeddings for conditioning
- Positional encoding for sequence modeling
- Causal attention masks for autoregressive generation
- Coverage mechanism to prevent loops
- Configurable section and bar constraints

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-arrangement.txt
```

### 2. Prepare Data

Place arrangement JSON files in `/data/processed/**/arrangement.json`:

```json
[
  {
    "style": "rock_punk",
    "tempo": 140,
    "duration_bars": 64,
    "sections": [
      {"type": "INTRO", "start_bar": 0, "length_bars": 4},
      {"type": "VERSE", "start_bar": 4, "length_bars": 16},
      {"type": "CHORUS", "start_bar": 20, "length_bars": 16},
      {"type": "OUTRO", "start_bar": 36, "length_bars": 4}
    ]
  }
]
```

### 3. Train Model

```bash
python scripts/train_arrangement.py --config configs/arrangement/default.yaml
```

For fast development:
```bash
python scripts/train_arrangement.py --config configs/arrangement/fast.yaml --debug
```

### 4. Generate Arrangements

```bash
python scripts/sample_arrangement.py \
  --checkpoint checkpoints/best_model.ckpt \
  --style rock_punk \
  --tempo 140 \
  --duration 64 \
  --num_samples 3
```

## Configuration

### Model Parameters

- `d_model`: Hidden dimension (default: 512)
- `n_heads`: Number of attention heads (default: 8)  
- `n_layers`: Number of transformer layers (default: 6)
- `coverage_penalty`: Strength of anti-repetition penalty (default: 0.3)
- `max_repeat_length`: Maximum allowed consecutive repetitions (default: 4)

### Training Parameters

- `teacher_forcing_ratio`: Initial teacher forcing probability (default: 0.8)
- `teacher_forcing_decay`: Decay rate per epoch (default: 0.995)
- `learning_rate`: Learning rate (default: 0.0001)
- `max_epochs`: Maximum training epochs (default: 100)

### Generation Parameters

- `temperature`: Sampling temperature (default: 0.9)
- `top_k`: Top-k sampling threshold (default: 50)
- `top_p`: Nucleus sampling threshold (default: 0.9)

## Data Format

### Input Requirements

- **STYLE**: One of `rock_punk`, `rnb_ballad`, `country_pop`
- **TARGET_BPM**: Integer BPM value (60-200)
- **TARGET_DURATION**: Target length in bars (16-128)

### Output Format

Generated arrangements are lists of section dictionaries:

```python
[
  {
    "type": "INTRO",
    "start_bar": 0,
    "length_bars": 4
  },
  {
    "type": "VERSE", 
    "start_bar": 4,
    "length_bars": 16
  }
  # ... more sections
]
```

### Supported Section Types

- `INTRO`: Introduction sections (2-8 bars)
- `VERSE`: Verse sections (8-32 bars)
- `CHORUS`: Chorus sections (8-32 bars)
- `BRIDGE`: Bridge sections (4-16 bars)
- `OUTRO`: Outro sections (2-8 bars)

## Testing

Run the test suite:

```bash
python run_tests.py
```

Or use pytest directly:

```bash
python -m pytest tests/test_arrangement.py -v
```

### Test Coverage

- ✅ Tokenizer encode/decode roundtrip
- ✅ Dataset loading and validation
- ✅ Model forward pass and shape consistency
- ✅ Training step execution
- ✅ Generation functionality
- ✅ DataModule integration

## Model Architecture Details

### Coverage Mechanism

The coverage mechanism tracks recently generated tokens and applies penalties to prevent repetitive patterns:

```python
def compute_coverage_penalty(self, logits, generated_tokens, penalty_weight=0.3):
    # Apply penalty proportional to recent token frequency
    # Prevents loops like: VERSE -> CHORUS -> VERSE -> CHORUS -> ...
```

### Style Conditioning

Styles are embedded and concatenated with tempo/duration features:

```python
# Style embedding (learnable)
style_emb = self.style_embedding(style_ids)

# Tempo/duration projection (normalized inputs)
tempo_emb = self.tempo_projection(tempo_norm)  
duration_emb = self.duration_projection(duration_norm)

# Combined conditioning
condition_encoding = self.condition_projection([style_emb, tempo_emb, duration_emb])
```

### Teacher Forcing Schedule

Teacher forcing ratio decays during training to improve generation quality:

```python
# Each epoch:
self.teacher_forcing_ratio = max(
    self.min_teacher_forcing,
    self.teacher_forcing_ratio * self.teacher_forcing_decay
)
```

## Configuration Examples

### Production Training

```yaml
# configs/arrangement/default.yaml
model:
  d_model: 512
  n_heads: 8
  n_layers: 6
  coverage_penalty: 0.3

training:
  max_epochs: 100
  learning_rate: 0.0001
  teacher_forcing_ratio: 0.8
```

### Fast Development

```yaml
# configs/arrangement/fast.yaml  
model:
  d_model: 256
  n_heads: 4
  n_layers: 3

training:
  max_epochs: 50
  learning_rate: 0.001
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size` or `d_model`
2. **Repetitive Generation**: Increase `coverage_penalty` or `temperature`
3. **Poor Convergence**: Check `learning_rate` and `teacher_forcing_ratio`
4. **Data Loading Errors**: Verify JSON format and file paths

### Performance Tips

- Use mixed precision training (`precision: "16-mixed"`)
- Increase `num_workers` for faster data loading
- Use GPU acceleration when available
- Monitor teacher forcing ratio decay

## Future Enhancements

- [ ] Beam search decoding
- [ ] Length penalty for generation
- [ ] Multi-scale attention for long sequences
- [ ] Reinforcement learning fine-tuning
- [ ] Style transfer between arrangements
- [ ] Conditional generation on specific patterns

## Citation

```bibtex
@software{arrangement_transformer,
  title={Arrangement Transformer for AI Music Generation},
  author={AI Music Composer Team},
  year={2024},
  url={https://github.com/ai-music-composer/arrangement-transformer}
}
```