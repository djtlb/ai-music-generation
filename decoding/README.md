# Decoding Constraints Module

This module provides constraint utilities for music generation during decoding, implementing section-aware masking, key constraints, groove patterns, and repetition control.

## Core Functions

### `section_mask(tokens, bar_idx, plan)`
Creates a boolean mask that enforces section-appropriate tokens based on the current bar position and arrangement plan.

**Parameters:**
- `tokens`: Token vocabulary with `.vocab` attribute
- `bar_idx`: Current bar index in the song
- `plan`: Arrangement plan with section timings and constraints

**Returns:** Boolean tensor `[vocab_size]` where `True` means allowed

**Example:**
```python
plan = {
    'sections': [
        {'type': 'INTRO', 'bars': 4},
        {'type': 'VERSE', 'bars': 8}
    ],
    'vocab': vocab
}
mask = section_mask(tokens, bar_idx=2, plan=plan)  # Bar 2 is in INTRO
# INTRO forbids LEAD and VOCAL tokens
```

### `key_mask(tokens, key, tolerance=2)`
Creates a mask that encourages notes within the specified key signature.

**Parameters:**
- `tokens`: Token vocabulary with `.vocab` attribute
- `key`: Key signature (e.g., 'C', 'F#', 'Bb')
- `tolerance`: Number of accidentals allowed outside the key

**Returns:** Float tensor `[vocab_size]` with penalties for out-of-key notes

**Example:**
```python
weights = key_mask(tokens, key='C', tolerance=1)
# C major notes get weight 1.0, others get penalties
```

### `groove_mask(tokens, template)`
Creates a mask that enforces a specific groove template or drum pattern.

**Parameters:**
- `tokens`: Token vocabulary with `.vocab` attribute  
- `template`: Groove template with timing and emphasis patterns

**Returns:** Float tensor `[vocab_size]` with groove-based weighting

**Example:**
```python
template = {
    'drum_pattern': {'kick': [0, 8], 'snare': [4, 12]},
    'time_feel': 'straight',
    'current_pos': 0  # Beat 1
}
weights = groove_mask(tokens, template)
# KICK emphasized on beats 1 and 3, SNARE on 2 and 4
```

### `repetition_penalty(logits, history, gamma=1.2)`
Applies repetition penalty to logits based on generation history.

**Parameters:**
- `logits`: Raw model logits `[vocab_size]`
- `history`: List of previously generated token indices
- `gamma`: Repetition penalty strength (>1 penalizes repetition)

**Returns:** Modified logits with repetition penalty applied

**Example:**
```python
history = [60, 60, 62]  # NOTE_ON_60 repeated twice
penalized_logits = repetition_penalty(logits, history, gamma=1.5)
# Repeated tokens get penalized
```

### `apply_all(logits, state, plan)`
Applies all constraint masks and penalties to logits in one call.

**Parameters:**
- `logits`: Raw model logits `[vocab_size]`
- `state`: Generation state with `bar_idx`, `history`, `current_pos`
- `plan`: Complete generation plan with sections, key, groove, etc.

**Returns:** Constrained logits ready for sampling

**Example:**
```python
state = {
    'bar_idx': 5,      # Current bar
    'history': [60, 61, 60],  # Previous tokens
    'current_pos': 4   # Current position in bar (16th note grid)
}

plan = {
    'sections': [...],
    'key': 'C',
    'groove_template': {...},
    'vocab': vocab,
    'repetition_penalty': 1.3
}

constrained_logits = apply_all(logits, state, plan)
```

## Constraint Types

### Section Constraints
- **INTRO**: Forbids `LEAD`, `VOCAL` (sparse instrumentation)  
- **VERSE**: No specific constraints (allows most tokens)
- **CHORUS**: Encourages full instrumentation
- **BRIDGE**: Forbids `LEAD` (different texture)
- **OUTRO**: Forbids `LEAD`, emphasizes rhythm section

### Key Constraints
- Encourages scale notes with full weight (1.0)
- Applies mild penalty (0.7) for near-key notes within tolerance
- Applies strong penalty (0.1) for distant chromatic notes
- Non-note tokens remain unaffected

### Groove Constraints  
- **KICK**: Emphasized on beats 1 and 3 (positions 0, 8)
- **SNARE**: Emphasized on beats 2 and 4 (positions 4, 12)  
- **HIHAT**: Timing varies by feel (straight vs swing)
- Weights applied multiplicatively to logits

### Repetition Constraints
- Penalty proportional to token frequency in recent history
- Formula: `logits / (gamma ^ count)` where count = occurrences
- Applied per-token based on generation history

## Testing

Run the validation script to test all functions:

```bash
python validate_constraints.py
```

Or run the comprehensive test suite:

```bash
python test_constraints.py
```

The tests cover:
- Section masking for different song sections
- Key signature constraints for various keys  
- Groove pattern enforcement
- Repetition penalty mechanics
- Integration scenarios with combined constraints

## Integration Example

```python
from decoding.constraints import apply_all
import torch

# Setup
vocab = {...}  # Your tokenizer vocabulary
logits = model(input_tokens)  # Raw model output

# Generation state
state = {
    'bar_idx': 12,           # Current bar in song
    'history': [60, 61, 60], # Recently generated tokens  
    'current_pos': 8         # Position in current bar
}

# Arrangement plan
plan = {
    'sections': [
        {'type': 'VERSE', 'bars': 8},
        {'type': 'CHORUS', 'bars': 8}
    ],
    'key': 'G',
    'groove_template': {
        'drum_pattern': {'kick': [0, 8], 'snare': [4, 12]},
        'time_feel': 'straight'  
    },
    'vocab': vocab,
    'repetition_penalty': 1.4
}

# Apply all constraints
constrained_logits = apply_all(logits, state, plan)

# Sample next token
probs = torch.softmax(constrained_logits, dim=-1)
next_token = torch.multinomial(probs, 1)
```

This creates musically coherent generation that respects song structure, key signatures, rhythmic patterns, and avoids excessive repetition.