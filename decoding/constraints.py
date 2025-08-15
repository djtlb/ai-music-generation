"""
Constraint utilities for decoding and generation

Provides section masking, key masking, groove masking, and repetition penalties
for constrained music generation.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import json


def section_mask(tokens: torch.Tensor, bar_idx: int, plan: Dict[str, Any]) -> torch.Tensor:
    """
    Create a mask that enforces section-appropriate tokens based on the current bar
    and arrangement plan.
    
    Args:
        tokens: Vocabulary token indices [vocab_size]
        bar_idx: Current bar index in the song
        plan: Arrangement plan containing section timings and allowed tokens
        
    Returns:
        Boolean mask [vocab_size] where True means allowed
    """
    vocab_size = len(tokens) if hasattr(tokens, '__len__') else tokens.shape[0]
    mask = torch.ones(vocab_size, dtype=torch.bool)
    
    # Get current section from plan
    current_section = None
    cumulative_bars = 0
    
    for section in plan.get('sections', []):
        section_bars = section.get('bars', 8)
        if cumulative_bars <= bar_idx < cumulative_bars + section_bars:
            current_section = section.get('type', 'VERSE')
            break
        cumulative_bars += section_bars
    
    if current_section is None:
        return mask  # Allow all if section not found
    
    # Define section-specific allowed tokens
    section_constraints = {
        'INTRO': {
            'forbidden_patterns': ['LEAD', 'VOCAL'],  # Sparse instrumentation
            'encouraged': ['KICK', 'BASS_PICK']
        },
        'VERSE': {
            'forbidden_patterns': [],
            'encouraged': ['BASS_PICK', 'ACOUSTIC_STRUM', 'SNARE']
        },
        'CHORUS': {
            'forbidden_patterns': [],
            'encouraged': ['LEAD', 'KICK', 'SNARE', 'PIANO']
        },
        'BRIDGE': {
            'forbidden_patterns': ['LEAD'],  # Different texture
            'encouraged': ['PIANO', 'ACOUSTIC_STRUM']
        },
        'OUTRO': {
            'forbidden_patterns': ['LEAD'],
            'encouraged': ['KICK', 'BASS_PICK']
        }
    }
    
    constraints = section_constraints.get(current_section, {'forbidden_patterns': [], 'encouraged': []})
    
    # Apply constraints if we have a token vocabulary mapping
    if hasattr(tokens, 'vocab') or 'vocab' in plan:
        vocab = getattr(tokens, 'vocab', plan.get('vocab', {}))
        reverse_vocab = {v: k for k, v in vocab.items()}
        
        # Mask forbidden patterns
        for token_idx in range(vocab_size):
            token_name = reverse_vocab.get(token_idx, '')
            for forbidden in constraints['forbidden_patterns']:
                if forbidden in token_name:
                    mask[token_idx] = False
    
    return mask


def key_mask(tokens: torch.Tensor, key: str, tolerance: int = 2) -> torch.Tensor:
    """
    Create a mask that encourages notes within the specified key signature.
    
    Args:
        tokens: Vocabulary token indices [vocab_size]
        key: Key signature (e.g., 'C', 'F#', 'Bb')
        tolerance: Number of accidentals allowed outside the key
        
    Returns:
        Float mask [vocab_size] with penalties for out-of-key notes
    """
    vocab_size = len(tokens) if hasattr(tokens, '__len__') else tokens.shape[0]
    mask = torch.ones(vocab_size, dtype=torch.float)
    
    # Parse key signature
    key_map = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }
    
    root = key_map.get(key, 0)
    
    # Major scale intervals
    major_scale = [0, 2, 4, 5, 7, 9, 11]
    key_notes = {(root + interval) % 12 for interval in major_scale}
    
    # Apply penalties to out-of-key notes
    if hasattr(tokens, 'vocab'):
        vocab = tokens.vocab
        reverse_vocab = {v: k for k, v in vocab.items()}
        
        for token_idx in range(vocab_size):
            token_name = reverse_vocab.get(token_idx, '')
            if token_name.startswith('NOTE_ON_'):
                try:
                    midi_note = int(token_name.split('_')[-1])
                    note_class = midi_note % 12
                    
                    if note_class not in key_notes:
                        # Apply penalty based on distance from key
                        min_distance = min(
                            abs(note_class - kn) % 12 
                            for kn in key_notes
                        )
                        if min_distance > tolerance:
                            mask[token_idx] = 0.1  # Strong penalty
                        else:
                            mask[token_idx] = 0.7  # Mild penalty
                except (ValueError, IndexError):
                    pass
    
    return mask


def groove_mask(tokens: torch.Tensor, template: Dict[str, Any]) -> torch.Tensor:
    """
    Create a mask that enforces a specific groove template or drum pattern.
    
    Args:
        tokens: Vocabulary token indices [vocab_size]
        template: Groove template with timing and emphasis patterns
        
    Returns:
        Float mask [vocab_size] with groove-based weighting
    """
    vocab_size = len(tokens) if hasattr(tokens, '__len__') else tokens.shape[0]
    mask = torch.ones(vocab_size, dtype=torch.float)
    
    # Extract groove information
    drum_pattern = template.get('drum_pattern', {})
    time_feel = template.get('time_feel', 'straight')
    emphasis = template.get('emphasis', [])
    
    if hasattr(tokens, 'vocab'):
        vocab = tokens.vocab
        reverse_vocab = {v: k for k, v in vocab.items()}
        
        # Get current position in bar (simplified)
        current_pos = template.get('current_pos', 0) % 16  # 16th note grid
        
        for token_idx in range(vocab_size):
            token_name = reverse_vocab.get(token_idx, '')
            
            # Apply groove-specific rules
            if 'KICK' in token_name:
                # Kick typically on beats 1 and 3 in 4/4
                if current_pos in [0, 8]:  # Strong beats
                    mask[token_idx] = 1.5
                elif current_pos in [4, 12]:  # Weak beats
                    mask[token_idx] = 0.8
                    
            elif 'SNARE' in token_name:
                # Snare typically on beats 2 and 4
                if current_pos in [4, 12]:  # Backbeats
                    mask[token_idx] = 1.5
                elif current_pos in [0, 8]:
                    mask[token_idx] = 0.3
                    
            elif 'HIHAT' in token_name or 'HH' in token_name:
                # Hi-hat can be more frequent
                if time_feel == 'swing':
                    # Emphasize swing positions
                    if current_pos % 3 == 0:
                        mask[token_idx] = 1.2
                elif time_feel == 'straight':
                    # Even emphasis
                    if current_pos % 2 == 0:
                        mask[token_idx] = 1.1
    
    return mask


def repetition_penalty(logits: torch.Tensor, history: List[int], gamma: float = 1.2) -> torch.Tensor:
    """
    Apply repetition penalty to logits based on generation history.
    
    Args:
        logits: Raw model logits [vocab_size]
        history: List of previously generated token indices
        gamma: Repetition penalty strength (>1 penalizes repetition)
        
    Returns:
        Modified logits with repetition penalty applied
    """
    if len(history) == 0:
        return logits
    
    # Count occurrences of each token
    vocab_size = logits.shape[0]
    counts = torch.zeros(vocab_size, device=logits.device)
    
    for token in history:
        if 0 <= token < vocab_size:
            counts[token] += 1
    
    # Apply penalty: divide by gamma^count for repeated tokens
    penalty = torch.ones_like(logits)
    repeated_mask = counts > 0
    penalty[repeated_mask] = gamma ** counts[repeated_mask]
    
    # Apply penalty by dividing logits
    penalized_logits = logits / penalty
    
    return penalized_logits


def apply_all(logits: torch.Tensor, state: Dict[str, Any], plan: Dict[str, Any]) -> torch.Tensor:
    """
    Apply all constraint masks and penalties to logits.
    
    Args:
        logits: Raw model logits [vocab_size]
        state: Current generation state containing history, position, etc.
        plan: Complete generation plan with sections, key, groove, etc.
        
    Returns:
        Constrained logits ready for sampling
    """
    constrained_logits = logits.clone()
    
    # Extract state information
    bar_idx = state.get('bar_idx', 0)
    history = state.get('history', [])
    current_pos = state.get('current_pos', 0)
    
    # Extract plan information
    key = plan.get('key', 'C')
    groove_template = plan.get('groove_template', {})
    vocab = plan.get('vocab', {})
    
    # Create token tensor with vocab attached for mask functions
    class TokensWithVocab:
        def __init__(self, logits, vocab):
            self.vocab = vocab
            self.shape = logits.shape
        
        def __len__(self):
            return self.shape[0]
    
    tokens = TokensWithVocab(logits, vocab)
    
    # Apply section mask
    try:
        sec_mask = section_mask(tokens, bar_idx, plan)
        # Convert boolean mask to additive mask
        sec_penalty = torch.where(sec_mask, 0.0, -float('inf'))
        constrained_logits += sec_penalty
    except Exception as e:
        print(f"Section mask error: {e}")
    
    # Apply key mask
    try:
        key_weights = key_mask(tokens, key, tolerance=2)
        constrained_logits = constrained_logits * key_weights
    except Exception as e:
        print(f"Key mask error: {e}")
    
    # Apply groove mask
    try:
        groove_template['current_pos'] = current_pos
        groove_weights = groove_mask(tokens, groove_template)
        constrained_logits = constrained_logits * groove_weights
    except Exception as e:
        print(f"Groove mask error: {e}")
    
    # Apply repetition penalty
    try:
        repetition_gamma = plan.get('repetition_penalty', 1.2)
        constrained_logits = repetition_penalty(constrained_logits, history, repetition_gamma)
    except Exception as e:
        print(f"Repetition penalty error: {e}")
    
    return constrained_logits


def _test_constraints():
    """Test constraint functions with toy vocabulary"""
    # Create toy vocabulary
    vocab = {
        'PAD': 0, 'EOS': 1, 'BAR': 2, 'POS_1': 3, 'POS_2': 4,
        'NOTE_ON_60': 5, 'NOTE_ON_62': 6, 'NOTE_ON_64': 7, 'NOTE_ON_67': 8,  # C, D, E, G
        'KICK': 9, 'SNARE': 10, 'HIHAT': 11, 'BASS_PICK': 12,
        'CHORD_C': 13, 'CHORD_F': 14, 'CHORD_G': 15,
        'SECTION_VERSE': 16, 'SECTION_CHORUS': 17, 'LEAD': 18
    }
    
    vocab_size = len(vocab)
    logits = torch.randn(vocab_size)
    
    # Create test tokens with vocab
    class TestTokens:
        def __init__(self, vocab):
            self.vocab = vocab
            self.shape = (len(vocab),)
        
        def __len__(self):
            return len(self.vocab)
    
    tokens = TestTokens(vocab)
    
    # Test section mask
    plan = {
        'sections': [
            {'type': 'INTRO', 'bars': 4},
            {'type': 'VERSE', 'bars': 8},
            {'type': 'CHORUS', 'bars': 8}
        ],
        'vocab': vocab
    }
    
    sec_mask = section_mask(tokens, bar_idx=2, plan=plan)  # Should be in INTRO
    assert sec_mask[vocab['LEAD']] == False, "LEAD should be forbidden in INTRO"
    print("✓ Section mask test passed")
    
    # Test key mask
    key_weights = key_mask(tokens, key='C', tolerance=1)
    assert key_weights[vocab['NOTE_ON_60']] > 0.9, "C note should be encouraged in C major"
    print("✓ Key mask test passed")
    
    # Test groove mask
    groove_template = {
        'drum_pattern': {'kick': [0, 8], 'snare': [4, 12]},
        'time_feel': 'straight',
        'current_pos': 0  # On beat 1
    }
    groove_weights = groove_mask(tokens, groove_template)
    assert groove_weights[vocab['KICK']] > 1.0, "KICK should be emphasized on beat 1"
    print("✓ Groove mask test passed")
    
    # Test repetition penalty
    history = [vocab['NOTE_ON_60'], vocab['NOTE_ON_60'], vocab['NOTE_ON_62']]
    penalized_logits = repetition_penalty(logits, history, gamma=1.5)
    assert penalized_logits[vocab['NOTE_ON_60']] < logits[vocab['NOTE_ON_60']], "Repeated note should be penalized"
    print("✓ Repetition penalty test passed")
    
    # Test apply_all
    state = {
        'bar_idx': 5,  # Should be in VERSE
        'history': [vocab['NOTE_ON_60'], vocab['KICK']],
        'current_pos': 4  # Beat 2
    }
    
    final_logits = apply_all(logits, state, plan)
    assert final_logits.shape == logits.shape, "Output shape should match input"
    print("✓ Apply all test passed")
    
    print("All constraint tests passed successfully!")


if __name__ == "__main__":
    _test_constraints()