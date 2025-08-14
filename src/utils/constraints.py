"""
Constraint utilities for melody and harmony generation

Provides constraint masks and repetition control for musical generation.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class MusicalConstraints:
    """Musical constraint definitions and utilities"""
    
    # Major scale intervals (semitones from root)
    MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
    
    # Minor scale intervals (natural minor)
    MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
    
    # Chord types and their intervals
    CHORD_INTERVALS = {
        'maj': [0, 4, 7],           # Major triad
        'min': [0, 3, 7],           # Minor triad
        'dim': [0, 3, 6],           # Diminished triad
        'aug': [0, 4, 8],           # Augmented triad
        'maj7': [0, 4, 7, 11],      # Major 7th
        'min7': [0, 3, 7, 10],      # Minor 7th
        'dom7': [0, 4, 7, 10],      # Dominant 7th
        'dim7': [0, 3, 6, 9],       # Diminished 7th
        'hdim7': [0, 3, 6, 10],     # Half-diminished 7th
        'maj9': [0, 4, 7, 11, 14],  # Major 9th
        'min9': [0, 3, 7, 10, 14],  # Minor 9th
        'sus2': [0, 2, 7],          # Suspended 2nd
        'sus4': [0, 5, 7],          # Suspended 4th
    }
    
    # Common chord progressions by style
    STYLE_PROGRESSIONS = {
        'rock_punk': {
            'verse': ['I', 'vi', 'IV', 'V'],
            'chorus': ['vi', 'IV', 'I', 'V'],
            'bridge': ['IV', 'V', 'vi', 'IV']
        },
        'rnb_ballad': {
            'verse': ['vi', 'IV', 'I', 'V'],
            'chorus': ['I', 'V', 'vi', 'IV'],
            'bridge': ['ii', 'V', 'I', 'vi']
        },
        'country_pop': {
            'verse': ['I', 'V', 'vi', 'IV'],
            'chorus': ['IV', 'I', 'V', 'vi'],
            'bridge': ['vi', 'IV', 'V', 'I']
        }
    }
    
    @staticmethod
    def get_scale_notes(key: int, is_major: bool = True) -> List[int]:
        """Get scale notes for a given key"""
        scale = MusicalConstraints.MAJOR_SCALE if is_major else MusicalConstraints.MINOR_SCALE
        return [(key + interval) % 12 for interval in scale]
    
    @staticmethod
    def get_chord_notes(root: int, chord_type: str) -> List[int]:
        """Get chord notes for a given root and chord type"""
        if chord_type not in MusicalConstraints.CHORD_INTERVALS:
            chord_type = 'maj'  # Default to major
        
        intervals = MusicalConstraints.CHORD_INTERVALS[chord_type]
        return [(root + interval) % 12 for interval in intervals]


class ConstraintMaskGenerator:
    """Generate constraint masks for musical generation"""
    
    def __init__(self, tokenizer_vocab: Dict[str, int]):
        """
        Initialize constraint mask generator
        
        Args:
            tokenizer_vocab: Vocabulary mapping from tokenizer
        """
        self.vocab = tokenizer_vocab
        self.reverse_vocab = {v: k for k, v in tokenizer_vocab.items()}
        
        # Parse vocabulary to identify different token types
        self._parse_vocab()
    
    def _parse_vocab(self):
        """Parse vocabulary to identify token types"""
        self.note_tokens = {}
        self.chord_tokens = {}
        self.control_tokens = {}
        
        for token, idx in self.vocab.items():
            if token.startswith('NOTE_ON_'):
                note = int(token.split('_')[-1])
                self.note_tokens[note] = idx
            elif token.startswith('CHORD_'):
                chord_name = token.replace('CHORD_', '')
                self.chord_tokens[chord_name] = idx
            elif token in ['STYLE_rock_punk', 'STYLE_rnb_ballad', 'STYLE_country_pop',
                          'SECTION_INTRO', 'SECTION_VERSE', 'SECTION_CHORUS', 
                          'SECTION_BRIDGE', 'SECTION_OUTRO', 'BAR', 'POS_1', 'POS_2']:
                self.control_tokens[token] = idx
    
    def create_scale_constraint_mask(
        self, 
        key: int, 
        is_major: bool = True,
        seq_len: int = 512,
        penalty_value: float = -10.0
    ) -> torch.Tensor:
        """
        Create constraint mask to enforce scale compatibility
        
        Args:
            key: Root key (0-11, C=0)
            is_major: Whether to use major or minor scale
            seq_len: Sequence length
            penalty_value: Penalty value for out-of-scale notes
            
        Returns:
            Constraint mask [seq_len, vocab_size]
        """
        vocab_size = len(self.vocab)
        mask = torch.zeros(seq_len, vocab_size)
        
        scale_notes = MusicalConstraints.get_scale_notes(key, is_major)
        
        # Apply penalty to out-of-scale notes
        for note_midi, token_idx in self.note_tokens.items():
            note_class = note_midi % 12
            if note_class not in scale_notes:
                mask[:, token_idx] = penalty_value
        
        return mask
    
    def create_chord_constraint_mask(
        self,
        chord_sequence: List[str],
        seq_len: int = 512,
        penalty_value: float = -5.0
    ) -> torch.Tensor:
        """
        Create constraint mask to enforce chord compatibility
        
        Args:
            chord_sequence: List of chord names per time step
            seq_len: Sequence length
            penalty_value: Penalty for non-chord tones
            
        Returns:
            Constraint mask [seq_len, vocab_size]
        """
        vocab_size = len(self.vocab)
        mask = torch.zeros(seq_len, vocab_size)
        
        bars_per_chord = seq_len // len(chord_sequence)
        
        for i, chord_name in enumerate(chord_sequence):
            start_pos = i * bars_per_chord
            end_pos = min((i + 1) * bars_per_chord, seq_len)
            
            # Parse chord name (simplified)
            if '_' in chord_name:
                root_str, chord_type = chord_name.split('_', 1)
            else:
                root_str, chord_type = chord_name, 'maj'
            
            # Convert root to MIDI number (simplified mapping)
            root_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
            root = root_map.get(root_str, 0)
            
            chord_notes = MusicalConstraints.get_chord_notes(root, chord_type)
            
            # Apply penalty to non-chord tones during this chord
            for note_midi, token_idx in self.note_tokens.items():
                note_class = note_midi % 12
                if note_class not in chord_notes:
                    mask[start_pos:end_pos, token_idx] = penalty_value
        
        return mask
    
    def create_repetition_constraint_mask(
        self,
        generated_sequence: torch.Tensor,
        current_position: int,
        window_size: int = 8,
        max_repetitions: int = 2,
        penalty_value: float = -3.0
    ) -> torch.Tensor:
        """
        Create constraint mask to control repetition
        
        Args:
            generated_sequence: Already generated tokens [seq_len]
            current_position: Current generation position
            window_size: Look-back window for repetition analysis
            max_repetitions: Maximum allowed repetitions
            penalty_value: Penalty for excessive repetition
            
        Returns:
            Constraint mask [vocab_size]
        """
        vocab_size = len(self.vocab)
        mask = torch.zeros(vocab_size)
        
        if current_position < window_size:
            return mask
        
        # Analyze recent tokens
        start_pos = max(0, current_position - window_size)
        recent_tokens = generated_sequence[start_pos:current_position]
        
        # Count token frequencies
        token_counts = torch.bincount(recent_tokens, minlength=vocab_size)
        
        # Apply penalty to tokens that appear too frequently
        over_limit = token_counts >= max_repetitions
        mask[over_limit] = penalty_value
        
        return mask
    
    def create_style_constraint_mask(
        self,
        style: str,
        section: str,
        key: int,
        seq_len: int = 512
    ) -> torch.Tensor:
        """
        Create style-specific constraint mask
        
        Args:
            style: Style name ('rock_punk', 'rnb_ballad', 'country_pop')
            section: Section name ('verse', 'chorus', 'bridge')
            key: Root key
            seq_len: Sequence length
            
        Returns:
            Combined constraint mask [seq_len, vocab_size]
        """
        vocab_size = len(self.vocab)
        mask = torch.zeros(seq_len, vocab_size)
        
        # Style-specific constraints
        if style == 'rock_punk':
            # Prefer power chords and pentatonic scales
            # Pentatonic: 1, 2, 3, 5, 6
            pentatonic_intervals = [0, 2, 4, 7, 9]
            scale_notes = [(key + interval) % 12 for interval in pentatonic_intervals]
            
            # Light penalty for non-pentatonic notes
            for note_midi, token_idx in self.note_tokens.items():
                note_class = note_midi % 12
                if note_class not in scale_notes:
                    mask[:, token_idx] = -2.0
                    
        elif style == 'rnb_ballad':
            # Prefer extended chords and chromatic passing tones
            # Allow all chromatic notes but favor chord tones
            pass  # More permissive
            
        elif style == 'country_pop':
            # Prefer major scales and simple triads
            major_scale_notes = MusicalConstraints.get_scale_notes(key, is_major=True)
            
            # Moderate penalty for out-of-scale notes
            for note_midi, token_idx in self.note_tokens.items():
                note_class = note_midi % 12
                if note_class not in major_scale_notes:
                    mask[:, token_idx] = -3.0
        
        return mask


class RepetitionController:
    """Control repetition patterns in generated sequences"""
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.reset()
    
    def reset(self):
        """Reset repetition tracking"""
        self.token_history = []
        self.phrase_history = []
        self.current_phrase = []
    
    def update(self, token: int, is_phrase_boundary: bool = False):
        """Update repetition tracking with new token"""
        self.token_history.append(token)
        self.current_phrase.append(token)
        
        if is_phrase_boundary:
            self.phrase_history.append(self.current_phrase.copy())
            self.current_phrase = []
    
    def get_repetition_penalty(
        self, 
        window_size: int = 16,
        phrase_window: int = 4,
        token_penalty: float = -2.0,
        phrase_penalty: float = -5.0
    ) -> torch.Tensor:
        """
        Get repetition penalty for next token prediction
        
        Args:
            window_size: Token-level repetition window
            phrase_window: Phrase-level repetition window
            token_penalty: Penalty for token repetition
            phrase_penalty: Penalty for phrase repetition
            
        Returns:
            Penalty vector [vocab_size]
        """
        penalty = torch.zeros(self.vocab_size)
        
        # Token-level repetition penalty
        if len(self.token_history) >= window_size:
            recent_tokens = self.token_history[-window_size:]
            token_counts = torch.bincount(torch.tensor(recent_tokens), minlength=self.vocab_size)
            
            # Apply penalty proportional to frequency
            penalty = penalty + token_counts.float() * token_penalty
        
        # Phrase-level repetition penalty
        if len(self.phrase_history) >= phrase_window and len(self.current_phrase) > 0:
            recent_phrases = self.phrase_history[-phrase_window:]
            current_start = self.current_phrase
            
            for phrase in recent_phrases:
                # Check if current phrase start matches previous phrase
                if len(phrase) >= len(current_start):
                    if phrase[:len(current_start)] == current_start:
                        # Penalize continuing this phrase pattern
                        if len(current_start) < len(phrase):
                            next_token = phrase[len(current_start)]
                            penalty[next_token] += phrase_penalty
        
        return penalty
    
    def detect_phrase_boundary(self, token: int, vocab: Dict[str, int]) -> bool:
        """Detect if token represents a phrase boundary"""
        reverse_vocab = {v: k for k, v in vocab.items()}
        token_name = reverse_vocab.get(token, '')
        
        # Simple heuristic: phrase boundaries at bar starts or specific tokens
        return token_name in ['BAR', 'SECTION_VERSE', 'SECTION_CHORUS', 'SECTION_BRIDGE']


def create_combined_constraint_mask(
    mask_generator: ConstraintMaskGenerator,
    repetition_controller: RepetitionController,
    key: int,
    chord_sequence: List[str],
    style: str,
    section: str,
    generated_sequence: torch.Tensor,
    current_position: int,
    seq_len: int = 512
) -> torch.Tensor:
    """
    Create combined constraint mask from all sources
    
    Args:
        mask_generator: Constraint mask generator
        repetition_controller: Repetition controller
        key: Musical key
        chord_sequence: Chord progression
        style: Musical style
        section: Song section
        generated_sequence: Generated tokens so far
        current_position: Current generation position
        seq_len: Total sequence length
        
    Returns:
        Combined constraint mask [vocab_size]
    """
    vocab_size = len(mask_generator.vocab)
    
    # Scale constraint
    scale_mask = mask_generator.create_scale_constraint_mask(key, seq_len=seq_len)
    current_scale_mask = scale_mask[min(current_position, seq_len-1), :]
    
    # Chord constraint
    chord_mask = mask_generator.create_chord_constraint_mask(chord_sequence, seq_len=seq_len)
    current_chord_mask = chord_mask[min(current_position, seq_len-1), :]
    
    # Style constraint
    style_mask = mask_generator.create_style_constraint_mask(style, section, key, seq_len=seq_len)
    current_style_mask = style_mask[min(current_position, seq_len-1), :]
    
    # Repetition constraint
    repetition_mask = mask_generator.create_repetition_constraint_mask(
        generated_sequence, current_position
    )
    
    # Repetition controller penalty
    repetition_penalty = repetition_controller.get_repetition_penalty()
    
    # Combine all constraints
    combined_mask = (current_scale_mask + current_chord_mask + 
                    current_style_mask + repetition_mask + repetition_penalty)
    
    return combined_mask


if __name__ == "__main__":
    # Test constraint utilities
    vocab = {
        'PAD': 0, 'EOS': 1, 'BAR': 2, 'POS_1': 3, 'POS_2': 4,
        'NOTE_ON_60': 5, 'NOTE_ON_62': 6, 'NOTE_ON_64': 7,  # C, D, E
        'CHORD_C_maj': 8, 'CHORD_F_maj': 9, 'CHORD_G_maj': 10,
        'STYLE_rock_punk': 11, 'SECTION_VERSE': 12
    }
    
    # Test mask generator
    mask_gen = ConstraintMaskGenerator(vocab)
    
    # Test scale constraint
    scale_mask = mask_gen.create_scale_constraint_mask(key=0, seq_len=64)
    print(f"Scale mask shape: {scale_mask.shape}")
    
    # Test chord constraint
    chord_sequence = ['C_maj', 'F_maj', 'G_maj', 'C_maj']
    chord_mask = mask_gen.create_chord_constraint_mask(chord_sequence, seq_len=64)
    print(f"Chord mask shape: {chord_mask.shape}")
    
    # Test repetition controller
    rep_controller = RepetitionController(len(vocab))
    rep_controller.update(5)  # NOTE_ON_60
    rep_controller.update(6)  # NOTE_ON_62
    rep_controller.update(5)  # NOTE_ON_60 (repetition)
    
    penalty = rep_controller.get_repetition_penalty()
    print(f"Repetition penalty shape: {penalty.shape}")
    print(f"Penalty for token 5: {penalty[5].item()}")
    
    print("Constraint utilities test completed successfully!")