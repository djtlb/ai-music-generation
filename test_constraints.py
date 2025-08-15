"""
Unit tests for decoding constraints module

Tests all constraint functions with toy vocabularies and scenarios.
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add the parent directory to path to import constraints
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decoding.constraints import (
    section_mask, key_mask, groove_mask, repetition_penalty, apply_all
)


class TestTokens:
    """Mock tokens class with vocabulary for testing"""
    def __init__(self, vocab):
        self.vocab = vocab
        self.shape = (len(vocab),)
    
    def __len__(self):
        return len(self.vocab)


@pytest.fixture
def toy_vocab():
    """Fixture providing a toy vocabulary for testing"""
    return {
        'PAD': 0, 'EOS': 1, 'BAR': 2, 'POS_1': 3, 'POS_2': 4,
        'NOTE_ON_60': 5, 'NOTE_ON_62': 6, 'NOTE_ON_64': 7, 'NOTE_ON_67': 8,  # C, D, E, G
        'NOTE_ON_61': 9, 'NOTE_ON_63': 10, 'NOTE_ON_66': 11,  # C#, D#, F#
        'KICK': 12, 'SNARE': 13, 'HIHAT': 14, 'BASS_PICK': 15,
        'CHORD_C': 16, 'CHORD_F': 17, 'CHORD_G': 18,
        'SECTION_VERSE': 19, 'SECTION_CHORUS': 20, 'LEAD': 21,
        'ACOUSTIC_STRUM': 22, 'PIANO': 23, 'VOCAL': 24
    }


@pytest.fixture
def test_plan(toy_vocab):
    """Fixture providing a test arrangement plan"""
    return {
        'sections': [
            {'type': 'INTRO', 'bars': 4},
            {'type': 'VERSE', 'bars': 8},
            {'type': 'CHORUS', 'bars': 8},
            {'type': 'BRIDGE', 'bars': 4},
            {'type': 'OUTRO', 'bars': 4}
        ],
        'key': 'C',
        'vocab': toy_vocab,
        'groove_template': {
            'drum_pattern': {'kick': [0, 8], 'snare': [4, 12]},
            'time_feel': 'straight',
            'emphasis': [0, 4, 8, 12]
        },
        'repetition_penalty': 1.3
    }


class TestSectionMask:
    """Tests for section_mask function"""
    
    def test_intro_section_constraints(self, toy_vocab, test_plan):
        """Test that INTRO section forbids LEAD and VOCAL"""
        tokens = TestTokens(toy_vocab)
        mask = section_mask(tokens, bar_idx=2, plan=test_plan)  # Bar 2 is in INTRO
        
        assert mask[toy_vocab['LEAD']] == False, "LEAD should be forbidden in INTRO"
        assert mask[toy_vocab['VOCAL']] == False, "VOCAL should be forbidden in INTRO"
        assert mask[toy_vocab['KICK']] == True, "KICK should be allowed in INTRO"
        assert mask[toy_vocab['BASS_PICK']] == True, "BASS_PICK should be allowed in INTRO"
    
    def test_bridge_section_constraints(self, toy_vocab, test_plan):
        """Test that BRIDGE section forbids LEAD"""
        tokens = TestTokens(toy_vocab)
        mask = section_mask(tokens, bar_idx=22, plan=test_plan)  # Bar 22 is in BRIDGE
        
        assert mask[toy_vocab['LEAD']] == False, "LEAD should be forbidden in BRIDGE"
        assert mask[toy_vocab['PIANO']] == True, "PIANO should be allowed in BRIDGE"
        assert mask[toy_vocab['ACOUSTIC_STRUM']] == True, "ACOUSTIC_STRUM should be allowed in BRIDGE"
    
    def test_verse_section_no_constraints(self, toy_vocab, test_plan):
        """Test that VERSE section allows most tokens"""
        tokens = TestTokens(toy_vocab)
        mask = section_mask(tokens, bar_idx=6, plan=test_plan)  # Bar 6 is in VERSE
        
        # VERSE should have no forbidden patterns
        assert mask[toy_vocab['LEAD']] == True, "LEAD should be allowed in VERSE"
        assert mask[toy_vocab['VOCAL']] == True, "VOCAL should be allowed in VERSE"
        assert mask[toy_vocab['BASS_PICK']] == True, "BASS_PICK should be allowed in VERSE"
    
    def test_out_of_range_bar(self, toy_vocab, test_plan):
        """Test behavior with bar index beyond plan"""
        tokens = TestTokens(toy_vocab)
        mask = section_mask(tokens, bar_idx=100, plan=test_plan)  # Beyond plan range
        
        # Should allow all tokens if section not found
        assert all(mask), "All tokens should be allowed for out-of-range bar"


class TestKeyMask:
    """Tests for key_mask function"""
    
    def test_c_major_key_mask(self, toy_vocab):
        """Test C major key mask favors scale notes"""
        tokens = TestTokens(toy_vocab)
        weights = key_mask(tokens, key='C', tolerance=1)
        
        # C major notes (C, D, E, G) should have weight 1.0
        assert weights[toy_vocab['NOTE_ON_60']] == 1.0, "C should have full weight"
        assert weights[toy_vocab['NOTE_ON_62']] == 1.0, "D should have full weight"
        assert weights[toy_vocab['NOTE_ON_64']] == 1.0, "E should have full weight"
        assert weights[toy_vocab['NOTE_ON_67']] == 1.0, "G should have full weight"
        
        # Non-scale notes should be penalized
        assert weights[toy_vocab['NOTE_ON_61']] < 1.0, "C# should be penalized"
        assert weights[toy_vocab['NOTE_ON_63']] < 1.0, "D# should be penalized"
    
    def test_key_tolerance(self, toy_vocab):
        """Test that tolerance parameter affects penalty severity"""
        tokens = TestTokens(toy_vocab)
        strict_weights = key_mask(tokens, key='C', tolerance=0)
        loose_weights = key_mask(tokens, key='C', tolerance=3)
        
        # Out-of-key notes should be more penalized with strict tolerance
        assert strict_weights[toy_vocab['NOTE_ON_61']] <= loose_weights[toy_vocab['NOTE_ON_61']]
    
    def test_different_key_signatures(self, toy_vocab):
        """Test different key signatures produce different masks"""
        tokens = TestTokens(toy_vocab)
        c_weights = key_mask(tokens, key='C', tolerance=1)
        g_weights = key_mask(tokens, key='G', tolerance=1)
        
        # Weights should differ between keys
        assert not torch.equal(c_weights, g_weights), "Different keys should produce different masks"
    
    def test_non_note_tokens_unaffected(self, toy_vocab):
        """Test that non-note tokens maintain weight 1.0"""
        tokens = TestTokens(toy_vocab)
        weights = key_mask(tokens, key='C', tolerance=1)
        
        assert weights[toy_vocab['KICK']] == 1.0, "KICK should maintain full weight"
        assert weights[toy_vocab['BAR']] == 1.0, "BAR should maintain full weight"
        assert weights[toy_vocab['CHORD_C']] == 1.0, "CHORD_C should maintain full weight"


class TestGrooveMask:
    """Tests for groove_mask function"""
    
    def test_kick_emphasis_on_strong_beats(self, toy_vocab):
        """Test that KICK is emphasized on beats 1 and 3"""
        tokens = TestTokens(toy_vocab)
        
        # Test beat 1 (position 0)
        template_beat1 = {
            'drum_pattern': {'kick': [0, 8], 'snare': [4, 12]},
            'time_feel': 'straight',
            'current_pos': 0
        }
        weights_beat1 = groove_mask(tokens, template_beat1)
        
        # Test beat 3 (position 8)
        template_beat3 = {
            'drum_pattern': {'kick': [0, 8], 'snare': [4, 12]},
            'time_feel': 'straight',
            'current_pos': 8
        }
        weights_beat3 = groove_mask(tokens, template_beat3)
        
        assert weights_beat1[toy_vocab['KICK']] > 1.0, "KICK should be emphasized on beat 1"
        assert weights_beat3[toy_vocab['KICK']] > 1.0, "KICK should be emphasized on beat 3"
    
    def test_snare_emphasis_on_backbeats(self, toy_vocab):
        """Test that SNARE is emphasized on beats 2 and 4"""
        tokens = TestTokens(toy_vocab)
        
        # Test beat 2 (position 4)
        template_beat2 = {
            'drum_pattern': {'kick': [0, 8], 'snare': [4, 12]},
            'time_feel': 'straight',
            'current_pos': 4
        }
        weights = groove_mask(tokens, template_beat2)
        
        assert weights[toy_vocab['SNARE']] > 1.0, "SNARE should be emphasized on beat 2"
        assert weights[toy_vocab['KICK']] < 1.0, "KICK should be de-emphasized on beat 2"
    
    def test_swing_feel_affects_timing(self, toy_vocab):
        """Test that swing feel changes hihat timing"""
        tokens = TestTokens(toy_vocab)
        
        straight_template = {
            'time_feel': 'straight',
            'current_pos': 0
        }
        swing_template = {
            'time_feel': 'swing',
            'current_pos': 0
        }
        
        straight_weights = groove_mask(tokens, straight_template)
        swing_weights = groove_mask(tokens, swing_template)
        
        # Results should differ between straight and swing feel
        # (exact differences depend on implementation)
        assert not torch.equal(straight_weights, swing_weights), "Straight and swing should produce different weights"


class TestRepetitionPenalty:
    """Tests for repetition_penalty function"""
    
    def test_repeated_tokens_penalized(self, toy_vocab):
        """Test that repeated tokens receive penalty"""
        vocab_size = len(toy_vocab)
        logits = torch.randn(vocab_size)
        history = [toy_vocab['NOTE_ON_60'], toy_vocab['NOTE_ON_60'], toy_vocab['NOTE_ON_62']]
        
        penalized_logits = repetition_penalty(logits, history, gamma=1.5)
        
        # Repeated token should be penalized
        assert penalized_logits[toy_vocab['NOTE_ON_60']] < logits[toy_vocab['NOTE_ON_60']], \
            "Repeated NOTE_ON_60 should be penalized"
        
        # Non-repeated tokens should be less affected
        assert abs(penalized_logits[toy_vocab['NOTE_ON_64']] - logits[toy_vocab['NOTE_ON_64']]) < 0.1, \
            "Non-repeated tokens should be minimally affected"
    
    def test_empty_history_no_penalty(self, toy_vocab):
        """Test that empty history applies no penalty"""
        vocab_size = len(toy_vocab)
        logits = torch.randn(vocab_size)
        history = []
        
        penalized_logits = repetition_penalty(logits, history, gamma=1.5)
        
        assert torch.equal(logits, penalized_logits), "Empty history should apply no penalty"
    
    def test_gamma_strength_affects_penalty(self, toy_vocab):
        """Test that higher gamma values increase penalty"""
        vocab_size = len(toy_vocab)
        logits = torch.randn(vocab_size)
        history = [toy_vocab['NOTE_ON_60']] * 3  # Repeat 3 times
        
        mild_penalty = repetition_penalty(logits, history, gamma=1.2)
        strong_penalty = repetition_penalty(logits, history, gamma=2.0)
        
        # Stronger gamma should produce larger penalty
        assert strong_penalty[toy_vocab['NOTE_ON_60']] < mild_penalty[toy_vocab['NOTE_ON_60']], \
            "Higher gamma should produce stronger penalty"


class TestApplyAll:
    """Tests for apply_all function"""
    
    def test_apply_all_combines_constraints(self, toy_vocab, test_plan):
        """Test that apply_all combines all constraint types"""
        vocab_size = len(toy_vocab)
        logits = torch.randn(vocab_size)
        
        state = {
            'bar_idx': 2,  # INTRO section
            'history': [toy_vocab['NOTE_ON_60'], toy_vocab['NOTE_ON_60']],  # Repeated note
            'current_pos': 0  # Beat 1
        }
        
        constrained_logits = apply_all(logits, state, test_plan)
        
        # Output should have same shape
        assert constrained_logits.shape == logits.shape, "Output shape should match input"
        
        # Should apply section constraints (LEAD forbidden in INTRO)
        assert constrained_logits[toy_vocab['LEAD']] == -float('inf'), \
            "LEAD should be forbidden in INTRO"
        
        # Should apply repetition penalty
        assert constrained_logits[toy_vocab['NOTE_ON_60']] < logits[toy_vocab['NOTE_ON_60']], \
            "Repeated note should be penalized"
    
    def test_apply_all_handles_missing_state(self, toy_vocab, test_plan):
        """Test that apply_all handles missing state gracefully"""
        vocab_size = len(toy_vocab)
        logits = torch.randn(vocab_size)
        
        # Minimal state
        state = {}
        
        # Should not crash
        try:
            constrained_logits = apply_all(logits, state, test_plan)
            assert constrained_logits.shape == logits.shape
        except Exception as e:
            pytest.fail(f"apply_all should handle missing state gracefully: {e}")
    
    def test_apply_all_handles_missing_plan(self, toy_vocab):
        """Test that apply_all handles missing plan elements gracefully"""
        vocab_size = len(toy_vocab)
        logits = torch.randn(vocab_size)
        
        state = {'bar_idx': 0, 'history': [], 'current_pos': 0}
        minimal_plan = {'vocab': toy_vocab}  # Minimal plan
        
        # Should not crash
        try:
            constrained_logits = apply_all(logits, state, minimal_plan)
            assert constrained_logits.shape == logits.shape
        except Exception as e:
            pytest.fail(f"apply_all should handle minimal plan gracefully: {e}")


def test_integration_scenario():
    """Integration test with realistic scenario"""
    # Create expanded vocabulary
    vocab = {
        'PAD': 0, 'EOS': 1, 'BAR': 2,
        'NOTE_ON_60': 3, 'NOTE_ON_62': 4, 'NOTE_ON_64': 5, 'NOTE_ON_67': 6,  # C major scale
        'NOTE_ON_61': 7, 'NOTE_ON_63': 8, 'NOTE_ON_66': 9,  # Chromatic notes
        'KICK': 10, 'SNARE': 11, 'HIHAT': 12,
        'BASS_PICK': 13, 'ACOUSTIC_STRUM': 14, 'LEAD': 15, 'VOCAL': 16,
        'CHORD_C': 17, 'CHORD_F': 18, 'CHORD_G': 19,
        'SECTION_VERSE': 20, 'SECTION_CHORUS': 21
    }
    
    # Create comprehensive plan
    plan = {
        'sections': [
            {'type': 'INTRO', 'bars': 4},
            {'type': 'VERSE', 'bars': 8},
            {'type': 'CHORUS', 'bars': 8},
        ],
        'key': 'C',
        'vocab': vocab,
        'groove_template': {
            'drum_pattern': {'kick': [0, 8], 'snare': [4, 12]},
            'time_feel': 'straight'
        },
        'repetition_penalty': 1.4
    }
    
    # Simulate generation in VERSE section with some history
    vocab_size = len(vocab)
    logits = torch.randn(vocab_size)
    
    state = {
        'bar_idx': 6,  # In VERSE
        'history': [vocab['NOTE_ON_60'], vocab['KICK'], vocab['NOTE_ON_60']],  # Some repetition
        'current_pos': 4  # Beat 2 (snare position)
    }
    
    # Apply all constraints
    result = apply_all(logits, state, plan)
    
    # Verify results
    assert result.shape == logits.shape, "Shape should be preserved"
    
    # Chromatic notes should be penalized (key constraint)
    assert result[vocab['NOTE_ON_61']] < logits[vocab['NOTE_ON_61']], \
        "Chromatic note should be penalized in C major"
    
    # SNARE should be emphasized on beat 2 (groove constraint)
    assert result[vocab['SNARE']] > logits[vocab['SNARE']], \
        "SNARE should be emphasized on beat 2"
    
    # Repeated NOTE_ON_60 should be penalized (repetition constraint)
    assert result[vocab['NOTE_ON_60']] < logits[vocab['NOTE_ON_60']], \
        "Repeated note should be penalized"
    
    print("✓ Integration test passed successfully!")


if __name__ == "__main__":
    # Run specific tests manually
    import torch
    
    # Create fixtures manually for standalone execution
    toy_vocab = {
        'PAD': 0, 'EOS': 1, 'BAR': 2, 'POS_1': 3, 'POS_2': 4,
        'NOTE_ON_60': 5, 'NOTE_ON_62': 6, 'NOTE_ON_64': 7, 'NOTE_ON_67': 8,
        'NOTE_ON_61': 9, 'NOTE_ON_63': 10, 'NOTE_ON_66': 11,
        'KICK': 12, 'SNARE': 13, 'HIHAT': 14, 'BASS_PICK': 15,
        'CHORD_C': 16, 'CHORD_F': 17, 'CHORD_G': 18,
        'SECTION_VERSE': 19, 'SECTION_CHORUS': 20, 'LEAD': 21,
        'ACOUSTIC_STRUM': 22, 'PIANO': 23, 'VOCAL': 24
    }
    
    test_plan = {
        'sections': [
            {'type': 'INTRO', 'bars': 4},
            {'type': 'VERSE', 'bars': 8},
            {'type': 'CHORUS', 'bars': 8},
            {'type': 'BRIDGE', 'bars': 4},
            {'type': 'OUTRO', 'bars': 4}
        ],
        'key': 'C',
        'vocab': toy_vocab,
        'groove_template': {
            'drum_pattern': {'kick': [0, 8], 'snare': [4, 12]},
            'time_feel': 'straight',
            'emphasis': [0, 4, 8, 12]
        },
        'repetition_penalty': 1.3
    }
    
    print("Running constraint tests...")
    
    # Test section mask
    tokens = TestTokens(toy_vocab)
    mask = section_mask(tokens, bar_idx=2, plan=test_plan)
    assert mask[toy_vocab['LEAD']] == False
    print("✓ Section mask test passed")
    
    # Test key mask
    weights = key_mask(tokens, key='C', tolerance=1)
    assert weights[toy_vocab['NOTE_ON_60']] == 1.0
    print("✓ Key mask test passed")
    
    # Test groove mask
    groove_template = {
        'drum_pattern': {'kick': [0, 8], 'snare': [4, 12]},
        'time_feel': 'straight',
        'current_pos': 0
    }
    groove_weights = groove_mask(tokens, groove_template)
    assert groove_weights[toy_vocab['KICK']] > 1.0
    print("✓ Groove mask test passed")
    
    # Test repetition penalty
    logits = torch.randn(len(toy_vocab))
    history = [toy_vocab['NOTE_ON_60'], toy_vocab['NOTE_ON_60']]
    penalized = repetition_penalty(logits, history, gamma=1.5)
    assert penalized[toy_vocab['NOTE_ON_60']] < logits[toy_vocab['NOTE_ON_60']]
    print("✓ Repetition penalty test passed")
    
    # Test apply_all
    state = {
        'bar_idx': 2,
        'history': [toy_vocab['NOTE_ON_60']],
        'current_pos': 0
    }
    result = apply_all(logits, state, test_plan)
    assert result.shape == logits.shape
    print("✓ Apply all test passed")
    
    # Run integration test
    test_integration_scenario()
    
    print("\nAll tests passed successfully! 🎵")