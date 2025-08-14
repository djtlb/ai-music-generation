"""
Test suite for constraint utilities and repetition control

Tests the musical constraint generation and repetition control functionality.
"""

import unittest
import torch
import numpy as np
from typing import Dict, List

# Import modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.constraints import (
    MusicalConstraints,
    ConstraintMaskGenerator, 
    RepetitionController,
    create_combined_constraint_mask
)


class TestMusicalConstraints(unittest.TestCase):
    """Test basic musical constraint definitions"""
    
    def test_scale_notes(self):
        """Test scale note generation"""
        # Test C major scale
        c_major = MusicalConstraints.get_scale_notes(0, is_major=True)
        expected_c_major = [0, 2, 4, 5, 7, 9, 11]  # C D E F G A B
        self.assertEqual(c_major, expected_c_major)
        
        # Test A minor scale
        a_minor = MusicalConstraints.get_scale_notes(9, is_major=False)
        expected_a_minor = [9, 11, 0, 2, 4, 5, 7]  # A B C D E F G
        self.assertEqual(a_minor, expected_a_minor)
        
        # Test F# major scale
        fs_major = MusicalConstraints.get_scale_notes(6, is_major=True)
        expected_fs_major = [6, 8, 10, 11, 1, 3, 5]  # F# G# A# B C# D# E#(F)
        self.assertEqual(fs_major, expected_fs_major)
    
    def test_chord_notes(self):
        """Test chord note generation"""
        # Test C major triad
        c_major = MusicalConstraints.get_chord_notes(0, 'maj')
        expected_c_major = [0, 4, 7]  # C E G
        self.assertEqual(c_major, expected_c_major)
        
        # Test A minor triad
        a_minor = MusicalConstraints.get_chord_notes(9, 'min')
        expected_a_minor = [9, 0, 4]  # A C E
        self.assertEqual(a_minor, expected_a_minor)
        
        # Test G7 chord
        g7 = MusicalConstraints.get_chord_notes(7, 'dom7')
        expected_g7 = [7, 11, 2, 5]  # G B D F
        self.assertEqual(g7, expected_g7)
        
        # Test invalid chord type defaults to major
        invalid_chord = MusicalConstraints.get_chord_notes(0, 'invalid')
        expected_major = [0, 4, 7]
        self.assertEqual(invalid_chord, expected_major)


class TestConstraintMaskGenerator(unittest.TestCase):
    """Test constraint mask generation"""
    
    def setUp(self):
        """Set up test vocabulary and mask generator"""
        self.vocab = {
            'PAD': 0, 'EOS': 1, 'BAR': 2, 'POS_1': 3, 'POS_2': 4,
            'NOTE_ON_60': 5,  # C4
            'NOTE_ON_62': 6,  # D4
            'NOTE_ON_64': 7,  # E4
            'NOTE_ON_65': 8,  # F4
            'NOTE_ON_67': 9,  # G4
            'NOTE_ON_69': 10, # A4
            'NOTE_ON_71': 11, # B4
            'NOTE_ON_61': 12, # C#4
            'CHORD_C_maj': 13,
            'CHORD_F_maj': 14,
            'CHORD_G_maj': 15,
            'STYLE_rock_punk': 16,
            'SECTION_VERSE': 17
        }
        
        self.mask_generator = ConstraintMaskGenerator(self.vocab)
    
    def test_vocab_parsing(self):
        """Test vocabulary parsing"""
        # Check note tokens were parsed correctly
        expected_notes = {60: 5, 62: 6, 64: 7, 65: 8, 67: 9, 69: 10, 71: 11, 61: 12}
        self.assertEqual(self.mask_generator.note_tokens, expected_notes)
        
        # Check chord tokens
        expected_chords = {'C_maj': 13, 'F_maj': 14, 'G_maj': 15}
        self.assertEqual(self.mask_generator.chord_tokens, expected_chords)
        
        # Check control tokens
        self.assertIn('STYLE_rock_punk', self.mask_generator.control_tokens)
        self.assertIn('SECTION_VERSE', self.mask_generator.control_tokens)
    
    def test_scale_constraint_mask(self):
        """Test scale constraint mask generation"""
        # Test C major scale constraint
        mask = self.mask_generator.create_scale_constraint_mask(
            key=0, is_major=True, seq_len=10, penalty_value=-5.0
        )
        
        # Check shape
        self.assertEqual(mask.shape, (10, len(self.vocab)))
        
        # C major scale notes: C(0), D(2), E(4), F(5), G(7), A(9), B(11)
        # In our vocab: C4(60%12=0), D4(62%12=2), E4(64%12=4), F4(65%12=5), G4(67%12=7), A4(69%12=9), B4(71%12=11)
        # Out of scale: C#4(61%12=1)
        
        # Check that C#4 (token 12) gets penalty
        self.assertEqual(mask[0, 12].item(), -5.0)
        
        # Check that C4 (token 5) doesn't get penalty
        self.assertEqual(mask[0, 5].item(), 0.0)
    
    def test_chord_constraint_mask(self):
        """Test chord constraint mask generation"""
        chord_sequence = ['C_maj', 'F_maj']
        mask = self.mask_generator.create_chord_constraint_mask(
            chord_sequence, seq_len=8, penalty_value=-3.0
        )
        
        # Check shape
        self.assertEqual(mask.shape, (8, len(self.vocab)))
        
        # First half should have C major chord constraint (C-E-G: 0-4-7)
        # Second half should have F major chord constraint (F-A-C: 5-9-0)
        
        # In first half, non-chord tones should be penalized
        # D4 (token 6, note class 2) is not in C major chord
        self.assertEqual(mask[0, 6].item(), -3.0)
        
        # C4 (token 5, note class 0) is in C major chord
        self.assertEqual(mask[0, 5].item(), 0.0)
    
    def test_repetition_constraint_mask(self):
        """Test repetition constraint mask generation"""
        # Create sequence with repetition
        sequence = torch.tensor([5, 6, 5, 7, 5])  # NOTE_ON_60 appears 3 times
        
        mask = self.mask_generator.create_repetition_constraint_mask(
            sequence, current_position=5, window_size=4, max_repetitions=2, penalty_value=-2.0
        )
        
        # Check shape
        self.assertEqual(mask.shape, (len(self.vocab),))
        
        # Token 5 appears 3 times (>= max_repetitions=2), should be penalized
        self.assertEqual(mask[5].item(), -2.0)
        
        # Token 6 appears 1 time (< max_repetitions), should not be penalized
        self.assertEqual(mask[6].item(), 0.0)
    
    def test_style_constraint_mask(self):
        """Test style-specific constraint mask generation"""
        mask = self.mask_generator.create_style_constraint_mask(
            style='rock_punk', section='verse', key=0, seq_len=10
        )
        
        # Check shape
        self.assertEqual(mask.shape, (10, len(self.vocab)))
        
        # Rock punk style should prefer pentatonic scale
        # Pentatonic: C(0), D(2), E(4), G(7), A(9)
        # F(5) and B(11) should be penalized
        
        # F4 (token 8, note class 5) should be penalized in rock_punk
        self.assertEqual(mask[0, 8].item(), -2.0)
        
        # C4 (token 5, note class 0) should not be penalized
        self.assertEqual(mask[0, 5].item(), 0.0)


class TestRepetitionController(unittest.TestCase):
    """Test repetition control functionality"""
    
    def setUp(self):
        """Set up repetition controller"""
        self.vocab_size = 20
        self.controller = RepetitionController(self.vocab_size)
        
        # Vocabulary for testing
        self.vocab = {
            'PAD': 0, 'EOS': 1, 'BAR': 2, 'NOTE_ON_60': 5, 'NOTE_ON_62': 6
        }
    
    def test_token_repetition_penalty(self):
        """Test token-level repetition penalty"""
        # Add some repeated tokens
        for _ in range(3):
            self.controller.update(5)  # Repeat token 5
        self.controller.update(6)  # Token 6 once
        
        penalty = self.controller.get_repetition_penalty(
            window_size=10, token_penalty=-1.0, phrase_penalty=0.0
        )
        
        # Check shape
        self.assertEqual(penalty.shape, (self.vocab_size,))
        
        # Token 5 should have higher penalty (appeared 3 times)
        self.assertLess(penalty[5].item(), penalty[6].item())
        
        # Token 6 appeared once, should have some penalty
        self.assertLess(penalty[6].item(), 0.0)
    
    def test_phrase_boundary_detection(self):
        """Test phrase boundary detection"""
        # Test with BAR token
        is_boundary = self.controller.detect_phrase_boundary(2, self.vocab)  # BAR token
        self.assertTrue(is_boundary)
        
        # Test with non-boundary token
        is_boundary = self.controller.detect_phrase_boundary(5, self.vocab)  # NOTE_ON_60
        self.assertFalse(is_boundary)
    
    def test_phrase_repetition_penalty(self):
        """Test phrase-level repetition penalty"""
        # Create a repeated phrase pattern
        phrase1 = [5, 6, 7]
        phrase2 = [8, 9, 10]
        
        # Add first phrase
        for token in phrase1:
            self.controller.update(token)
        self.controller.update(2, is_phrase_boundary=True)  # End phrase
        
        # Add second phrase
        for token in phrase2:
            self.controller.update(token)
        self.controller.update(2, is_phrase_boundary=True)  # End phrase
        
        # Start repeating first phrase
        for token in phrase1[:2]:  # Add first two tokens
            self.controller.update(token)
        
        # Get penalty - should penalize continuing the repeated phrase
        penalty = self.controller.get_repetition_penalty(
            phrase_window=2, phrase_penalty=-5.0, token_penalty=0.0
        )
        
        # Token 7 (next in repeated phrase) should be penalized
        self.assertEqual(penalty[7].item(), -5.0)
    
    def test_reset_functionality(self):
        """Test controller reset"""
        # Add some history
        self.controller.update(5)
        self.controller.update(6, is_phrase_boundary=True)
        
        # Check history exists
        self.assertGreater(len(self.controller.token_history), 0)
        self.assertGreater(len(self.controller.phrase_history), 0)
        
        # Reset
        self.controller.reset()
        
        # Check history is cleared
        self.assertEqual(len(self.controller.token_history), 0)
        self.assertEqual(len(self.controller.phrase_history), 0)
        self.assertEqual(len(self.controller.current_phrase), 0)


class TestCombinedConstraints(unittest.TestCase):
    """Test combined constraint mask functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.vocab = {
            'PAD': 0, 'EOS': 1, 'BAR': 2,
            'NOTE_ON_60': 5, 'NOTE_ON_62': 6, 'NOTE_ON_64': 7,
            'CHORD_C_maj': 8, 'STYLE_rock_punk': 9
        }
        
        self.mask_generator = ConstraintMaskGenerator(self.vocab)
        self.repetition_controller = RepetitionController(len(self.vocab))
    
    def test_combined_constraint_mask(self):
        """Test creation of combined constraint mask"""
        # Set up parameters
        key = 0  # C
        chord_sequence = ['C_maj']
        style = 'rock_punk'
        section = 'verse'
        generated_sequence = torch.tensor([5, 6, 5])
        current_position = 3
        seq_len = 10
        
        # Create combined mask
        combined_mask = create_combined_constraint_mask(
            mask_generator=self.mask_generator,
            repetition_controller=self.repetition_controller,
            key=key,
            chord_sequence=chord_sequence,
            style=style,
            section=section,
            generated_sequence=generated_sequence,
            current_position=current_position,
            seq_len=seq_len
        )
        
        # Check shape
        self.assertEqual(combined_mask.shape, (len(self.vocab),))
        
        # Check that mask contains penalty values (should be negative)
        # Some tokens should be penalized due to various constraints
        has_penalties = (combined_mask < 0).any()
        self.assertTrue(has_penalties)


class TestConstraintIntegration(unittest.TestCase):
    """Integration tests for constraint system"""
    
    def test_musical_realism(self):
        """Test that constraints produce musically realistic results"""
        vocab = {
            'PAD': 0, 'EOS': 1, 'BAR': 2,
            'NOTE_ON_60': 5,  # C4
            'NOTE_ON_61': 6,  # C#4
            'NOTE_ON_62': 7,  # D4
            'NOTE_ON_64': 8,  # E4
            'NOTE_ON_65': 9,  # F4
            'NOTE_ON_67': 10, # G4
            'NOTE_ON_69': 11, # A4
            'NOTE_ON_71': 12, # B4
        }
        
        mask_gen = ConstraintMaskGenerator(vocab)
        
        # Test C major scale constraint
        scale_mask = mask_gen.create_scale_constraint_mask(key=0, is_major=True, seq_len=5)
        
        # In C major, C# should be penalized more than C
        c_penalty = scale_mask[0, 5].item()    # C4
        cs_penalty = scale_mask[0, 6].item()   # C#4
        
        self.assertLess(cs_penalty, c_penalty)  # C# should have more penalty (more negative)
    
    def test_style_consistency(self):
        """Test that different styles produce different constraints"""
        vocab = {'NOTE_ON_60': 0, 'NOTE_ON_61': 1, 'NOTE_ON_65': 2}  # C, C#, F
        mask_gen = ConstraintMaskGenerator(vocab)
        
        # Rock punk should penalize certain notes differently than country pop
        rock_mask = mask_gen.create_style_constraint_mask('rock_punk', 'verse', 0, 5)
        country_mask = mask_gen.create_style_constraint_mask('country_pop', 'verse', 0, 5)
        
        # Masks should be different
        self.assertFalse(torch.equal(rock_mask, country_mask))


def run_constraint_tests():
    """Run all constraint-related tests"""
    print("Running constraint utility tests...")
    
    # Create test suite
    test_classes = [
        TestMusicalConstraints,
        TestConstraintMaskGenerator,
        TestRepetitionController,
        TestCombinedConstraints,
        TestConstraintIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nTesting {test_class.__name__}:")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        passed_tests += result.testsRun - len(result.failures) - len(result.errors)
        
        if result.failures:
            print(f"Failures in {test_class.__name__}:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        
        if result.errors:
            print(f"Errors in {test_class.__name__}:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    print(f"\nTest Summary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("All constraint tests passed! ✅")
        return True
    else:
        print("Some tests failed! ❌")
        return False


if __name__ == '__main__':
    success = run_constraint_tests()
    exit(0 if success else 1)