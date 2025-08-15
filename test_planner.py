#!/usr/bin/env python3
"""
Unit tests for the music planner module.
Tests various free-text prompts and validates stable JSON output with tolerances.
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from planner import MusicPlanner


class TestMusicPlanner(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment with sample configs."""
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.configs_dir = cls.temp_dir / "configs"
        cls._create_sample_configs()
        cls.planner = MusicPlanner(str(cls.configs_dir))
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _create_sample_configs(cls):
        """Create minimal sample configs for testing."""
        # Create directory structure
        genres_dir = cls.configs_dir / "genres"
        styles_dir = cls.configs_dir / "styles"
        genres_dir.mkdir(parents=True)
        styles_dir.mkdir(parents=True)
        
        # Sample pop genre config
        pop_config = {
            'name': 'pop',
            'display_name': 'Pop',
            'bpm': {'min': 85, 'max': 135, 'preferred': 120},
            'keys': {'preferred': ['C', 'G', 'Am', 'F']},
            'instruments': {'required': ['KICK', 'SNARE', 'BASS_PICK', 'PIANO', 'LEAD']},
            'structure': {
                'common_forms': [['INTRO', 'VERSE', 'CHORUS', 'VERSE', 'CHORUS', 'OUTRO']],
                'section_lengths': {'INTRO': 8, 'VERSE': 16, 'CHORUS': 16, 'OUTRO': 8}
            },
            'mix_targets': {'lufs': -11.0, 'spectral_centroid_hz': 2200},
            'groove': {'swing': 0.0, 'pocket': 'tight', 'energy': 'high'},
            'qa_thresholds': {'min_hook_strength': 0.7}
        }
        
        with open(genres_dir / "pop.yaml", 'w') as f:
            import yaml
            yaml.dump(pop_config, f)
        
        # Sample rock genre config
        rock_config = {
            'name': 'rock',
            'display_name': 'Rock',
            'bpm': {'min': 100, 'max': 180, 'preferred': 140},
            'keys': {'preferred': ['E', 'A', 'D', 'G']},
            'instruments': {'required': ['KICK', 'SNARE', 'BASS_PICK', 'GUITAR_RHYTHM', 'GUITAR_LEAD']},
            'structure': {
                'section_lengths': {'INTRO': 8, 'VERSE': 16, 'CHORUS': 16, 'BRIDGE': 16, 'OUTRO': 8}
            },
            'mix_targets': {'lufs': -9.5, 'spectral_centroid_hz': 2800},
            'groove': {'swing': 0.0, 'pocket': 'loose', 'energy': 'high'}
        }
        
        with open(genres_dir / "rock.yaml", 'w') as f:
            yaml.dump(rock_config, f)
        
        # Sample dance_pop style config (child of pop)
        pop_styles_dir = styles_dir / "pop"
        pop_styles_dir.mkdir()
        
        dance_pop_config = {
            'parent': 'pop',
            'name': 'dance_pop',
            'display_name': 'Dance Pop',
            'bpm': {'min': 110, 'max': 135, 'preferred': 128},
            'mix_targets': {'lufs': -8.5, 'spectral_centroid_hz': 2500}
        }
        
        with open(pop_styles_dir / "dance_pop.yaml", 'w') as f:
            yaml.dump(dance_pop_config, f)
    
    def test_basic_pop_song(self):
        """Test basic pop song planning."""
        lyrics = """
        VERSE: Walking down the street tonight
        Feeling like everything's alright
        CHORUS: We can fly, we can touch the sky
        Nothing's gonna stop us now
        """
        
        genre = "pop, 120 bpm, bright energy"
        
        result = self.planner.plan(lyrics, genre)
        
        # Validate required fields
        self.assertIn('style', result)
        self.assertIn('bpm', result)
        self.assertIn('key', result)
        self.assertIn('arrangement', result)
        self.assertIn('drum_template', result)
        self.assertIn('hook_type', result)
        self.assertIn('mix_targets', result)
        self.assertIn('lyrics_sections', result)
        
        # Validate values
        self.assertEqual(result['style'], 'pop')
        self.assertEqual(result['bpm'], 120)
        self.assertIn('VERSE', [s['type'] for s in result['lyrics_sections']])
        self.assertIn('CHORUS', [s['type'] for s in result['lyrics_sections']])
        self.assertEqual(result['hook_type'], 'chorus_hook')
    
    def test_dance_pop_with_explicit_bpm(self):
        """Test dance pop with explicit BPM extraction."""
        lyrics = "energetic dance track with big hooks"
        genre = "dance pop, 128 bpm, four on the floor, club energy"
        
        result = self.planner.plan(lyrics, genre)
        
        self.assertEqual(result['style'], 'pop/dance_pop')
        self.assertEqual(result['bpm'], 128)
        self.assertEqual(result['drum_template'], 'four_on_floor')
    
    def test_rock_punk_song(self):
        """Test rock punk song planning."""
        lyrics = """
        verse: anger in the streets
        chorus: fight the power now
        verse: we won't back down
        chorus: fight the power now
        """
        
        genre = "punk rock, 160 bpm, aggressive, distorted"
        
        result = self.planner.plan(lyrics, genre)
        
        self.assertEqual(result['style'], 'rock/punk')
        self.assertEqual(result['bpm'], 160)
        self.assertEqual(result['drum_template'], 'punk_drums')
    
    def test_rnb_ballad_with_key(self):
        """Test R&B ballad with key signature."""
        lyrics = """
        VERSE: Love is all we need tonight
        CHORUS: Hold me close, don't let me go
        BRIDGE: When the world gets cold
        CHORUS: Hold me close, don't let me go
        """
        
        genre = "r&b ballad, 75 bpm, in D minor, lush pads"
        
        result = self.planner.plan(lyrics, genre)
        
        self.assertEqual(result['style'], 'rnb_soul/ballad')
        self.assertEqual(result['bpm'], 75)
        self.assertEqual(result['key'], 'Dm')
        self.assertTrue(result['arrangement']['structure'])
        
    def test_drill_halftime_feel(self):
        """Test drill with halftime feel."""
        lyrics = "dark drill track with sliding 808s"
        genre = "drill, 140 bpm halftime, dark piano, UK drill style"
        
        result = self.planner.plan(lyrics, genre)
        
        self.assertEqual(result['style'], 'hiphop_rap/drill')
        self.assertEqual(result['bpm'], 140)
        self.assertEqual(result['time_feel'], 'halftime')
        self.assertEqual(result['drum_template'], 'drill_pattern_halftime')
    
    def test_minimal_input(self):
        """Test with minimal input."""
        lyrics = ""
        genre = "pop"
        
        result = self.planner.plan(lyrics, genre)
        
        # Should still return valid JSON with defaults
        self.assertEqual(result['style'], 'pop')
        self.assertIsInstance(result['bpm'], int)
        self.assertIn(result['bpm'], range(60, 201))  # Valid BPM range
        self.assertTrue(result['arrangement']['structure'])
        self.assertEqual(result['hook_type'], 'chorus_hook')
    
    def test_no_explicit_structure(self):
        """Test with lyrics but no explicit structure markers."""
        lyrics = """
        Walking down the street at night
        Feeling everything is right
        We can fly so high tonight
        Nothing stops us from the light
        Remember when we used to dream
        Life was more than what it seems
        We can fly so high tonight
        Nothing stops us from the light
        """
        
        genre = "indie pop, medium tempo"
        
        result = self.planner.plan(lyrics, genre)
        
        self.assertEqual(result['style'], 'pop/indie_pop')
        # Should infer structure since there's substantial content
        self.assertTrue(len(result['arrangement']['structure']) >= 4)
    
    def test_country_pop_crossover(self):
        """Test country pop crossover detection."""
        lyrics = """
        VERSE: Driving down that dusty road
        CHORUS: Country heart but city soul
        """
        
        genre = "country pop, acoustic guitar, 110 bpm"
        
        result = self.planner.plan(lyrics, genre)
        
        self.assertEqual(result['style'], 'country/pop')
        self.assertEqual(result['bpm'], 110)
        
    def test_bpm_constraints(self):
        """Test BPM constraint validation."""
        lyrics = "test song"
        
        # Test too low BPM
        genre = "slow ballad, 40 bpm"
        result = self.planner.plan(lyrics, genre)
        self.assertGreaterEqual(result['bpm'], 60)
        
        # Test too high BPM
        genre = "speed metal, 250 bpm"
        result = self.planner.plan(lyrics, genre)
        self.assertLessEqual(result['bpm'], 200)
    
    def test_swing_feel(self):
        """Test swing time feel detection."""
        lyrics = "jazz influenced track"
        genre = "jazz pop, swing feel, 100 bpm"
        
        result = self.planner.plan(lyrics, genre)
        
        self.assertEqual(result['time_feel'], 'swing')
        self.assertTrue(result['drum_template'].endswith('_swing'))
    
    def test_bridge_detection(self):
        """Test bridge section detection."""
        lyrics = """
        VERSE: First verse here
        CHORUS: Main hook here
        VERSE: Second verse
        CHORUS: Main hook again
        BRIDGE: Different section for contrast
        CHORUS: Final hook
        """
        
        genre = "pop rock"
        
        result = self.planner.plan(lyrics, genre)
        
        # Should detect bridge in structure
        self.assertIn('BRIDGE', result['arrangement']['structure'])
        bridge_section = next((s for s in result['lyrics_sections'] if s['type'] == 'BRIDGE'), None)
        self.assertIsNotNone(bridge_section)
    
    def test_stability_multiple_runs(self):
        """Test that multiple runs with same input produce consistent results."""
        lyrics = """
        VERSE: Consistent test song
        CHORUS: Should be stable output
        """
        genre = "pop, 120 bpm, major key"
        
        results = []
        for _ in range(5):
            result = self.planner.plan(lyrics, genre)
            results.append(result)
        
        # Key fields should be identical across runs
        for result in results[1:]:
            self.assertEqual(result['style'], results[0]['style'])
            self.assertEqual(result['bpm'], results[0]['bpm'])
            self.assertEqual(result['key'], results[0]['key'])
            self.assertEqual(result['time_feel'], results[0]['time_feel'])
            self.assertEqual(result['drum_template'], results[0]['drum_template'])
    
    def test_total_bars_calculation(self):
        """Test total bars calculation."""
        lyrics = """
        INTRO: Starting up
        VERSE: First verse content
        CHORUS: Main hook
        VERSE: Second verse
        CHORUS: Hook again
        OUTRO: Ending
        """
        
        genre = "pop"
        
        result = self.planner.plan(lyrics, genre)
        
        # Should calculate total bars based on structure
        expected_bars = 8 + 16 + 16 + 16 + 16 + 8  # Intro + Verse + Chorus + Verse + Chorus + Outro
        self.assertEqual(result['arrangement']['total_bars'], expected_bars)
    
    def test_tolerances_and_ranges(self):
        """Test that outputs fall within acceptable tolerances."""
        test_cases = [
            ("pop dance track", "dance pop, 128 bpm"),
            ("slow ballad", "r&b ballad, 70 bpm"),
            ("energetic rock", "rock, 150 bpm"),
            ("country song", "country, 100 bpm"),
        ]
        
        for lyrics, genre in test_cases:
            result = self.planner.plan(lyrics, genre)
            
            # BPM should be reasonable
            self.assertIn(result['bpm'], range(60, 201))
            
            # Should have valid structure
            self.assertIsInstance(result['arrangement']['structure'], list)
            self.assertTrue(len(result['arrangement']['structure']) > 0)
            
            # Should have valid key
            self.assertRegex(result['key'], r'^[A-G][#b]?m?$')
            
            # Should have drum template
            self.assertIsInstance(result['drum_template'], str)
            self.assertTrue(len(result['drum_template']) > 0)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty inputs
        result = self.planner.plan("", "")
        self.assertIsInstance(result, dict)
        self.assertIn('style', result)
        
        # Very long inputs (should truncate gracefully)
        long_lyrics = "test verse " * 1000
        long_genre = "pop music " * 100
        result = self.planner.plan(long_lyrics, long_genre)
        self.assertIsInstance(result, dict)
        
        # Special characters
        lyrics_special = "VERSE: Test with émotions & symbols 123!@#"
        genre_special = "pop with 🎵 unicode"
        result = self.planner.plan(lyrics_special, genre_special)
        self.assertIsInstance(result, dict)
    
    def test_json_serializable(self):
        """Test that all outputs are JSON serializable."""
        test_cases = [
            ("basic pop song", "pop"),
            ("rock anthem", "rock, 140 bpm"),
            ("hip hop track", "hip hop, 100 bpm"),
        ]
        
        for lyrics, genre in test_cases:
            result = self.planner.plan(lyrics, genre)
            
            # Should be JSON serializable
            try:
                json_str = json.dumps(result)
                parsed_back = json.loads(json_str)
                self.assertEqual(result, parsed_back)
            except (TypeError, ValueError) as e:
                self.fail(f"Result not JSON serializable: {e}")


class TestPlannerIntegration(unittest.TestCase):
    """Integration tests with real config files if available."""
    
    def setUp(self):
        # Try to use real configs if they exist
        real_configs = Path("configs")
        if real_configs.exists():
            self.planner = MusicPlanner("configs")
        else:
            self.skipTest("Real configs not available")
    
    def test_real_config_integration(self):
        """Test with real configuration files."""
        lyrics = """
        VERSE: This is a real test
        CHORUS: Using actual configs
        """
        genre = "pop rock, 125 bpm, energetic"
        
        result = self.planner.plan(lyrics, genre)
        
        # Should work with real configs
        self.assertIsInstance(result, dict)
        self.assertIn('style', result)
        self.assertIn('bpm', result)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)