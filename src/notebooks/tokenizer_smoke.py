#!/usr/bin/env python3
"""
MIDI Tokenizer Smoke Test (Python Implementation)
=================================================

This is a Python implementation equivalent to the TypeScript tokenizer
for compatibility with Python-based ML pipelines.

Since this is a web-based environment, this serves as a reference
implementation showing how the tokenizer would work in Python.
"""

import json
import time
from typing import Dict, List, Any, Tuple, Optional

class MidiTokenizer:
    """Python implementation of the MIDI tokenizer for ML training"""
    
    def __init__(self):
        self.vocab = self._create_vocabulary()
        self.token_to_id = {}
        self.id_to_token = {}
        self._build_token_maps()
    
    def _create_vocabulary(self) -> Dict[str, List]:
        """Create the vocabulary matching the TypeScript implementation"""
        return {
            'styles': ['rock_punk', 'rnb_ballad', 'country_pop'],
            'sections': ['INTRO', 'VERSE', 'CHORUS', 'BRIDGE', 'OUTRO'],
            'instruments': [
                'KICK', 'SNARE', 'HIHAT', 'CRASH', 'RIDE',
                'BASS_PICK', 'BASS_SLAP', 
                'ACOUSTIC_STRUM', 'ELECTRIC_POWER', 'ELECTRIC_CLEAN',
                'PIANO', 'SYNTH_PAD', 'SYNTH_LEAD',
                'LEAD', 'HARMONY'
            ],
            'tempos': list(range(60, 261, 2)),  # 60-260 BPM in steps of 2
            'keys': [
                'C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B',
                'Cm', 'C#m', 'Dbm', 'Dm', 'D#m', 'Ebm', 'Em', 'Fm', 'F#m', 'Gbm', 'Gm', 'G#m', 'Abm', 'Am', 'A#m', 'Bbm', 'Bm'
            ],
            'positions': [f'POS_{i}' for i in range(64)],
            'pitches': list(range(128)),  # MIDI pitch range 0-127
            'velocities': [30, 50, 70, 90, 110, 125],
            'durations': ['DUR_1', 'DUR_2', 'DUR_3', 'DUR_4', 'DUR_6', 'DUR_8', 'DUR_16'],
            'chords': [
                # Major triads
                'C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B',
                # Minor triads
                'Cm', 'C#m', 'Dbm', 'Dm', 'D#m', 'Ebm', 'Em', 'Fm', 'F#m', 'Gbm', 'Gm', 'G#m', 'Abm', 'Am', 'A#m', 'Bbm', 'Bm',
                # Seventh chords
                'C7', 'Cm7', 'Cmaj7', 'D7', 'Dm7', 'Dmaj7', 'E7', 'Em7', 'Emaj7', 'F7', 'Fm7', 'Fmaj7',
                'G7', 'Gm7', 'Gmaj7', 'A7', 'Am7', 'Amaj7', 'B7', 'Bm7', 'Bmaj7',
                # Extended chords
                'C9', 'Cm9', 'C11', 'C13', 'Cadd9', 'Csus2', 'Csus4', 'Cdim', 'Caug'
            ],
            'special': ['<PAD>', '<START>', '<END>', '<UNK>', 'BAR', 'NOTE_ON', 'NOTE_OFF']
        }
    
    def _build_token_maps(self):
        """Build bidirectional token-ID mappings"""
        token_id = 0
        
        for category, tokens in self.vocab.items():
            for token in tokens:
                token_str = str(token)
                prefixed_token = token_str if category == 'special' else f"{category.upper()}_{token_str}"
                
                self.token_to_id[prefixed_token] = token_id
                self.id_to_token[token_id] = prefixed_token
                token_id += 1
    
    def encode(self, midi: Dict[str, Any]) -> List[int]:
        """Encode MIDI to token sequence"""
        tokens = ['<START>']
        
        # Add metadata
        tokens.extend([
            f"STYLES_{midi['style']}",
            f"TEMPOS_{midi['tempo']}",
            f"KEYS_{midi['key']}"
        ])
        
        # Create timeline of events
        events = []
        
        # Add section markers
        for section in midi.get('sections', []):
            events.append({
                'type': 'SECTION',
                'time': section['start'] * 16,
                'data': section['type']
            })
        
        # Add note events
        for track_name, notes in midi.get('tracks', {}).items():
            for note in notes:
                # Note on
                events.append({
                    'type': 'NOTE_ON',
                    'time': note['start'],
                    'data': {
                        'track': note['track'],
                        'pitch': note['pitch'],
                        'velocity': self._bucket_velocity(note['velocity'])
                    }
                })
                
                # Note off
                events.append({
                    'type': 'NOTE_OFF',
                    'time': note['start'] + note['duration'],
                    'data': {
                        'track': note['track'],
                        'pitch': note['pitch'],
                        'duration': self._bucket_duration(note['duration'])
                    }
                })
        
        # Sort events by time
        events.sort(key=lambda x: x['time'])
        
        # Convert events to tokens
        current_bar = 0
        current_position = 0
        
        for event in events:
            event_bar = event['time'] // 16
            event_pos = event['time'] % 16
            
            # Add bar token if needed
            if event_bar > current_bar:
                tokens.append('BAR')
                current_bar = event_bar
                current_position = 0
            
            # Add position token if needed
            if event_pos > current_position:
                tokens.append(f"POSITIONS_POS_{event_pos}")
                current_position = event_pos
            
            # Add event-specific tokens
            if event['type'] == 'SECTION':
                tokens.append(f"SECTIONS_{event['data']}")
            elif event['type'] == 'NOTE_ON':
                tokens.extend([
                    'NOTE_ON',
                    f"INSTRUMENTS_{event['data']['track']}",
                    f"PITCHES_{event['data']['pitch']}",
                    f"VELOCITIES_{event['data']['velocity']}"
                ])
            elif event['type'] == 'NOTE_OFF':
                tokens.extend([
                    'NOTE_OFF',
                    f"INSTRUMENTS_{event['data']['track']}",
                    f"PITCHES_{event['data']['pitch']}",
                    f"DURATIONS_{event['data']['duration']}"
                ])
        
        tokens.append('<END>')
        
        # Convert to IDs
        return [self.token_to_id.get(token, self.token_to_id.get('<UNK>', 0)) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> Dict[str, Any]:
        """Decode token sequence back to MIDI"""
        tokens = [self.id_to_token.get(id, '<UNK>') for id in token_ids]
        
        result = {
            'style': 'rock_punk',
            'tempo': 120,
            'key': 'C',
            'sections': [],
            'tracks': {}
        }
        
        current_time = 0
        current_bar = 0
        active_notes = {}
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token.startswith('STYLES_'):
                result['style'] = token.replace('STYLES_', '')
            elif token.startswith('TEMPOS_'):
                result['tempo'] = int(token.replace('TEMPOS_', ''))
            elif token.startswith('KEYS_'):
                result['key'] = token.replace('KEYS_', '')
            elif token.startswith('SECTIONS_'):
                section_type = token.replace('SECTIONS_', '')
                result['sections'].append({
                    'type': section_type,
                    'start': current_bar,
                    'length': 4
                })
            elif token == 'BAR':
                current_bar += 1
                current_time = current_bar * 16
            elif token.startswith('POSITIONS_POS_'):
                pos = int(token.replace('POSITIONS_POS_', ''))
                current_time = current_bar * 16 + pos
            elif token == 'NOTE_ON' and i + 3 < len(tokens):
                track_token = tokens[i + 1]
                pitch_token = tokens[i + 2]
                velocity_token = tokens[i + 3]
                
                if (track_token.startswith('INSTRUMENTS_') and 
                    pitch_token.startswith('PITCHES_') and 
                    velocity_token.startswith('VELOCITIES_')):
                    
                    track = track_token.replace('INSTRUMENTS_', '')
                    pitch = int(pitch_token.replace('PITCHES_', ''))
                    velocity = int(velocity_token.replace('VELOCITIES_', ''))
                    
                    note_key = f"{track}_{pitch}"
                    active_notes[note_key] = {
                        'track': track,
                        'pitch': pitch,
                        'start': current_time,
                        'velocity': velocity
                    }
                    i += 3
            elif token == 'NOTE_OFF' and i + 3 < len(tokens):
                track_token = tokens[i + 1]
                pitch_token = tokens[i + 2]
                duration_token = tokens[i + 3]
                
                if (track_token.startswith('INSTRUMENTS_') and 
                    pitch_token.startswith('PITCHES_') and 
                    duration_token.startswith('DURATIONS_')):
                    
                    track = track_token.replace('INSTRUMENTS_', '')
                    pitch = int(pitch_token.replace('PITCHES_', ''))
                    duration = self._unbucket_duration(duration_token.replace('DURATIONS_', ''))
                    
                    note_key = f"{track}_{pitch}"
                    active_note = active_notes.get(note_key)
                    
                    if active_note:
                        if track not in result['tracks']:
                            result['tracks'][track] = []
                        
                        result['tracks'][track].append({
                            'pitch': active_note['pitch'],
                            'velocity': active_note['velocity'],
                            'start': active_note['start'],
                            'duration': duration,
                            'track': active_note['track']
                        })
                        
                        del active_notes[note_key]
                    i += 3
            
            i += 1
        
        return result
    
    def _bucket_velocity(self, velocity: int) -> int:
        """Bucket velocity into 6 dynamic levels"""
        if velocity < 40: return 30
        if velocity < 60: return 50
        if velocity < 80: return 70
        if velocity < 100: return 90
        if velocity < 120: return 110
        return 125
    
    def _bucket_duration(self, duration: int) -> str:
        """Bucket duration into musical note values"""
        if duration <= 1: return 'DUR_1'
        if duration <= 2: return 'DUR_2'
        if duration <= 3: return 'DUR_3'
        if duration <= 4: return 'DUR_4'
        if duration <= 6: return 'DUR_6'
        if duration <= 8: return 'DUR_8'
        return 'DUR_16'
    
    def _unbucket_duration(self, duration_token: str) -> int:
        """Convert bucketed duration back to numeric value"""
        duration_map = {
            'DUR_1': 1, 'DUR_2': 2, 'DUR_3': 3, 'DUR_4': 4,
            'DUR_6': 6, 'DUR_8': 8, 'DUR_16': 16
        }
        return duration_map.get(duration_token, 4)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.token_to_id)


def run_smoke_test():
    """Run comprehensive smoke test"""
    print("=== MIDI Tokenizer Python Smoke Test ===\\n")
    
    # Initialize tokenizer
    tokenizer = MidiTokenizer()
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Test MIDI
    test_midi = {
        'style': 'rock_punk',
        'tempo': 140,
        'key': 'Em',
        'sections': [{'type': 'VERSE', 'start': 0, 'length': 4}],
        'tracks': {
            'KICK': [
                {'pitch': 36, 'velocity': 110, 'start': 0, 'duration': 1, 'track': 'KICK'},
                {'pitch': 36, 'velocity': 110, 'start': 4, 'duration': 1, 'track': 'KICK'}
            ],
            'SNARE': [
                {'pitch': 38, 'velocity': 100, 'start': 2, 'duration': 1, 'track': 'SNARE'}
            ]
        }
    }
    
    print("\\nOriginal MIDI:")
    print(json.dumps(test_midi, indent=2))
    
    # Encode
    start_time = time.time()
    tokens = tokenizer.encode(test_midi)
    encode_time = (time.time() - start_time) * 1000
    
    print(f"\\nEncoded to {len(tokens)} tokens in {encode_time:.2f}ms")
    print("Token IDs:", tokens[:10], "..." if len(tokens) > 10 else "")
    
    # Decode
    start_time = time.time()
    decoded = tokenizer.decode(tokens)
    decode_time = (time.time() - start_time) * 1000
    
    print(f"\\nDecoded in {decode_time:.2f}ms")
    print("Decoded MIDI:")
    print(json.dumps(decoded, indent=2))
    
    # Verify round-trip
    re_encoded = tokenizer.encode(decoded)
    round_trip_success = tokens == re_encoded
    
    print("\\nRound-trip verification:")
    print(f"Original tokens: {len(tokens)}")
    print(f"Re-encoded tokens: {len(re_encoded)}")
    print(f"Round-trip successful: {round_trip_success}")
    
    if round_trip_success:
        print("\\n🎉 Python tokenizer implementation successful!")
    else:
        print("\\n⚠️ Round-trip failed - check implementation")
    
    return {
        'vocab_size': tokenizer.get_vocab_size(),
        'token_count': len(tokens),
        'encode_time': encode_time,
        'decode_time': decode_time,
        'round_trip_success': round_trip_success
    }


if __name__ == "__main__":
    results = run_smoke_test()
    print("\\n=== Test Results ===")
    for key, value in results.items():
        print(f"{key}: {value}")