#!/usr/bin/env python3
"""
Create sample tokenized MIDI data for testing FAISS indexing.
Generates realistic tokenized patterns in refs_midi directories.
"""

import json
import random
from pathlib import Path


def create_sample_tokenized_data():
    """Create sample tokenized MIDI data for testing."""
    
    # Style configurations
    styles = {
        "pop": {
            "tempo_range": (100, 130),
            "common_chords": ["C", "F", "G", "Am", "Dm"],
            "sections": ["INTRO", "VERSE", "CHORUS", "BRIDGE", "OUTRO"],
            "instruments": ["PIANO", "BASS_PICK", "KICK", "SNARE"]
        },
        "rock": {
            "tempo_range": (120, 160),
            "common_chords": ["E", "A", "B", "C#m", "F#m"],
            "sections": ["INTRO", "VERSE", "CHORUS", "BRIDGE", "SOLO", "OUTRO"],
            "instruments": ["ACOUSTIC_STRUM", "BASS_PICK", "KICK", "SNARE", "LEAD"]
        },
        "rnb_soul": {
            "tempo_range": (80, 110),
            "common_chords": ["Cmaj7", "Dm7", "G7", "Am7", "Fmaj7"],
            "sections": ["INTRO", "VERSE", "CHORUS", "BRIDGE", "OUTRO"],
            "instruments": ["PIANO", "BASS_PICK", "KICK", "SNARE"]
        }
    }
    
    # Child style variations
    child_styles = {
        "pop": {
            "dance_pop": {
                "tempo_modifier": 10,  # +10 BPM
                "extra_instruments": ["SYNTH_LEAD"],
                "velocity_boost": 10
            },
            "pop_rock": {
                "tempo_modifier": 5,
                "extra_instruments": ["ACOUSTIC_STRUM"],
                "velocity_boost": 5
            }
        },
        "rock": {
            "punk": {
                "tempo_modifier": 20,
                "extra_instruments": ["DISTORTION"],
                "velocity_boost": 15
            }
        }
    }
    
    def generate_bar_tokens(style_name, child_style=None, bar_idx=0):
        """Generate tokens for a single bar."""
        style = styles[style_name]
        
        tokens = []
        
        # Style and tempo
        if child_style:
            tokens.append(f"STYLE={child_style}")
            base_tempo = random.randint(*style["tempo_range"])
            modifier = child_styles[style_name][child_style]["tempo_modifier"]
            tempo = base_tempo + modifier
        else:
            tokens.append(f"STYLE={style_name}")
            tempo = random.randint(*style["tempo_range"])
        
        tokens.append(f"TEMPO={tempo}")
        tokens.append(f"KEY={random.choice(['C', 'G', 'D', 'A', 'E'])}")
        
        # Section
        section = random.choice(style["sections"])
        tokens.append(f"SECTION={section}")
        
        # Bar marker
        tokens.append("BAR")
        tokens.append(f"POS=1")
        
        # Chord
        chord = random.choice(style["common_chords"])
        tokens.append(f"CHORD={chord}")
        
        # Generate notes for instruments
        for inst in style["instruments"]:
            # Add instrument notes
            for pos in [1, 2, 3, 4]:  # Quarter note positions
                tokens.append(f"POS={pos}")
                tokens.append(f"INST={inst}")
                
                # Note events
                if random.random() > 0.3:  # 70% chance of note
                    note = random.randint(36, 84)  # MIDI note range
                    velocity = random.randint(60, 100)
                    
                    # Apply child style velocity boost
                    if child_style and "velocity_boost" in child_styles[style_name][child_style]:
                        velocity += child_styles[style_name][child_style]["velocity_boost"]
                        velocity = min(127, velocity)
                    
                    tokens.append("NOTE_ON")
                    tokens.append(str(note))
                    tokens.append(f"VEL={velocity}")
                    
                    # Duration
                    duration = random.choice(["DUR=4", "DUR=8", "DUR=16"])  # Quarter, eighth, sixteenth
                    tokens.append(duration)
        
        return tokens
    
    def create_style_pack_data(style_name):
        """Create tokenized data for a parent style."""
        print(f"Creating data for {style_name}...")
        
        style_dir = Path("style_packs") / style_name
        refs_midi_dir = style_dir / "refs_midi"
        refs_midi_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate multiple tokenized files
        for file_idx in range(3):
            # Token file format
            tokens_file = refs_midi_dir / f"{style_name}_sample_{file_idx}.tokens"
            all_tokens = []
            
            # Generate 4 bars
            for bar_idx in range(4):
                bar_tokens = generate_bar_tokens(style_name, bar_idx=bar_idx)
                all_tokens.extend(bar_tokens)
            
            with open(tokens_file, 'w') as f:
                f.write(' '.join(all_tokens))
            
            # JSON file format
            json_file = refs_midi_dir / f"{style_name}_sample_{file_idx}.json"
            tokenized_bars = []
            
            for bar_idx in range(4):
                bar_tokens = generate_bar_tokens(style_name, bar_idx=bar_idx)
                tokenized_bars.append(bar_tokens)
            
            json_data = {
                "style": style_name,
                "bars": len(tokenized_bars),
                "tokenized_bars": tokenized_bars
            }
            
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)
        
        print(f"Created {style_name} parent data")
    
    def create_child_style_data(parent_style, child_name):
        """Create tokenized data for a child style."""
        print(f"Creating child data for {parent_style}/{child_name}...")
        
        child_dir = Path("style_packs") / parent_style / child_name
        refs_midi_dir = child_dir / "refs_midi"
        refs_midi_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate child-specific patterns
        for file_idx in range(2):  # Fewer files for child styles
            # Token file
            tokens_file = refs_midi_dir / f"{child_name}_sample_{file_idx}.tokens"
            all_tokens = []
            
            # Generate 4 bars with child style characteristics
            for bar_idx in range(4):
                bar_tokens = generate_bar_tokens(parent_style, child_style=child_name, bar_idx=bar_idx)
                all_tokens.extend(bar_tokens)
            
            with open(tokens_file, 'w') as f:
                f.write(' '.join(all_tokens))
            
            # JSON file
            json_file = refs_midi_dir / f"{child_name}_sample_{file_idx}.json"
            tokenized_bars = []
            
            for bar_idx in range(4):
                bar_tokens = generate_bar_tokens(parent_style, child_style=child_name, bar_idx=bar_idx)
                tokenized_bars.append(bar_tokens)
            
            json_data = {
                "style": child_name,
                "parent_style": parent_style,
                "bars": len(tokenized_bars),
                "tokenized_bars": tokenized_bars
            }
            
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)
        
        # Create meta.json for child
        meta_file = child_dir / "meta.json"
        meta_data = {
            "name": child_name,
            "parent": parent_style,
            "description": f"{child_name.replace('_', ' ').title()} style variation",
            "characteristics": child_styles[parent_style][child_name]
        }
        
        with open(meta_file, 'w') as f:
            json.dump(meta_data, f, indent=2)
        
        print(f"Created {parent_style}/{child_name} child data")
    
    # Create parent style data
    for style_name in styles.keys():
        create_style_pack_data(style_name)
    
    # Create child style data
    for parent_style, children in child_styles.items():
        for child_name in children.keys():
            create_child_style_data(parent_style, child_name)
    
    print("\nSample tokenized data creation complete!")
    print("Generated data for:")
    print("Parent styles:", list(styles.keys()))
    print("Child styles:", {p: list(c.keys()) for p, c in child_styles.items()})


if __name__ == "__main__":
    create_sample_tokenized_data()