#!/usr/bin/env python3
"""
Test script for the sampler-based sound design renderer.
Creates sample MIDI files and tests the rendering pipeline.
"""

import os
import sys
import tempfile
import mido
from pathlib import Path

# Add the audio module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from audio.render import render_midi_to_stems, RenderConfig

def create_test_midi_file(filename: str, style: str = "rock_punk") -> str:
    """Create a simple test MIDI file."""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Set track name based on style
    track.append(mido.MetaMessage('track_name', name=f'Test {style}'))
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))
    
    # Add some basic notes for different instruments
    instruments = [
        (36, 'kick'),    # Kick drum
        (38, 'snare'),   # Snare drum
        (28, 'bass'),    # Bass note
        (64, 'guitar'),  # Guitar chord
        (60, 'piano'),   # Piano note
        (72, 'lead')     # Lead note
    ]
    
    current_time = 0
    for i, (note, name) in enumerate(instruments):
        # Note on
        track.append(mido.Message('note_on', channel=i, note=note, velocity=80, time=current_time))
        # Note off (quarter note duration)
        track.append(mido.Message('note_off', channel=i, note=note, velocity=0, time=480))
        current_time = 240  # Eighth note apart
    
    mid.save(filename)
    return filename

def test_render_pipeline():
    """Test the complete rendering pipeline."""
    print("Testing Sampler-Based Sound Design Renderer")
    print("=" * 50)
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test each style
        styles = ['rock_punk', 'rnb_ballad', 'country_pop']
        
        for style in styles:
            print(f"\nTesting style: {style}")
            print("-" * 30)
            
            # Create test MIDI file
            midi_file = os.path.join(temp_dir, f"test_{style}.mid")
            create_test_midi_file(midi_file, style)
            print(f"Created test MIDI: {midi_file}")
            
            # Create render configuration
            config = RenderConfig(
                sample_rate=48000,
                bit_depth=24,
                normalize_stems=True,
                normalize_target_lufs=-18.0,
                render_length_seconds=10.0
            )
            
            # Test output directory
            output_dir = os.path.join(temp_dir, "test_stems")
            
            try:
                # Render stems
                print("Rendering stems...")
                result = render_midi_to_stems(
                    midi_path=midi_file,
                    style=style,
                    output_dir=output_dir,
                    song_id=f"test_{style}",
                    config=config
                )
                
                print(f"✓ Rendering successful for {style}")
                print(f"  Generated {len(result)} stems:")
                for role, path in result.items():
                    if os.path.exists(path):
                        size = os.path.getsize(path) / 1024  # KB
                        print(f"    {role}: {size:.1f} KB")
                    else:
                        print(f"    {role}: FILE NOT FOUND - {path}")
                        
            except Exception as e:
                print(f"✗ Rendering failed for {style}: {e}")
                continue
    
    print(f"\nTest completed!")

def test_instrument_registry():
    """Test the instrument registry system."""
    print("\nTesting Instrument Registry")
    print("=" * 50)
    
    from audio.render import InstrumentRegistry
    
    try:
        registry = InstrumentRegistry("configs/instruments")
        
        styles = registry.list_styles()
        print(f"Available styles: {styles}")
        
        for style in styles:
            instruments = registry.list_instruments(style)
            print(f"\n{style} instruments:")
            for inst_role in instruments:
                config = registry.get_instrument(style, inst_role)
                if config:
                    print(f"  {inst_role}: {config.name} ({config.sample_path})")
                    
    except Exception as e:
        print(f"✗ Registry test failed: {e}")

def test_midi_parsing():
    """Test MIDI file parsing."""
    print("\nTesting MIDI Parsing")
    print("=" * 50)
    
    from audio.render import MIDIRenderer, InstrumentRegistry, RenderConfig
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test MIDI
        midi_file = os.path.join(temp_dir, "test_parse.mid")
        create_test_midi_file(midi_file)
        
        try:
            registry = InstrumentRegistry("configs/instruments")
            config = RenderConfig()
            renderer = MIDIRenderer(registry, config)
            
            # Parse MIDI file
            tracks = renderer.parse_midi_file(midi_file)
            print(f"Parsed {len(tracks)} tracks:")
            
            for track_name, events in tracks.items():
                print(f"  {track_name}: {len(events)} events")
                if events:
                    first_event = events[0]
                    print(f"    First event: Note {first_event['note']}, "
                          f"Vel {first_event['velocity']}, "
                          f"Time {first_event['start_time']:.2f}s")
                    
        except Exception as e:
            print(f"✗ MIDI parsing failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the sampler-based sound design renderer")
    parser.add_argument("--test", choices=['all', 'render', 'registry', 'midi'], 
                       default='all', help="Which test to run")
    
    args = parser.parse_args()
    
    if args.test in ['all', 'registry']:
        test_instrument_registry()
    
    if args.test in ['all', 'midi']:
        test_midi_parsing()
    
    if args.test in ['all', 'render']:
        test_render_pipeline()