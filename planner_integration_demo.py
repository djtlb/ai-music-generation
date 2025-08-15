#!/usr/bin/env python3
"""
Integration example showing how planner.py fits into the existing pipeline.

This demonstrates the flow:
User Input (lyrics + genre) -> Planner -> Control JSON -> Pipeline Modules
"""

import sys
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from planner import MusicPlanner


def simulate_pipeline_integration():
    """Simulate how the planner integrates with the full pipeline."""
    print("🎵 Music Generation Pipeline Integration Demo")
    print("=" * 60)
    
    # Initialize planner
    planner = MusicPlanner("configs")
    
    # Simulate user input (this would come from the web UI)
    user_lyrics = """
    VERSE: Walking through the city lights tonight
    Everything feels like it's gonna be alright
    CHORUS: We're alive, we're electric
    Dancing through the night ecstatic
    VERSE: Lost in rhythm, lost in sound
    Our feet don't even touch the ground
    CHORUS: We're alive, we're electric
    Dancing through the night ecstatic
    BRIDGE: When the morning comes we'll remember this
    Every moment was electric bliss
    CHORUS: We're alive, we're electric
    Dancing through the night ecstatic
    """
    
    user_genre = "electro pop, 126 bpm, synth-heavy, club energy, wide stereo"
    
    print(f"📝 User Lyrics Input:\n{user_lyrics}")
    print(f"🎛️ User Genre Input: {user_genre}")
    print()
    
    # Step 1: Planner converts text to control JSON
    print("Step 1: 🧠 Planning - Converting text to control JSON...")
    control_json = planner.plan(user_lyrics, user_genre)
    
    print("✅ Control JSON Generated:")
    print(json.dumps(control_json, indent=2))
    print()
    
    # Step 2: Show how this feeds into other pipeline modules
    print("Step 2: 🔄 Pipeline Module Integration:")
    print("-" * 40)
    
    # Tokenizer would use this
    print("🔤 Tokenizer Module:")
    print(f"  - Style: {control_json['style']}")
    print(f"  - BPM: {control_json['bpm']}")
    print(f"  - Key: {control_json['key']}")
    print(f"  - Time Feel: {control_json['time_feel']}")
    print()
    
    # Arrangement transformer would use this
    print("🏗️ Arrangement Transformer:")
    print(f"  - Target Structure: {control_json['arrangement']['structure']}")
    print(f"  - Section Lengths: {control_json['arrangement']['section_lengths']}")
    print(f"  - Total Bars: {control_json['arrangement']['total_bars']}")
    print()
    
    # Melody/Harmony generator would use this
    print("🎼 Melody/Harmony Generator:")
    print(f"  - Key Signature: {control_json['key']}")
    print(f"  - Chord Progressions: {list(control_json.get('chord_progressions', {}).keys())}")
    print(f"  - Instruments: {control_json.get('instruments', {}).get('required', [])}")
    print()
    
    # Sound design would use this
    print("🔊 Sound Design Engine:")
    print(f"  - Drum Template: {control_json['drum_template']}")
    print(f"  - Style: {control_json['style']}")
    print(f"  - Groove: {control_json.get('groove', {})}")
    print()
    
    # Mixing/mastering would use this
    print("🎚️ Mixing/Mastering Engine:")
    mix_targets = control_json.get('mix_targets', {})
    print(f"  - Target LUFS: {mix_targets.get('lufs', 'N/A')}")
    print(f"  - Spectral Centroid: {mix_targets.get('spectral_centroid_hz', 'N/A')} Hz")
    print(f"  - Style-specific targets loaded")
    print()
    
    # Step 3: Show how lyrics sections are structured for alignment
    print("Step 3: 📝 Lyric Alignment Preparation:")
    print("-" * 40)
    lyrics_sections = control_json.get('lyrics_sections', [])
    for i, section in enumerate(lyrics_sections):
        print(f"  Section {i+1}: {section['type']}")
        print(f"    Content: {section['content'][:50]}...")
        print(f"    Line Count: {section['line_count']}")
    print()
    
    # Step 4: Show validation and QA thresholds
    print("Step 4: ✅ Quality Assurance Setup:")
    print("-" * 40)
    qa_thresholds = control_json.get('qa_thresholds', {})
    for metric, threshold in qa_thresholds.items():
        print(f"  {metric}: {threshold}")
    print()
    
    print("🎉 Pipeline Ready! Control JSON provides all parameters needed for:")
    print("   • Tokenization")
    print("   • Arrangement Generation") 
    print("   • Melody/Harmony Creation")
    print("   • Sound Design")
    print("   • Mixing/Mastering")
    print("   • Quality Assessment")


def demo_different_genres():
    """Demonstrate planner with different genres."""
    print("\n" + "="*60)
    print("🎵 Multi-Genre Demo")
    print("="*60)
    
    planner = MusicPlanner("configs")
    
    test_cases = [
        ("Pop Hit", "radio-ready pop anthem", "pop rock, 125 bpm, guitar-driven, sing-along chorus"),
        ("Trap Banger", "hard trap beat with aggressive lyrics", "trap, 140 bpm halftime, heavy 808s, dark vibe"),
        ("R&B Slow Jam", "intimate love song", "r&b ballad, 72 bpm, in F# minor, silky smooth"),
        ("Country Crossover", "small town to big city story", "country pop, 115 bpm, acoustic guitar, storytelling"),
        ("EDM Drop", "festival main stage energy", "progressive house, 128 bpm, big room, massive drops"),
    ]
    
    for title, lyrics, genre in test_cases:
        print(f"\n🎵 {title}:")
        print(f"   Lyrics concept: {lyrics}")
        print(f"   Genre: {genre}")
        
        result = planner.plan(lyrics, genre)
        
        print(f"   → Style: {result['style']}")
        print(f"   → BPM: {result['bpm']}")
        print(f"   → Key: {result['key']}")
        print(f"   → Drum Template: {result['drum_template']}")
        print(f"   → Hook Type: {result['hook_type']}")


if __name__ == "__main__":
    simulate_pipeline_integration()
    demo_different_genres()