#!/usr/bin/env python3
"""
Simple CLI script to test the planner functionality.
"""

import sys
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from planner import MusicPlanner


def main():
    """Test the planner with sample inputs."""
    print("Testing Music Planner...")
    
    # Initialize planner
    planner = MusicPlanner("configs")
    
    # Test cases
    test_cases = [
        {
            "name": "Pop Dance Track",
            "lyrics": """
            VERSE: Dancing through the night
            Feeling everything's alright
            CHORUS: We can fly, we can touch the sky
            Nothing's gonna stop us now
            """,
            "genre": "dance pop, 128 bpm, four on the floor, bright energy"
        },
        {
            "name": "Rock Anthem",
            "lyrics": """
            VERSE: Standing on the edge of time
            CHORUS: We are the champions of our destiny
            BRIDGE: When the world comes crashing down
            CHORUS: We are the champions of our destiny
            """,
            "genre": "rock anthem, 140 bpm, powerful drums, electric guitars"
        },
        {
            "name": "R&B Ballad",
            "lyrics": """
            VERSE: Love is all we need tonight
            Hold me close under the moonlight
            CHORUS: Forever and always, you and me
            BRIDGE: Through the storms we'll make it through
            """,
            "genre": "r&b ballad, 75 bpm, in D minor, lush strings, intimate"
        },
        {
            "name": "Hip Hop Track",
            "lyrics": """
            Started from the bottom now we here
            Every day grinding, vision crystal clear
            """,
            "genre": "hip hop, 95 bpm, heavy 808s, trap influenced"
        },
        {
            "name": "Minimal Input",
            "lyrics": "",
            "genre": "pop"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}: {test_case['name']}")
        print('='*50)
        
        try:
            result = planner.plan(test_case['lyrics'], test_case['genre'])
            
            print(f"Input Lyrics: {test_case['lyrics'][:100]}...")
            print(f"Input Genre: {test_case['genre']}")
            print("\nGenerated Control JSON:")
            print(json.dumps(result, indent=2))
            
            # Validate essential fields
            required_fields = ['style', 'bpm', 'key', 'arrangement', 'drum_template', 'hook_type']
            missing = [field for field in required_fields if field not in result]
            
            if missing:
                print(f"\n❌ MISSING FIELDS: {missing}")
            else:
                print(f"\n✅ All required fields present")
                print(f"   Style: {result['style']}")
                print(f"   BPM: {result['bpm']}")
                print(f"   Key: {result['key']}")
                print(f"   Structure: {result['arrangement']['structure']}")
                print(f"   Drum Template: {result['drum_template']}")
                print(f"   Hook Type: {result['hook_type']}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()