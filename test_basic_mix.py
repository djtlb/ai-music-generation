#!/usr/bin/env python3
"""
Basic test script to verify the auto-mixing system functionality.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from mix import AutoMixChain, load_style_targets
from mix.utils import create_white_noise_stems

def test_basic_functionality():
    """Test basic auto-mixing functionality."""
    print("Testing basic auto-mixing functionality...")
    
    try:
        # Load style targets
        print("Loading style targets...")
        targets = load_style_targets()
        print(f"Loaded targets for styles: {list(targets.keys())}")
        
        # Create test stems
        print("Creating test stems...")
        stems = create_white_noise_stems(4, duration=2.0)
        print(f"Created {len(stems)} stems, each with shape {stems[0].shape}")
        
        # Create auto-mix chain
        print("Creating auto-mix chain...")
        auto_mix = AutoMixChain(n_stems=4, style_targets=targets)
        auto_mix.eval()
        print("Auto-mix chain created successfully")
        
        # Test with each style
        for style in ['rock_punk', 'rnb_ballad', 'country_pop']:
            print(f"\nTesting style: {style}")
            
            try:
                with torch.no_grad():
                    mixed_audio, analysis = auto_mix(stems, style)
                
                print(f"  Mixed audio shape: {mixed_audio.shape}")
                print(f"  LUFS: {analysis['lufs']:.1f}")
                print(f"  Spectral centroid: {analysis['spectral_centroid']:.0f} Hz")
                print(f"  Stereo M/S ratio: {analysis['stereo_ms_ratio']:.2f}")
                
                if 'lufs_target' in analysis:
                    print(f"  LUFS error: {analysis['lufs_error']:.1f} dB")
                if 'centroid_target' in analysis:
                    print(f"  Centroid error: {analysis['centroid_error']:.0f} Hz")
                    
            except Exception as e:
                print(f"  ERROR processing {style}: {e}")
                return False
                
        print("\nAll tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_stem_counts():
    """Test with different numbers of stems."""
    print("\nTesting different stem counts...")
    
    targets = load_style_targets()
    
    for n_stems in [2, 4, 6, 8]:
        print(f"Testing with {n_stems} stems...")
        try:
            stems = create_white_noise_stems(n_stems, duration=1.0)
            auto_mix = AutoMixChain(n_stems=n_stems, style_targets=targets)
            auto_mix.eval()
            
            with torch.no_grad():
                mixed_audio, analysis = auto_mix(stems, 'rock_punk')
                
            print(f"  SUCCESS: Output shape {mixed_audio.shape}")
            
        except Exception as e:
            print(f"  ERROR with {n_stems} stems: {e}")
            return False
            
    return True

if __name__ == '__main__':
    print("="*60)
    print("AUTO-MIXING SYSTEM BASIC TEST")
    print("="*60)
    
    success = True
    
    # Run basic functionality test
    success &= test_basic_functionality()
    
    # Test different stem counts
    success &= test_different_stem_counts()
    
    print("\n" + "="*60)
    if success:
        print("ALL TESTS PASSED! ✓")
        print("The auto-mixing system is working correctly.")
    else:
        print("SOME TESTS FAILED! ✗")
        print("Please check the error messages above.")
    print("="*60)
    
    sys.exit(0 if success else 1)