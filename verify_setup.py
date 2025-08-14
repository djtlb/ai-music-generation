#!/usr/bin/env python3
"""
Quick verification script to check that all modules import correctly.
"""

def test_imports():
    """Test that all key modules can be imported."""
    try:
        print("Testing mix module imports...")
        from mix import AutoMixChain, load_style_targets
        from mix.utils import create_white_noise_stems, compute_lufs
        print("✓ Mix module imports successful")
        
        print("Testing style targets loading...")
        targets = load_style_targets()
        print(f"✓ Loaded {len(targets)} style targets")
        
        print("Testing white noise generation...")
        stems = create_white_noise_stems(2, duration=1.0)
        print(f"✓ Generated {len(stems)} test stems")
        
        print("Testing auto-mix creation...")
        auto_mix = AutoMixChain(n_stems=2, style_targets=targets)
        print("✓ AutoMixChain created successfully")
        
        print("\nAll imports and basic functionality verified!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("="*50)
    print("VERIFYING AUTO-MIX SETUP")
    print("="*50)
    
    success = test_imports()
    
    if success:
        print("\n✓ Setup verification successful!")
        print("The auto-mixing system is ready to use.")
    else:
        print("\n✗ Setup verification failed!")
        print("Please check the error messages above.")
    
    print("="*50)