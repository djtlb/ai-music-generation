#!/usr/bin/env python3
"""
Integration test for the pipeline configuration and basic functionality
Tests the configuration loading and pipeline initialization without full execution
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_config_loading():
    """Test that configuration files load correctly"""
    print("🧪 Testing configuration loading...")
    
    try:
        # Test importing the pipeline module
        sys.path.append(str(Path(__file__).parent))
        from run_pipeline import load_config, MusicCompositionPipeline
        
        # Test loading each style config
        styles = ["rock_punk", "rnb_ballad", "country_pop"]
        
        for style in styles:
            config_path = f"configs/{style}.yaml"
            if Path(config_path).exists():
                config = load_config(config_path, style)
                print(f"   ✅ {style}: loaded config successfully")
                print(f"      BPM: {config.bpm}, Key: {config.key}, Duration: {config.duration_bars} bars")
            else:
                print(f"   ❌ {style}: config file not found at {config_path}")
                return False
        
        # Test default config
        default_config = load_config(None, "rock_punk")
        print(f"   ✅ Default config loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration loading failed: {e}")
        return False

def test_pipeline_initialization():
    """Test that pipeline can be initialized without errors"""
    print("\n🧪 Testing pipeline initialization...")
    
    try:
        from run_pipeline import load_config, MusicCompositionPipeline
        
        # Test initializing pipeline for each style
        for style in ["rock_punk", "rnb_ballad", "country_pop"]:
            config = load_config(None, style)
            
            # Override to use short duration for test
            config.duration_bars = 8
            
            pipeline = MusicCompositionPipeline(config)
            print(f"   ✅ {style}: pipeline initialized successfully")
            print(f"      Output dir: {pipeline.output_dir}")
            print(f"      Style: {pipeline.style}, BPM: {pipeline.bpm}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Pipeline initialization failed: {e}")
        return False

def test_fallback_functions():
    """Test that fallback functions work when dependencies are missing"""
    print("\n🧪 Testing fallback functions...")
    
    try:
        from run_pipeline import load_config, MusicCompositionPipeline
        
        config = load_config(None, "rock_punk")
        config.duration_bars = 8
        pipeline = MusicCompositionPipeline(config)
        
        # Test fallback arrangement generation
        arrangement = pipeline._generate_fallback_arrangement()
        print(f"   ✅ Fallback arrangement: {len(arrangement['sections'])} sections")
        
        # Test fallback report generation
        report = pipeline._generate_basic_report()
        print(f"   ✅ Fallback report: {list(report.keys())}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Fallback function test failed: {e}")
        return False

def test_output_directories():
    """Test that output directory structure is created correctly"""
    print("\n🧪 Testing output directory structure...")
    
    try:
        from run_pipeline import load_config, MusicCompositionPipeline
        
        config = load_config(None, "rock_punk")
        config.duration_bars = 8
        pipeline = MusicCompositionPipeline(config)
        
        # Check that directories were created
        if pipeline.output_dir.exists():
            print(f"   ✅ Output directory created: {pipeline.output_dir}")
        else:
            print(f"   ❌ Output directory not created: {pipeline.output_dir}")
            return False
        
        if pipeline.stems_dir.exists():
            print(f"   ✅ Stems directory created: {pipeline.stems_dir}")
        else:
            print(f"   ❌ Stems directory not created: {pipeline.stems_dir}")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Output directory test failed: {e}")
        return False

def main():
    """Run integration tests"""
    print("🔧 AI Music Pipeline Integration Test")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Fallback Functions", test_fallback_functions),
        ("Output Directory Structure", test_output_directories)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name:25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("\n💡 Next steps:")
        print("   1. Install dependencies: pip install -r requirements-pipeline.txt")
        print("   2. Run demo: python demo_pipeline.py")
        print("   3. Test pipeline: python run_pipeline.py --style rock_punk --duration_bars 8")
        return 0
    else:
        print("⚠️  Some tests failed. Check configuration and dependencies.")
        return 1

if __name__ == "__main__":
    sys.exit(main())