#!/usr/bin/env python3
"""
Test script for the end-to-end AI Music Composition Pipeline

This script tests the pipeline with all three supported styles and validates outputs.
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path
import json

def run_pipeline_test(style: str, duration_bars: int = 16, verbose: bool = False):
    """Test the pipeline with a specific style"""
    print(f"\n🎵 Testing {style} style...")
    
    # Build command
    cmd = [
        sys.executable, "run_pipeline.py",
        "--style", style,
        "--duration_bars", str(duration_bars),
        "--config", f"configs/{style}.yaml"
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    # Run pipeline
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✅ {style} pipeline completed in {elapsed:.1f}s")
        
        # Parse output to find the export directory
        output_lines = result.stdout.split('\n')
        export_dir = None
        for line in output_lines:
            if "Output:" in line:
                export_dir = line.split("Output:")[-1].strip()
                break
        
        if export_dir and Path(export_dir).exists():
            # Validate outputs
            export_path = Path(export_dir)
            final_wav = export_path / "final.wav"
            report_json = export_path / "report.json"
            
            if final_wav.exists() and report_json.exists():
                # Check report
                try:
                    with open(report_json) as f:
                        report = json.load(f)
                    
                    if report.get("pipeline", {}).get("success"):
                        print(f"   ✅ All outputs generated successfully")
                        print(f"   📁 Output directory: {export_dir}")
                        print(f"   🎧 Final audio: {final_wav}")
                        
                        # Show some analysis data
                        audio_analysis = report.get("audio_analysis", {})
                        if audio_analysis:
                            duration = audio_analysis.get("duration_seconds", 0)
                            lufs = audio_analysis.get("estimated_lufs", 0)
                            print(f"   📊 Duration: {duration:.1f}s, LUFS: {lufs:.1f}")
                        
                        return True
                    else:
                        print(f"   ❌ Pipeline reported failure in report.json")
                        return False
                        
                except Exception as e:
                    print(f"   ⚠️  Could not parse report.json: {e}")
                    return False
            else:
                print(f"   ❌ Missing output files: final.wav={final_wav.exists()}, report.json={report_json.exists()}")
                return False
        else:
            print(f"   ❌ Export directory not found: {export_dir}")
            return False
    else:
        print(f"❌ {style} pipeline failed in {elapsed:.1f}s")
        print(f"   Error: {result.stderr}")
        return False

def main():
    """Run pipeline tests for all styles"""
    print("🧪 Testing AI Music Composition Pipeline")
    print("=" * 50)
    
    styles = ["rock_punk", "rnb_ballad", "country_pop"]
    results = {}
    
    # Quick tests with short duration
    test_duration = 8  # bars
    
    for style in styles:
        try:
            results[style] = run_pipeline_test(style, duration_bars=test_duration)
        except Exception as e:
            print(f"❌ {style} test crashed: {e}")
            results[style] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    passed = sum(results.values())
    total = len(results)
    
    for style, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {style:12} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All pipeline tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())