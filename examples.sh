#!/bin/bash
# Example usage scripts for the auto-mixing system

echo "Auto-Mixing & Mastering CLI Examples"
echo "====================================="

echo ""
echo "1. Basic Auto-Mix with Test Stems:"
echo "python mix_master.py --test-stems 4 --style rock_punk --output test_rock.wav --validate"

echo ""
echo "2. Mix Real Audio Stems:"
echo "python mix_master.py --stems kick.wav snare.wav bass.wav guitar.wav \\"
echo "                     --style rnb_ballad --output mixed_rnb.wav \\"
echo "                     --analysis-output analysis.json"

echo ""
echo "3. Custom Style Targets:"
echo "python mix_master.py --stems *.wav --style country_pop \\"
echo "                     --targets custom_targets.yaml --output final.wav"

echo ""
echo "4. Validation Suite:"
echo "python validate_mix.py --verbose --output validation_report.json"

echo ""
echo "5. Quick Validation:"
echo "python validate_mix.py --quick --styles rock_punk rnb_ballad"

echo ""
echo "6. Test with White Noise:"
echo "python validate_mix.py --test-filter white_noise --styles rock_punk"

echo ""
echo "Available Styles:"
echo "- rock_punk    (-9.5 LUFS, 2800Hz centroid, 0.6 M/S ratio)"
echo "- rnb_ballad   (-12.0 LUFS, 1800Hz centroid, 0.8 M/S ratio)"
echo "- country_pop  (-10.5 LUFS, 2200Hz centroid, 0.7 M/S ratio)"

echo ""
echo "Performance Testing:"
echo "python test_basic_mix.py"

echo ""
echo "Full Test Suite:"
echo "python test_auto_mix.py"

echo ""
echo "Setup Verification:"
echo "python verify_setup.py"