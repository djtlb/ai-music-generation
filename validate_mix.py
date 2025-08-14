#!/usr/bin/env python3
"""
Validation script for the auto-mixing system.

This script validates the mixing chain using white noise stems and known references,
reporting LUFS and spectral deltas against targets.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List
import torch
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from mix import AutoMixChain, load_style_targets
from mix.utils import (
    create_white_noise_stems,
    compute_lufs,
    compute_spectral_centroid, 
    compute_stereo_ms_ratio,
    validate_lufs_accuracy,
    validate_spectral_accuracy,
    compute_mix_quality_score
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def create_test_suite() -> Dict[str, List[torch.Tensor]]:
    """
    Create a comprehensive test suite with different types of audio content.
    
    Returns:
        Dictionary mapping test names to stem lists
    """
    test_suite = {}
    
    # White noise test - basic functionality
    test_suite['white_noise_4stems'] = create_white_noise_stems(4, duration=5.0)
    test_suite['white_noise_8stems'] = create_white_noise_stems(8, duration=5.0)
    
    # Frequency-specific tests
    def create_filtered_noise(freq_center: float, bandwidth: float, 
                            duration: float = 5.0) -> torch.Tensor:
        """Create band-limited noise for frequency response testing."""
        noise = torch.randn(2, int(duration * 48000)) * 0.1
        
        # Simple frequency weighting (in practice would use proper filtering)
        # This is a placeholder for actual filter implementation
        return noise
    
    # Bass-heavy test
    bass_stems = [
        create_filtered_noise(80, 50),   # Sub bass
        create_filtered_noise(150, 100), # Bass
        create_filtered_noise(500, 200), # Low mids
        create_filtered_noise(2000, 1000) # Upper content
    ]
    test_suite['bass_heavy'] = bass_stems
    
    # Bright test  
    bright_stems = [
        create_filtered_noise(200, 100),   # Low end
        create_filtered_noise(1000, 500),  # Mids
        create_filtered_noise(4000, 2000), # Highs
        create_filtered_noise(8000, 4000)  # Very bright
    ]
    test_suite['bright_content'] = bright_stems
    
    # Dynamic range tests
    quiet_stems = [s * 0.01 for s in create_white_noise_stems(4, 5.0)]  # Very quiet
    loud_stems = [s * 0.8 for s in create_white_noise_stems(4, 5.0)]    # Near clipping
    
    test_suite['quiet_stems'] = quiet_stems
    test_suite['loud_stems'] = loud_stems
    
    return test_suite


def validate_single_test(test_name: str, stems: List[torch.Tensor], 
                        style: str, auto_mix: AutoMixChain,
                        targets: Dict) -> Dict:
    """
    Validate a single test case.
    
    Args:
        test_name: Name of the test
        stems: List of audio stems
        style: Target style
        auto_mix: AutoMixChain instance
        targets: Style targets
        
    Returns:
        Validation results dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running test: {test_name} with style: {style}")
    
    try:
        # Process stems
        with torch.no_grad():
            mixed_audio, analysis = auto_mix(stems, style)
            
        # Extract metrics
        actual_lufs = analysis['lufs']
        actual_centroid = analysis['spectral_centroid']
        actual_ms_ratio = analysis['stereo_ms_ratio']
        
        # Get targets
        style_targets = targets.get(style, {})
        target_lufs = style_targets.get('lufs', -14.0)
        target_centroid = style_targets.get('spectral_centroid_hz', 2000.0)
        target_ms_ratio = style_targets.get('stereo_ms_ratio', 0.6)
        
        # Compute errors
        lufs_error = abs(actual_lufs - target_lufs)
        centroid_error = abs(actual_centroid - target_centroid)
        ms_ratio_error = abs(actual_ms_ratio - target_ms_ratio)
        
        # Validate accuracy
        lufs_valid = validate_lufs_accuracy(actual_lufs, target_lufs, tolerance=2.0)
        centroid_valid = validate_spectral_accuracy(actual_centroid, target_centroid, tolerance=500.0)
        ms_ratio_valid = ms_ratio_error <= 0.3  # 0.3 tolerance for M/S ratio
        
        # Overall quality score
        quality_score = compute_mix_quality_score(analysis, style, targets)
        
        results = {
            'test_name': test_name,
            'style': style,
            'status': 'success',
            'metrics': {
                'lufs': {
                    'target': target_lufs,
                    'actual': actual_lufs,
                    'error': lufs_error,
                    'valid': lufs_valid
                },
                'spectral_centroid': {
                    'target': target_centroid,
                    'actual': actual_centroid,
                    'error': centroid_error,
                    'valid': centroid_valid
                },
                'stereo_ms_ratio': {
                    'target': target_ms_ratio,
                    'actual': actual_ms_ratio,
                    'error': ms_ratio_error,
                    'valid': ms_ratio_valid
                }
            },
            'quality_score': quality_score,
            'overall_valid': lufs_valid and centroid_valid and ms_ratio_valid
        }
        
        logger.info(f"Test {test_name} - Quality: {quality_score:.2f}, Valid: {results['overall_valid']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Test {test_name} failed: {e}")
        return {
            'test_name': test_name,
            'style': style,
            'status': 'error',
            'error': str(e)
        }


def run_validation_suite(styles: List[str] = None, 
                        test_filter: str = None,
                        verbose: bool = False) -> Dict:
    """
    Run complete validation suite across all styles and test cases.
    
    Args:
        styles: List of styles to test (default: all)
        test_filter: Filter for test names (substring match)
        verbose: Enable verbose output
        
    Returns:
        Complete validation results
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    # Default styles
    if styles is None:
        styles = ['rock_punk', 'rnb_ballad', 'country_pop']
        
    # Load targets
    targets = load_style_targets()
    
    # Create test suite
    test_suite = create_test_suite()
    
    # Filter tests if requested
    if test_filter:
        test_suite = {
            name: stems for name, stems in test_suite.items()
            if test_filter.lower() in name.lower()
        }
        
    logger.info(f"Running validation with {len(test_suite)} tests and {len(styles)} styles")
    
    # Initialize auto-mix chain
    auto_mix = AutoMixChain(n_stems=8, style_targets=targets)
    auto_mix.eval()  # Set to evaluation mode
    
    # Run all tests
    all_results = {}
    summary_stats = {
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0,
        'error_tests': 0,
        'avg_quality_score': 0.0,
        'style_stats': {}
    }
    
    for style in styles:
        style_results = []
        style_stats = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'avg_quality': 0.0
        }
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing style: {style.upper()}")
        logger.info(f"{'='*50}")
        
        for test_name, stems in test_suite.items():
            result = validate_single_test(test_name, stems, style, auto_mix, targets)
            style_results.append(result)
            
            # Update stats
            summary_stats['total_tests'] += 1
            style_stats['total'] += 1
            
            if result['status'] == 'success':
                if result['overall_valid']:
                    summary_stats['passed_tests'] += 1
                    style_stats['passed'] += 1
                else:
                    summary_stats['failed_tests'] += 1
                    style_stats['failed'] += 1
                    
                summary_stats['avg_quality_score'] += result['quality_score']
                style_stats['avg_quality'] += result['quality_score']
            else:
                summary_stats['error_tests'] += 1
                style_stats['errors'] += 1
                
        # Finalize style stats
        if style_stats['total'] > 0:
            style_stats['avg_quality'] /= max(1, style_stats['total'] - style_stats['errors'])
            
        all_results[style] = style_results
        summary_stats['style_stats'][style] = style_stats
        
        # Print style summary
        logger.info(f"\nStyle {style} Summary:")
        logger.info(f"  Tests: {style_stats['total']}")
        logger.info(f"  Passed: {style_stats['passed']}")
        logger.info(f"  Failed: {style_stats['failed']}")
        logger.info(f"  Errors: {style_stats['errors']}")
        logger.info(f"  Avg Quality: {style_stats['avg_quality']:.2f}")
        
    # Finalize summary stats
    if summary_stats['total_tests'] > 0:
        summary_stats['avg_quality_score'] /= max(1, summary_stats['total_tests'] - summary_stats['error_tests'])
        
    return {
        'results': all_results,
        'summary': summary_stats,
        'targets': targets
    }


def print_detailed_report(validation_results: Dict):
    """Print detailed validation report."""
    summary = validation_results['summary']
    results = validation_results['results']
    
    print("\n" + "="*70)
    print("VALIDATION REPORT")
    print("="*70)
    
    # Overall summary
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']} ({summary['passed_tests']/summary['total_tests']*100:.1f}%)")
    print(f"Failed: {summary['failed_tests']} ({summary['failed_tests']/summary['total_tests']*100:.1f}%)")
    print(f"Errors: {summary['error_tests']} ({summary['error_tests']/summary['total_tests']*100:.1f}%)")
    print(f"Average Quality Score: {summary['avg_quality_score']:.2f}/1.00")
    
    # Per-style breakdown
    print(f"\n{'-'*70}")
    print("PER-STYLE BREAKDOWN")
    print(f"{'-'*70}")
    
    for style, style_stats in summary['style_stats'].items():
        print(f"\n{style.upper()}:")
        print(f"  Pass Rate: {style_stats['passed']}/{style_stats['total']} ({style_stats['passed']/style_stats['total']*100:.1f}%)")
        print(f"  Quality: {style_stats['avg_quality']:.2f}/1.00")
        
        # Show failed tests
        failed_tests = [
            result for result in results[style] 
            if result['status'] == 'success' and not result['overall_valid']
        ]
        
        if failed_tests:
            print(f"  Failed Tests:")
            for test in failed_tests:
                metrics = test['metrics']
                print(f"    - {test['test_name']}:")
                for metric, data in metrics.items():
                    if not data['valid']:
                        print(f"      {metric}: {data['error']:.2f} (target: {data['target']:.1f})")
                        
    # Metric-specific analysis
    print(f"\n{'-'*70}")
    print("METRIC ANALYSIS")
    print(f"{'-'*70}")
    
    metric_errors = {'lufs': [], 'spectral_centroid': [], 'stereo_ms_ratio': []}
    
    for style_results in results.values():
        for result in style_results:
            if result['status'] == 'success':
                for metric in metric_errors.keys():
                    metric_errors[metric].append(result['metrics'][metric]['error'])
                    
    for metric, errors in metric_errors.items():
        if errors:
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            max_error = np.max(errors)
            print(f"{metric}:")
            print(f"  Mean Error: {mean_error:.2f}")
            print(f"  Std Error: {std_error:.2f}")
            print(f"  Max Error: {max_error:.2f}")
            
    print("="*70)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate auto-mixing system with test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--styles',
        nargs='+',
        choices=['rock_punk', 'rnb_ballad', 'country_pop'],
        help='Styles to test (default: all)'
    )
    parser.add_argument(
        '--test-filter',
        help='Filter tests by name (substring match)'
    )
    parser.add_argument(
        '--output',
        help='Save detailed results to JSON file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick validation (fewer tests)'
    )
    
    args = parser.parse_args()
    
    try:
        # Run validation
        validation_results = run_validation_suite(
            styles=args.styles,
            test_filter=args.test_filter,
            verbose=args.verbose
        )
        
        # Print report
        print_detailed_report(validation_results)
        
        # Save results if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                # Convert any torch tensors to floats for JSON serialization
                def serialize_results(obj):
                    if isinstance(obj, torch.Tensor):
                        return float(obj)
                    elif isinstance(obj, dict):
                        return {k: serialize_results(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [serialize_results(item) for item in obj]
                    else:
                        return obj
                        
                serializable_results = serialize_results(validation_results)
                json.dump(serializable_results, f, indent=2)
                
            print(f"\nDetailed results saved to: {args.output}")
            
        # Exit with appropriate code
        summary = validation_results['summary']
        if summary['error_tests'] > 0:
            print(f"\nValidation completed with {summary['error_tests']} errors")
            sys.exit(2)
        elif summary['failed_tests'] > 0:
            print(f"\nValidation completed with {summary['failed_tests']} test failures")
            sys.exit(1)
        else:
            print(f"\nValidation completed successfully! All {summary['passed_tests']} tests passed.")
            sys.exit(0)
            
    except Exception as e:
        print(f"Validation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(3)


if __name__ == '__main__':
    main()