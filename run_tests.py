#!/usr/bin/env python3
"""
Run tests for the arrangement transformer
"""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run all arrangement tests"""
    test_file = Path(__file__).parent / "tests" / "test_arrangement.py"
    
    print("Running arrangement transformer tests...")
    
    try:
        # Run pytest on the test file
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(test_file), 
            "-v", "--tb=short"
        ], capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)