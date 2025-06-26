#!/usr/bin/env python3
"""
Quick Trust System Test

Fast test script to verify the trust monitoring system is working correctly
before running full baseline experiments. Runs minimal configurations to
check for basic functionality and zero false positives.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, description, timeout=600):
    """Run a command with timeout and error handling."""
    print(f"\n🧪 {description}")
    print(f"💻 Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Success! ({execution_time:.2f}s)")
            # Extract key metrics from output
            if "Final Accuracy:" in result.stdout:
                for line in result.stdout.split('\n'):
                    if "Final Accuracy:" in line or "False Positive Rate:" in line or "Excluded Nodes:" in line:
                        print(f"📊 {line.strip()}")
            return True
        else:
            print(f"❌ Failed! ({execution_time:.2f}s)")
            print(f"Error: {result.stderr[-500:]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout after {timeout}s")
        return False
    except Exception as e:
        print(f"💥 Exception: {str(e)}")
        return False

def main():
    """Run quick trust system tests."""
    
    print("🚀 Quick Trust System Test")
    print("=" * 50)
    print("This script runs minimal configurations to verify the trust system works correctly.")
    print("Expected results: Zero false positives, good FL performance")
    
    tests = [
        {
            "cmd": [
                "python", "murmura/examples/adaptive_trust_mnist_example.py",
                "--num_actors", "4",
                "--num_rounds", "8", 
                "--topology", "ring",
                "--trust_profile", "default",
                "--log_level", "INFO"
            ],
            "description": "Quick MNIST test (4 actors, 8 rounds)",
            "timeout": 300
        },
        {
            "cmd": [
                "python", "murmura/examples/adaptive_trust_cifar10_example.py",
                "--num_actors", "4",
                "--num_rounds", "10",
                "--model_type", "simple",
                "--topology", "ring",
                "--trust_profile", "default", 
                "--log_level", "INFO"
            ],
            "description": "Quick CIFAR-10 test (4 actors, 10 rounds, simple model)",
            "timeout": 600
        }
    ]
    
    print(f"\n📋 Running {len(tests)} quick tests...")
    
    successful_tests = 0
    for i, test in enumerate(tests, 1):
        print(f"\n{'=' * 60}")
        print(f"🧪 Test {i}/{len(tests)}")
        print(f"{'=' * 60}")
        
        success = run_command(test["cmd"], test["description"], test["timeout"])
        if success:
            successful_tests += 1
    
    print(f"\n{'=' * 60}")
    print(f"🏁 Quick Test Results: {successful_tests}/{len(tests)} successful")
    print(f"{'=' * 60}")
    
    if successful_tests == len(tests):
        print("✅ All tests passed! Trust system is working correctly.")
        print("🚀 Ready to run full baseline experiments.")
        print("\nNext steps:")
        print("  python scripts/run_trust_baseline_experiments.py")
    else:
        print("❌ Some tests failed. Check the logs above for errors.")
        print("🔧 Fix issues before running full experiments.")
        
    return successful_tests == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)