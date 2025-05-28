#!/usr/bin/env python3
"""
Pre-Experiment Validation Script
================================

This script validates that your environment is properly configured
before running the automated experimentation suite.
"""

import importlib
import json
import os
import subprocess
from pathlib import Path


def check_ray_cluster():
    """Check Ray cluster status and resources."""
    print("🔍 Checking Ray cluster...")

    try:
        result = subprocess.run(['ray', 'status'], capture_output=True, text=True, check=True)

        if "cluster_resources" in result.stdout.lower():
            print("✅ Ray cluster is running")

            # Extract resource information
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CPU' in line or 'GPU' in line or 'memory' in line:
                    print(f"   {line}")

            return True
        else:
            print("❌ Ray cluster status unclear")
            return False

    except subprocess.CalledProcessError:
        print("❌ Ray cluster not accessible")
        print("   Run: ray start --head --port=6379")
        return False
    except FileNotFoundError:
        print("❌ Ray not installed")
        print("   Run: pip install ray[default]")
        return False

def check_python_dependencies():
    """Check required Python packages."""
    print("\n🔍 Checking Python dependencies...")

    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'torch', 'datasets',
        'ray', 'openpyxl', 'Pillow', 'pydantic'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print(f"   Run: pip install {' '.join(missing_packages)}")
        return False

    return True

def check_murmura_scripts():
    """Check that Murmura example scripts exist."""
    print("\n🔍 Checking Murmura example scripts...")

    required_scripts = [
        "murmura/examples/mnist_example.py",
        "murmura/examples/decentralized_mnist_example.py",
        "murmura/examples/dp_mnist_example.py",
        "murmura/examples/dp_decentralized_mnist_example.py",
        "murmura/examples/skin_lesion_example.py",
        "murmura/examples/decentralized_skin_lesion_example.py",
        "murmura/examples/dp_skin_lesion_example.py",
        "murmura/examples/dp_decentralized_skin_lesion_example.py"
    ]

    missing_scripts = []

    for script in required_scripts:
        if Path(script).exists():
            print(f"   ✅ {script}")
        else:
            print(f"   ❌ {script}")
            missing_scripts.append(script)

    if missing_scripts:
        print(f"\n❌ Missing scripts: {len(missing_scripts)}")
        return False

    return True

def test_single_experiment():
    """Test a single minimal experiment."""
    print("\n🔍 Testing single experiment...")

    test_cmd = [
        "python", "murmura/examples/mnist_example.py",
        "--num_actors", "2",
        "--rounds", "1",
        "--epochs", "1",
        "--batch_size", "32",
        "--topology", "star",
        "--partition_strategy", "iid",
        "--log_level", "ERROR"
    ]

    if os.environ.get("RAY_ADDRESS"):
        test_cmd.extend(["--ray_address", os.environ["RAY_ADDRESS"]])

    try:
        print("   Running minimal MNIST test...")
        result = subprocess.run(
            test_cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            check=True
        )

        if "Final Test Accuracy:" in result.stdout or "Final accuracy:" in result.stdout:
            print("   ✅ Test experiment completed successfully")
            return True
        else:
            print("   ❌ Test experiment completed but no accuracy found")
            print("   Output:", result.stdout[-200:])  # Last 200 chars
            return False

    except subprocess.TimeoutExpired:
        print("   ❌ Test experiment timed out")
        return False
    except subprocess.CalledProcessError as e:
        print("   ❌ Test experiment failed")
        print("   Error:", e.stderr[-200:] if e.stderr else "No error output")
        return False

def estimate_resources():
    """Estimate resource requirements."""
    print("\n📊 Resource estimation...")

    # Experiment counts (reduced from original plan)
    experiments = {
        "Priority 1 (Core)": 162,  # 3×3×3×3×3×2 reduced
        "Priority 2 (Privacy)": 12,  # 3×2×2
        "Priority 3 (Scale)": 24,   # 3×4×2
        "Total": 198
    }

    # Time estimates (minutes per experiment)
    time_estimates = {
        "Fast (5 actors)": 5,
        "Medium (10 actors)": 10,
        "Slow (15+ actors)": 15
    }

    print("📈 Experiment counts:")
    for name, count in experiments.items():
        print(f"   {name}: {count} experiments")

    print("\n⏱️ Time estimates (with 2 parallel):")
    for config, time_per_exp in time_estimates.items():
        total_hours = (experiments["Total"] * time_per_exp) / (60 * 2)  # 2 parallel
        print(f"   {config}: ~{total_hours:.1f} hours ({total_hours/24:.1f} days)")

    print("\n💾 Storage estimates:")
    print("   Results CSV: ~5-10 MB")
    print("   Log files: ~50-100 MB")
    print("   Total output: ~100 MB")

def check_environment_variables():
    """Check important environment variables."""
    print("\n🔍 Checking environment variables...")

    env_vars = {
        "RAY_ADDRESS": "Ray cluster address",
        "CUDA_VISIBLE_DEVICES": "GPU allocation (optional)",
        "PYTHONPATH": "Python path (optional)"
    }

    for var, description in env_vars.items():
        value = os.environ.get(var)
        if value:
            print(f"   ✅ {var}={value}")
        else:
            print(f"   ⚠️  {var} not set ({description})")

def generate_test_plan():
    """Generate a small test plan."""
    print("\n📋 Generating test experiment plan...")

    test_experiments = [
        {
            "dataset": "mnist",
            "paradigm": "federated",
            "topology": "star",
            "heterogeneity": "iid",
            "privacy": "none",
            "scale": 5
        },
        {
            "dataset": "mnist",
            "paradigm": "decentralized",
            "topology": "ring",
            "heterogeneity": "moderate_noniid",
            "privacy": "none",
            "scale": 5
        }
    ]

    test_file = Path("test_experiment_plan.json")
    with open(test_file, 'w') as f:
        json.dump(test_experiments, f, indent=2)

    print(f"   ✅ Test plan saved to {test_file}")
    print(f"   Run: python automated_experiment_runner.py --dry_run")

def main():
    """Main validation routine."""
    print("🚀 Murmura Experiment Validation")
    print("=" * 40)

    checks = [
        ("Ray Cluster", check_ray_cluster),
        ("Python Dependencies", check_python_dependencies),
        ("Murmura Scripts", check_murmura_scripts),
        ("Environment Variables", check_environment_variables),
    ]

    results = []

    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ {name} check failed with error: {e}")
            results.append((name, False))

    # Optional single experiment test
    print("\n" + "=" * 40)
    run_test = input("Run single experiment test? (y/N): ").lower().strip()

    if run_test == 'y':
        test_result = test_single_experiment()
        results.append(("Single Experiment Test", test_result))

    # Resource estimation and test plan
    estimate_resources()
    generate_test_plan()

    # Summary
    print("\n" + "=" * 40)
    print("📋 Validation Summary:")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {name}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 All checks passed! Ready to run experiments.")
        print("   Next steps:")
        print("   1. python automated_experiment_runner.py --dry_run")
        print("   2. python automated_experiment_runner.py --priority 1")
    else:
        print("\n⚠️  Some checks failed. Please fix issues before running experiments.")

        # Provide specific guidance
        if not any(name == "Ray Cluster" and result for name, result in results):
            print("\n🔧 To fix Ray cluster:")
            print("   ray start --head --port=6379")
            print("   export RAY_ADDRESS='ray://127.0.0.1:10001'")

        if not any(name == "Python Dependencies" and result for name, result in results):
            print("\n🔧 To fix dependencies:")
            print("   pip install pandas numpy matplotlib torch datasets ray[default] openpyxl Pillow pydantic")

if __name__ == "__main__":
    main()
