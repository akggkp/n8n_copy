#!/usr/bin/env python
"""
Master Test Runner
Runs all standalone tests in sequence
"""
import subprocess
import sys
import os


def run_test(test_file, test_name):
    """Run a single test file"""
    print("\n" + "=" * 70)
    print(f"Running: {test_name}")
    print("=" * 70)

    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run {test_name}: {e}")
        return False


if __name__ == "__main__":
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "STANDALONE PIPELINE TESTS" + " " * 28 + "║")
    print("║" + " " * 15 + "Running ALL Tests" + " " * 33 + "║")
    print("╚" + "=" * 68 + "╝")

    # Create tests directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)

    tests = [
        ('tests/test_video_processor.py', 'Video Processor Service'),
        ('tests/test_ml_service.py', 'ML Service'),
        ('tests/test_database_models.py', 'Database Models'),
        ('tests/test_orchestrator_tasks.py', 'Orchestrator Tasks'),
    ]

    results = []

    for test_file, test_name in tests:
        if os.path.exists(test_file):
            passed = run_test(test_file, test_name)
            results.append((test_name, passed))
        else:
            print(f"\n⚠️  Test file not found: {test_file}")
            results.append((test_name, False))

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")

    if total_passed == len(results):
        print("\n" + "🎉" * 20)
        print("ALL TESTS PASSED!")
        print("✅ System is ready for Docker deployment")
        print("🎉" * 20)
        sys.exit(0)
    else:
        print("\n" + "⚠️ " * 15)
        print("SOME TESTS FAILED")
        print("Fix issues before building Docker images")
        print("⚠️ " * 15)
        sys.exit(1)
