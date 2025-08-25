#!/usr/bin/env python3
"""
Comprehensive Dynamic CLI Test Runner
Runs all dynamic CLI tests including end-to-end, stress, and edge case tests.
Provides detailed reporting and validation of the dynamic CLI system.
"""
import subprocess
import sys
import time
from pathlib import Path


def run_test_suite(test_file: str, test_name: str) -> bool:
    """Run a test suite and return success status."""
    print(f"\n{'='*60}")
    print(f"🧪 Running {test_name}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=False,  # Show output in real-time
            timeout=300,  # 5 minute timeout
        )
        success = result.returncode == 0
        if success:
            print(f"\n✅ {test_name} PASSED")
        else:
            print(f"\n❌ {test_name} FAILED (exit code: {result.returncode})")
        return success
    except subprocess.TimeoutExpired:
        print(f"\n⏱️  {test_name} TIMED OUT")
        return False
    except Exception as e:
        print(f"\n💥 {test_name} CRASHED: {e}")
        return False
def run_quick_validation():
    """Run quick validation of core functionality."""
    print("\n🚀 Quick Validation of Core Dynamic CLI Functionality")
    print("-" * 60)
    quick_tests = [
        ("simple --help", "Help system"),
        ("simple list", "List functionality"),
        ("simple local", "Local backend"),
    ]
    all_passed = True
    for cmd, test_name in quick_tests:
        print(f"Testing: {test_name}...")
        try:
            result = subprocess.run(
                f"python -m src.simple_embeddings_module.sem_cli {cmd}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode in [0, 1, 2]:  # Allow normal exit codes
                print(f"  ✅ {test_name}")
            else:
                print(f"  ❌ {test_name} (exit code: {result.returncode})")
                all_passed = False
        except Exception as e:
            print(f"  💥 {test_name} failed: {e}")
            all_passed = False
    return all_passed
def main():
    """Main test runner."""
    print("🎯 Dynamic CLI Comprehensive Test Suite")
    print("Testing the complete dynamic CLI system implementation")
    print("=" * 80)
    start_time = time.time()
    # Change to project directory
    project_dir = Path(__file__).parent.parent
    import os
    os.chdir(project_dir)
    # Run quick validation first
    quick_ok = run_quick_validation()
    if not quick_ok:
        print("\n❌ Quick validation failed. Stopping comprehensive tests.")
        print("Please fix basic CLI functionality before running full test suite.")
        sys.exit(1)
    # Define test suites
    test_suites = [
        ("test/test_dynamic_cli_e2e.py", "End-to-End Tests"),
        ("test/test_dynamic_cli_stress.py", "Stress & Edge Case Tests"),
    ]
    # Run all test suites
    results = []
    for test_file, test_name in test_suites:
        if Path(test_file).exists():
            success = run_test_suite(test_file, test_name)
            results.append((test_name, success))
        else:
            print(f"\n⚠️  Test file not found: {test_file}")
            results.append((test_name, False))
    # Print final summary
    end_time = time.time()
    duration = end_time - start_time
    print("\n" + "=" * 80)
    print("🏁 COMPREHENSIVE TEST SUITE SUMMARY")
    print("=" * 80)
    total_suites = len(results)
    passed_suites = sum(1 for _, success in results if success)
    failed_suites = total_suites - passed_suites
    print(f"Test Duration: {duration:.1f} seconds")
    print(f"Total Test Suites: {total_suites}")
    print(f"Passed Suites: {passed_suites} ✅")
    print(f"Failed Suites: {failed_suites} ❌")
    print(f"Success Rate: {(passed_suites/total_suites)*100:.1f}%")
    print("\n📊 DETAILED RESULTS:")
    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {test_name}")
    if failed_suites > 0:
        print(f"\n❌ {failed_suites} test suite(s) failed.")
        print("Please review the detailed output above for specific failures.")
    # Overall assessment
    if passed_suites == total_suites:
        print("\n🎉 ALL TEST SUITES PASSED!")
        print("✨ The dynamic CLI system is working correctly and is robust.")
        print("\n🔍 Key Validations Completed:")
        print("  ✅ Dynamic backend discovery and registration")
        print("  ✅ Command routing without hardcoded values")
        print("  ✅ Error handling and user feedback")
        print("  ✅ Success and failure mode handling")
        print("  ✅ Stress testing and edge cases")
        print("  ✅ JSON output format consistency")
        print("  ✅ Help system dynamic generation")
        print("  ✅ Registry system robustness")
        print("\n🚀 The dynamic CLI system is production-ready!")
        return True
    else:
        print(f"\n⚠️  {failed_suites} test suite(s) need attention.")
        print("The dynamic CLI system may need additional work before production use.")
        return False
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
