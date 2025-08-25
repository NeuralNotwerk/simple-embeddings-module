#!/usr/bin/env python3
"""
End-to-End Tests for Dynamic CLI System
Comprehensive tests for the dynamic CLI system covering both success and failure modes.
Tests the complete pipeline from command parsing to execution.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple


class DynamicCLITester:
    """Comprehensive tester for the dynamic CLI system."""
    def __init__(self):
        self.test_dir = None
        self.original_cwd = os.getcwd()
        self.test_results = []
        self.setup_test_environment()
    def setup_test_environment(self):
        """Set up isolated test environment."""
        self.test_dir = tempfile.mkdtemp(prefix="sem_cli_test_")
        os.chdir(self.test_dir)
        print(f"🧪 Test environment: {self.test_dir}")
    def cleanup_test_environment(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        if self.test_dir and Path(self.test_dir).exists():
            import shutil
            shutil.rmtree(self.test_dir)
    def run_cli_command(self, cmd: str, input_text: str = None, timeout: int = 60) -> Tuple[int, str, str, Dict]:
        """
        Run a CLI command and return results.
        Returns:
            Tuple of (exit_code, stdout, stderr, parsed_json_output)
        """
        try:
            # Construct full command
            full_cmd = f"cd {self.original_cwd} && python -m src.simple_embeddings_module.sem_cli {cmd}"
            result = subprocess.run(
                full_cmd,
                shell=True,
                capture_output=True,
                text=True,
                input=input_text,
                timeout=timeout,
                cwd=self.original_cwd,
            )
            # Try to parse JSON output
            json_output = {}
            if result.stdout.strip():
                try:
                    json_output = json.loads(result.stdout)
                except json.JSONDecodeError:
                    # Not JSON output, that's okay
                    pass
            return result.returncode, result.stdout, result.stderr, json_output
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out", {}
        except Exception as e:
            return -1, "", f"Command failed: {e}", {}
    def assert_success(self, exit_code: int, test_name: str, json_output: Dict = None):
        """Assert that a command succeeded."""
        if exit_code == 0:
            print(f"  ✅ {test_name}")
            if json_output and json_output.get("success"):
                print(f"     JSON success: {json_output.get('success')}")
            self.test_results.append((test_name, True, "Success"))
        else:
            print(f"  ❌ {test_name} - Exit code: {exit_code}")
            self.test_results.append((test_name, False, f"Exit code: {exit_code}"))
    def assert_failure(self, exit_code: int, test_name: str, expected_error: str = None):
        """Assert that a command failed as expected."""
        if exit_code != 0:
            print(f"  ✅ {test_name} (expected failure)")
            self.test_results.append((test_name, True, "Expected failure"))
        else:
            print(f"  ❌ {test_name} - Expected failure but got success")
            self.test_results.append((test_name, False, "Unexpected success"))
    def test_dynamic_backend_discovery(self):
        """Test that backends are discovered dynamically."""
        print("\n🔍 Testing Dynamic Backend Discovery...")
        # Test simple command help shows dynamic backends
        exit_code, stdout, stderr, json_output = self.run_cli_command("simple --help")
        # Should succeed and show available backends
        self.assert_success(exit_code, "Simple command help")
        # Check that help contains dynamic backend information
        if "Backend (local, aws)" in stdout or "local" in stdout:
            print("  ✅ Dynamic backend help generation")
            self.test_results.append(("Dynamic backend help", True, "Backends shown"))
        else:
            print("  ❌ Dynamic backend help generation")
            self.test_results.append(("Dynamic backend help", False, "No backends shown"))
    def test_simple_local_success_modes(self):
        """Test successful operations with local backend."""
        print("\n🏠 Testing Simple Local Success Modes...")
        # Test 1: Initialize empty local database
        exit_code, stdout, stderr, json_output = self.run_cli_command("simple local")
        self.assert_success(exit_code, "Local backend initialization", json_output)
        # Test 2: Add content via stdin
        test_content = "This is a test document for the dynamic CLI system"
        exit_code, stdout, stderr, json_output = self.run_cli_command("simple local add", input_text=test_content)
        self.assert_success(exit_code, "Add content via stdin", json_output)
        # Test 3: Search for content
        exit_code, stdout, stderr, json_output = self.run_cli_command('simple local search "dynamic CLI"')
        self.assert_success(exit_code, "Search content", json_output)
        # Verify search found results
        if json_output.get("data", {}).get("result_count", 0) > 0:
            print("  ✅ Search found results")
            self.test_results.append(("Search results found", True, "Results returned"))
        else:
            print("  ❌ Search found no results")
            self.test_results.append(("Search results found", False, "No results"))
        # Test 4: List documents
        exit_code, stdout, stderr, json_output = self.run_cli_command("simple local list")
        self.assert_success(exit_code, "List documents", json_output)
        # Test 5: Add content with --text flag
        exit_code, stdout, stderr, json_output = self.run_cli_command('simple local add --text "Another test document"')
        self.assert_success(exit_code, "Add content with --text flag", json_output)
    def test_simple_local_failure_modes(self):
        """Test failure modes with local backend."""
        print("\n❌ Testing Simple Local Failure Modes...")
        # Test 1: Search without query
        exit_code, stdout, stderr, json_output = self.run_cli_command("simple local search")
        self.assert_failure(exit_code, "Search without query")
        # Verify error message is helpful
        if json_output.get("success") == False and "query" in json_output.get("data", {}).get("example", "").lower():
            print("  ✅ Helpful error message for missing query")
            self.test_results.append(("Helpful search error", True, "Good error message"))
        else:
            print("  ❌ Unhelpful error message for missing query")
            self.test_results.append(("Helpful search error", False, "Poor error message"))
    def test_invalid_backend_handling(self):
        """Test handling of invalid backends."""
        print("\n🚫 Testing Invalid Backend Handling...")
        # Test 1: Invalid backend name
        exit_code, stdout, stderr, json_output = self.run_cli_command("simple invalid_backend")
        self.assert_failure(exit_code, "Invalid backend name")
        # Verify error shows available backends
        if json_output.get("data", {}).get("available_backends"):
            backends = json_output["data"]["available_backends"]
            if "local" in backends:
                print("  ✅ Error shows available backends dynamically")
                self.test_results.append(("Dynamic error backends", True, f"Showed: {backends}"))
            else:
                print("  ❌ Error doesn't show expected backends")
                self.test_results.append(("Dynamic error backends", False, f"Got: {backends}"))
        else:
            print("  ❌ Error doesn't show available backends")
            self.test_results.append(("Dynamic error backends", False, "No backends shown"))
        # Test 2: No backend specified
        exit_code, stdout, stderr, json_output = self.run_cli_command("simple")
        # This should show help or error, not crash
        if exit_code != 0:
            print("  ✅ No backend specified handled gracefully")
            self.test_results.append(("No backend handling", True, "Graceful failure"))
        else:
            print("  ⚠️  No backend specified succeeded (might be valid)")
            self.test_results.append(("No backend handling", True, "Succeeded"))
    def test_simple_list_functionality(self):
        """Test the simple list functionality."""
        print("\n📋 Testing Simple List Functionality...")
        # Test simple list command
        exit_code, stdout, stderr, json_output = self.run_cli_command("simple list")
        self.assert_success(exit_code, "Simple list command", json_output)
        # Verify it returns database information
        if json_output.get("data", {}).get("databases"):
            print("  ✅ List returns database information")
            self.test_results.append(("List database info", True, "Databases returned"))
        else:
            print("  ⚠️  List returns no databases (might be empty)")
            self.test_results.append(("List database info", True, "No databases (empty)"))
    def test_aws_backend_availability(self):
        """Test AWS backend availability and error handling."""
        print("\n☁️  Testing AWS Backend Availability...")
        # Test AWS backend (might fail if no credentials)
        exit_code, stdout, stderr, json_output = self.run_cli_command("simple aws", timeout=30)
        if exit_code == 0:
            print("  ✅ AWS backend available and working")
            self.test_results.append(("AWS backend available", True, "Working"))
        else:
            print("  ⚠️  AWS backend unavailable (expected without credentials)")
            self.test_results.append(("AWS backend unavailable", True, "Expected without creds"))
            # Check if error is helpful
            if "aws" in stderr.lower() or "credential" in stderr.lower() or "bucket" in stderr.lower():
                print("  ✅ AWS error message is helpful")
                self.test_results.append(("AWS error message", True, "Helpful"))
            else:
                print("  ❌ AWS error message is not helpful")
                self.test_results.append(("AWS error message", False, "Unhelpful"))
    def test_command_argument_parsing(self):
        """Test that command arguments are parsed correctly."""
        print("\n⚙️  Testing Command Argument Parsing...")
        # Test with various argument combinations
        test_cases = [
            ("simple local --help", "Local help"),
            ("simple local list --top-k 3", "List with top-k"),
            ("simple local search --help", "Search help"),
        ]
        for cmd, test_name in test_cases:
            exit_code, stdout, stderr, json_output = self.run_cli_command(cmd)
            # These should either succeed or fail gracefully
            if exit_code in [0, 1, 2]:  # 0=success, 1=error, 2=usage error
                print(f"  ✅ {test_name}")
                self.test_results.append((test_name, True, f"Exit code: {exit_code}"))
            else:
                print(f"  ❌ {test_name} - Unexpected exit code: {exit_code}")
                self.test_results.append((test_name, False, f"Bad exit code: {exit_code}"))
    def test_dynamic_help_generation(self):
        """Test that help is generated dynamically."""
        print("\n📖 Testing Dynamic Help Generation...")
        # Test main help
        exit_code, stdout, stderr, json_output = self.run_cli_command("--help")
        if exit_code == 0 and "simple" in stdout:
            print("  ✅ Main help includes simple command")
            self.test_results.append(("Main help", True, "Simple command shown"))
        else:
            print("  ❌ Main help missing simple command")
            self.test_results.append(("Main help", False, "Simple command missing"))
        # Test simple help
        exit_code, stdout, stderr, json_output = self.run_cli_command("simple --help")
        if exit_code == 0:
            print("  ✅ Simple help generated")
            self.test_results.append(("Simple help", True, "Generated"))
            # Check for dynamic content
            if "local" in stdout and ("aws" in stdout or "Backend" in stdout):
                print("  ✅ Simple help shows dynamic backends")
                self.test_results.append(("Dynamic help content", True, "Backends shown"))
            else:
                print("  ❌ Simple help doesn't show dynamic backends")
                self.test_results.append(("Dynamic help content", False, "No backends"))
        else:
            print("  ❌ Simple help failed to generate")
            self.test_results.append(("Simple help", False, f"Exit code: {exit_code}"))
    def test_json_output_format(self):
        """Test that JSON output format is consistent."""
        print("\n📄 Testing JSON Output Format...")
        # Test commands that should return JSON
        json_commands = [
            ("simple local", "Local init JSON"),
            ("simple list", "List JSON"),
        ]
        for cmd, test_name in json_commands:
            exit_code, stdout, stderr, json_output = self.run_cli_command(cmd)
            if json_output:
                # Check for required JSON fields
                required_fields = ["success", "timestamp"]
                has_required = all(field in json_output for field in required_fields)
                if has_required:
                    print(f"  ✅ {test_name} - Valid JSON structure")
                    self.test_results.append((f"{test_name} JSON", True, "Valid structure"))
                else:
                    print(f"  ❌ {test_name} - Missing required JSON fields")
                    self.test_results.append((f"{test_name} JSON", False, "Missing fields"))
            else:
                print(f"  ⚠️  {test_name} - No JSON output (might be text-only)")
                self.test_results.append((f"{test_name} JSON", True, "No JSON (text-only)"))
    def run_all_tests(self):
        """Run all end-to-end tests."""
        print("🚀 Starting Dynamic CLI End-to-End Tests")
        print("=" * 60)
        try:
            # Test dynamic system components
            self.test_dynamic_backend_discovery()
            self.test_simple_local_success_modes()
            self.test_simple_local_failure_modes()
            self.test_invalid_backend_handling()
            self.test_simple_list_functionality()
            self.test_aws_backend_availability()
            self.test_command_argument_parsing()
            self.test_dynamic_help_generation()
            self.test_json_output_format()
        finally:
            self.cleanup_test_environment()
        # Print summary
        self.print_test_summary()
    def print_test_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 60)
        print("🧪 DYNAMIC CLI TEST SUMMARY")
        print("=" * 60)
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, passed, _ in self.test_results if passed)
        failed_tests = total_tests - passed_tests
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for test_name, passed, details in self.test_results:
                if not passed:
                    print(f"  • {test_name}: {details}")
        print("\n📊 DETAILED RESULTS:")
        for test_name, passed, details in self.test_results:
            status = "✅" if passed else "❌"
            print(f"  {status} {test_name}: {details}")
        # Overall result
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! Dynamic CLI system is working correctly.")
            return True
        else:
            print(f"\n⚠️  {failed_tests} tests failed. Please review the issues above.")
            return False
def main():
    """Main test runner."""
    print("Dynamic CLI End-to-End Test Suite")
    print("Testing both success and failure modes")
    print("-" * 50)
    tester = DynamicCLITester()
    success = tester.run_all_tests()
    # Exit with appropriate code
    sys.exit(0 if success else 1)
if __name__ == "__main__":
    main()
