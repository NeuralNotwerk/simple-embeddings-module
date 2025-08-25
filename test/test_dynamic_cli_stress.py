#!/usr/bin/env python3
"""
Stress and Edge Case Tests for Dynamic CLI System
Tests edge cases, error conditions, and stress scenarios to ensure robustness.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple


class DynamicCLIStressTester:
    """Stress and edge case tester for the dynamic CLI system."""
    def __init__(self):
        self.test_dir = None
        self.original_cwd = os.getcwd()
        self.test_results = []
        self.setup_test_environment()
    def setup_test_environment(self):
        """Set up isolated test environment."""
        self.test_dir = tempfile.mkdtemp(prefix="sem_cli_stress_")
        os.chdir(self.test_dir)
        print(f"🧪 Stress test environment: {self.test_dir}")
    def cleanup_test_environment(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        if self.test_dir and Path(self.test_dir).exists():
            import shutil
            shutil.rmtree(self.test_dir)
    def run_cli_command(self, cmd: str, input_text: str = None, timeout: int = 30) -> Tuple[int, str, str, Dict]:
        """Run a CLI command and return results."""
        try:
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
            json_output = {}
            if result.stdout.strip():
                try:
                    json_output = json.loads(result.stdout)
                except json.JSONDecodeError:
                    pass
            return result.returncode, result.stdout, result.stderr, json_output
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out", {}
        except Exception as e:
            return -1, "", f"Command failed: {e}", {}
    def record_test(self, test_name: str, passed: bool, details: str = ""):
        """Record test result."""
        status = "✅" if passed else "❌"
        print(f"  {status} {test_name}")
        if details:
            print(f"     {details}")
        self.test_results.append((test_name, passed, details))
    def test_malformed_commands(self):
        """Test handling of malformed commands."""
        print("\n🔧 Testing Malformed Commands...")
        malformed_commands = [
            ("simple", "No backend specified"),
            ("simple ''", "Empty backend"),
            ("simple local ''", "Empty operation"),
            ("simple local search", "Search without query"),
            ("simple local add", "Add without content"),
            ("simple local invalid_operation", "Invalid operation"),
            ("simple local search --invalid-flag", "Invalid flag"),
        ]
        for cmd, test_name in malformed_commands:
            exit_code, stdout, stderr, json_output = self.run_cli_command(cmd)
            # These should fail gracefully (non-zero exit but no crash)
            if exit_code != 0 and exit_code != -1:
                self.record_test(f"Malformed: {test_name}", True, "Graceful failure")
            else:
                self.record_test(f"Malformed: {test_name}", False, f"Exit code: {exit_code}")
    def test_large_content_handling(self):
        """Test handling of large content."""
        print("\n📏 Testing Large Content Handling...")
        # Test with large text input
        large_text = "This is a test document. " * 1000  # ~25KB
        exit_code, stdout, stderr, json_output = self.run_cli_command(
            "simple local add", input_text=large_text, timeout=60
        )
        if exit_code == 0:
            self.record_test("Large content addition", True, f"Added {len(large_text)} chars")
        else:
            self.record_test("Large content addition", False, f"Failed with exit code {exit_code}")
        # Test search with large query
        large_query = "test " * 100  # Large query
        exit_code, stdout, stderr, json_output = self.run_cli_command(
            f'simple local search "{large_query}"', timeout=60
        )
        if exit_code == 0:
            self.record_test("Large query search", True, "Handled large query")
        else:
            self.record_test("Large query search", False, f"Failed with exit code {exit_code}")
    def test_special_characters(self):
        """Test handling of special characters in input."""
        print("\n🔤 Testing Special Characters...")
        special_texts = [
            ("Unicode: 你好世界 🌍", "Unicode text"),
            ("Quotes: \"Hello 'world'\"", "Mixed quotes"),
            ("Symbols: @#$%^&*()[]{}|\\", "Special symbols"),
            ("Newlines:\nLine 1\nLine 2", "Newlines"),
            ("Tabs:\tTabbed\tcontent", "Tabs"),
        ]
        for text, test_name in special_texts:
            exit_code, stdout, stderr, json_output = self.run_cli_command("simple local add", input_text=text)
            if exit_code == 0:
                self.record_test(f"Special chars: {test_name}", True, "Handled correctly")
            else:
                self.record_test(f"Special chars: {test_name}", False, f"Failed: {exit_code}")
    def test_concurrent_operations(self):
        """Test concurrent CLI operations."""
        print("\n⚡ Testing Concurrent Operations...")
        # This is a basic test - in a real scenario you'd use threading
        # For now, just test rapid sequential operations
        operations = [
            "simple local add --text 'Doc 1'",
            "simple local add --text 'Doc 2'",
            "simple local search 'Doc'",
            "simple local list",
        ]
        all_passed = True
        for i, cmd in enumerate(operations):
            exit_code, stdout, stderr, json_output = self.run_cli_command(cmd)
            if exit_code != 0:
                all_passed = False
                break
        if all_passed:
            self.record_test("Rapid sequential operations", True, "All operations succeeded")
        else:
            self.record_test("Rapid sequential operations", False, "Some operations failed")
    def test_memory_stress(self):
        """Test memory usage with multiple documents."""
        print("\n🧠 Testing Memory Stress...")
        # Add multiple documents
        num_docs = 50
        success_count = 0
        for i in range(num_docs):
            exit_code, stdout, stderr, json_output = self.run_cli_command(
                f"simple local add --text 'Document number {i} with some content to test memory usage'"
            )
            if exit_code == 0:
                success_count += 1
        if success_count == num_docs:
            self.record_test("Memory stress test", True, f"Added {num_docs} documents")
        else:
            self.record_test("Memory stress test", False, f"Only added {success_count}/{num_docs}")
        # Test search performance with many documents
        exit_code, stdout, stderr, json_output = self.run_cli_command(
            "simple local search 'Document number'", timeout=30
        )
        if exit_code == 0 and json_output.get("data", {}).get("result_count", 0) > 0:
            self.record_test("Search with many docs", True, "Search still works")
        else:
            self.record_test("Search with many docs", False, "Search failed or no results")
    def test_filesystem_edge_cases(self):
        """Test filesystem-related edge cases."""
        print("\n📁 Testing Filesystem Edge Cases...")
        # Test with read-only directory (simulate)
        # Note: This is a simplified test - real read-only testing would need special setup
        # Test with very long database name
        long_name = "a" * 100
        exit_code, stdout, stderr, json_output = self.run_cli_command(f"simple local --db {long_name}")
        # Should either work or fail gracefully
        if exit_code in [0, 1]:
            self.record_test("Long database name", True, "Handled gracefully")
        else:
            self.record_test("Long database name", False, f"Unexpected exit: {exit_code}")
        # Test with special characters in path
        special_path = "./test path with spaces"
        exit_code, stdout, stderr, json_output = self.run_cli_command(f'simple local --path "{special_path}"')
        if exit_code in [0, 1]:
            self.record_test("Path with spaces", True, "Handled gracefully")
        else:
            self.record_test("Path with spaces", False, f"Unexpected exit: {exit_code}")
    def test_timeout_scenarios(self):
        """Test timeout and interruption scenarios."""
        print("\n⏱️  Testing Timeout Scenarios...")
        # Test with very short timeout to see if command handles interruption
        exit_code, stdout, stderr, json_output = self.run_cli_command(
            "simple local add --text 'Quick test'", timeout=1  # Very short timeout
        )
        # Should either complete quickly or timeout gracefully
        if exit_code in [0, -1]:
            self.record_test("Short timeout handling", True, "Handled appropriately")
        else:
            self.record_test("Short timeout handling", False, f"Unexpected behavior: {exit_code}")
    def test_error_message_quality(self):
        """Test quality and helpfulness of error messages."""
        print("\n💬 Testing Error Message Quality...")
        error_scenarios = [
            ("simple nonexistent", "Invalid backend"),
            ("simple local search", "Missing query"),
            ("simple local --invalid-flag", "Invalid flag"),
        ]
        for cmd, scenario in error_scenarios:
            exit_code, stdout, stderr, json_output = self.run_cli_command(cmd)
            # Check if error message is helpful
            error_text = stderr + stdout
            has_helpful_info = any(
                word in error_text.lower() for word in ["usage", "help", "available", "example", "try", "error"]
            )
            if exit_code != 0 and has_helpful_info:
                self.record_test(f"Error quality: {scenario}", True, "Helpful error message")
            else:
                self.record_test(f"Error quality: {scenario}", False, "Unhelpful error message")
    def test_registry_robustness(self):
        """Test robustness of the dynamic registry system."""
        print("\n🏗️  Testing Registry Robustness...")
        # Test multiple help requests (registry should be stable)
        help_commands = [
            "simple --help",
            "simple local --help",
            "--help",
        ]
        all_stable = True
        for cmd in help_commands:
            exit_code, stdout, stderr, json_output = self.run_cli_command(cmd)
            if exit_code not in [0, 1, 2]:  # Allow normal exit codes
                all_stable = False
                break
        if all_stable:
            self.record_test("Registry stability", True, "Multiple help requests stable")
        else:
            self.record_test("Registry stability", False, "Registry became unstable")
        # Test backend discovery consistency
        exit_code1, stdout1, _, _ = self.run_cli_command("simple --help")
        exit_code2, stdout2, _, _ = self.run_cli_command("simple --help")
        if exit_code1 == exit_code2 and stdout1 == stdout2:
            self.record_test("Backend discovery consistency", True, "Consistent results")
        else:
            self.record_test("Backend discovery consistency", False, "Inconsistent results")
    def run_all_stress_tests(self):
        """Run all stress and edge case tests."""
        print("🔥 Starting Dynamic CLI Stress Tests")
        print("=" * 60)
        try:
            self.test_malformed_commands()
            self.test_large_content_handling()
            self.test_special_characters()
            self.test_concurrent_operations()
            self.test_memory_stress()
            self.test_filesystem_edge_cases()
            self.test_timeout_scenarios()
            self.test_error_message_quality()
            self.test_registry_robustness()
        finally:
            self.cleanup_test_environment()
        self.print_test_summary()
    def print_test_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 60)
        print("🔥 DYNAMIC CLI STRESS TEST SUMMARY")
        print("=" * 60)
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, passed, _ in self.test_results if passed)
        failed_tests = total_tests - passed_tests
        print(f"Total Stress Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        if failed_tests > 0:
            print("\n❌ FAILED STRESS TESTS:")
            for test_name, passed, details in self.test_results:
                if not passed:
                    print(f"  • {test_name}: {details}")
        print("\n📊 DETAILED STRESS TEST RESULTS:")
        for test_name, passed, details in self.test_results:
            status = "✅" if passed else "❌"
            print(f"  {status} {test_name}: {details}")
        if failed_tests == 0:
            print("\n🎉 ALL STRESS TESTS PASSED! Dynamic CLI system is robust.")
            return True
        else:
            print(f"\n⚠️  {failed_tests} stress tests failed. System may need hardening.")
            return False
def main():
    """Main stress test runner."""
    print("Dynamic CLI Stress Test Suite")
    print("Testing edge cases, error conditions, and stress scenarios")
    print("-" * 60)
    tester = DynamicCLIStressTester()
    success = tester.run_all_stress_tests()
    sys.exit(0 if success else 1)
if __name__ == "__main__":
    main()
