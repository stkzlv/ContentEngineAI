#!/usr/bin/env python3
"""Test runner script for ContentEngineAI.

This script provides convenient ways to run different types of tests
and generate test reports.
"""

import argparse
import subprocess
import sys


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run ContentEngineAI tests")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "all", "coverage", "lint", "type-check"],
        default="all",
        help="Type of tests to run (default: all)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--parallel",
        "-n",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1)",
    )
    parser.add_argument(
        "--markers",
        help="Run only tests with specific markers (e.g., 'unit and not slow')",
    )
    parser.add_argument(
        "--output",
        choices=["text", "html", "xml"],
        default="text",
        help="Test output format (default: text)",
    )

    args = parser.parse_args()

    # Base pytest command
    base_cmd = ["poetry", "run", "pytest"]

    if args.verbose:
        base_cmd.append("-v")

    if args.parallel > 1:
        base_cmd.extend(["-n", str(args.parallel)])

    if args.markers:
        base_cmd.extend(["-m", args.markers])

    # Add output format options
    if args.output == "html":
        base_cmd.extend(["--cov-report=html:htmlcov"])
    elif args.output == "xml":
        base_cmd.extend(["--cov-report=xml"])

    success = True

    if args.type == "unit":
        success = run_command(base_cmd + ["tests/", "-m", "unit"], "Unit Tests")

    elif args.type == "integration":
        success = run_command(
            base_cmd + ["tests/", "-m", "integration"], "Integration Tests"
        )

    elif args.type == "coverage":
        success = run_command(
            base_cmd
            + [
                "tests/",
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
            ],
            "Tests with Coverage Report",
        )

    elif args.type == "lint":
        success = run_command(["poetry", "run", "ruff", "check", "src"], "Code Linting")

        if success:
            success = run_command(
                ["poetry", "run", "ruff", "format", "--check", "src"],
                "Code Formatting Check",
            )

    elif args.type == "type-check":
        success = run_command(["poetry", "run", "mypy", "src"], "Type Checking")

    elif args.type == "all":
        # Run linting first
        print("Running code quality checks...")
        lint_success = run_command(
            ["poetry", "run", "ruff", "check", "src"], "Code Linting"
        )

        if lint_success:
            format_success = run_command(
                ["poetry", "run", "ruff", "format", "--check", "src"],
                "Code Formatting Check",
            )
            lint_success = lint_success and format_success

        # Run type checking
        type_success = run_command(["poetry", "run", "mypy", "src"], "Type Checking")

        # Run tests with coverage
        test_success = run_command(
            base_cmd + ["tests/", "--cov=src", "--cov-report=term-missing"],
            "All Tests with Coverage",
        )

        success = lint_success and type_success and test_success

    if success:
        print(f"\n🎉 All {args.type} checks passed!")
        return 0
    else:
        print(f"\n💥 Some {args.type} checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
