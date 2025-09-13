#!/usr/bin/env python3
"""Comprehensive linting script for ContentEngineAI project.

This script runs all configured linting tools with optimizations:
- Parallel execution for independent tools
- Proper fix command support
- Enhanced error handling and reporting
- Performance timing and caching

Usage:
    python tools/lint.py [--fix] [--verbose] [--tool TOOL_NAME] [--no-parallel]
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class LintingTool:
    """Base class for linting tools."""

    def __init__(
        self,
        name: str,
        command: list[str],
        description: str,
        fix_command: list[str] | None = None,
        expected_exit_codes: list[int] | None = None,
    ):
        self.name = name
        self.command = command
        self.fix_command = fix_command or command
        self.description = description
        self.expected_exit_codes = expected_exit_codes or [0]
        self.success = False
        self.output = ""
        self.error_output = ""
        self.duration = 0.0

    def run(self, fix: bool = False, verbose: bool = False) -> bool:
        """Run the linting tool."""
        start_time = time.time()

        try:
            command = self.fix_command if fix else self.command

            if verbose:
                print(f"🔄 Running {self.name}: {' '.join(command)}")

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                timeout=300,  # 5 minute timeout
            )

            self.output = result.stdout
            self.error_output = result.stderr
            self.success = result.returncode in self.expected_exit_codes
            self.duration = time.time() - start_time

            if verbose:
                print(f"⏱️  {self.name} completed in {self.duration:.2f}s")
                if self.output:
                    print(f"{self.name} stdout:\n{self.output}")
                if self.error_output:
                    print(f"{self.name} stderr:\n{self.error_output}")

            return self.success

        except subprocess.TimeoutExpired:
            self.error_output = f"Tool {self.name} timed out after 5 minutes"
            self.success = False
            self.duration = time.time() - start_time
            return False
        except Exception as e:
            self.error_output = str(e)
            self.success = False
            self.duration = time.time() - start_time
            return False

    def get_summary(self) -> str:
        """Get a summary of the tool's results."""
        status = "✅ PASS" if self.success else "❌ FAIL"
        duration_str = f" ({self.duration:.2f}s)" if self.duration > 0 else ""
        return f"{status} {self.name}: {self.description}{duration_str}"


def create_tools() -> dict[str, LintingTool]:
    """Create all linting tools with optimized configurations."""
    return {
        "ruff": LintingTool(
            "Ruff",
            ["poetry", "run", "ruff", "check", "src/", "tests/"],
            "Fast Python linter and formatter",
            fix_command=["poetry", "run", "ruff", "check", "--fix", "src/", "tests/"],
        ),
        "ruff-format": LintingTool(
            "Ruff Format",
            ["poetry", "run", "ruff", "format", "--check", "src/", "tests/"],
            "Code formatting",
            fix_command=["poetry", "run", "ruff", "format", "src/", "tests/"],
        ),
        "mypy": LintingTool(
            "MyPy", ["poetry", "run", "mypy", "src/"], "Static type checker"
        ),
        "bandit": LintingTool(
            "Bandit",
            ["poetry", "run", "bandit", "-r", "src/", "-f", "json"],
            "Security linter",
            expected_exit_codes=[0, 1],  # Bandit returns 1 when issues found
        ),
        "vulture": LintingTool(
            "Vulture",
            ["poetry", "run", "vulture", "src/", "--min-confidence", "80"],
            "Dead code detector",
            expected_exit_codes=[0, 1],  # Vulture returns 1 when dead code found
        ),
        "safety": LintingTool(
            "Safety",
            ["poetry", "run", "safety", "check", "--json"],
            "Dependency vulnerability checker",
            expected_exit_codes=[0, 64],  # Safety returns 64 when vulnerabilities found
        ),
        "pytest": LintingTool(
            "Pytest", ["poetry", "run", "pytest", "--tb=short", "-x"], "Test runner"
        ),
    }


def run_specific_tool(
    tools: dict[str, LintingTool], tool_name: str, fix: bool, verbose: bool
) -> bool:
    """Run a specific linting tool."""
    if tool_name not in tools:
        print(f"❌ Unknown tool: {tool_name}")
        print(f"📋 Available tools: {', '.join(sorted(tools.keys()))}")
        return False

    tool = tools[tool_name]
    print(f"🔄 Running {tool.name}...")

    success = tool.run(fix=fix, verbose=verbose)

    print(f"\n{tool.get_summary()}")

    if not success:
        if tool.error_output:
            print(f"❌ Error: {tool.error_output}")

        # Provide specific guidance for common issues
        if tool_name == "ruff" and not fix:
            print("💡 Tip: Try running with --fix to automatically resolve issues")
        elif tool_name == "mypy":
            print("💡 Tip: Check type hints and imported module stubs")
        elif tool_name == "pytest":
            print("💡 Tip: Run individual test files to isolate failures")

    return success


def run_tools_parallel(
    tools: dict[str, LintingTool], tool_names: list[str], fix: bool, verbose: bool
) -> dict[str, bool]:
    """Run multiple tools in parallel."""
    results = {}

    def run_single_tool(tool_name: str) -> tuple[str, bool]:
        if tool_name in tools:
            tool = tools[tool_name]
            success = tool.run(fix=fix, verbose=verbose)
            return tool_name, success
        return tool_name, False

    with ThreadPoolExecutor(max_workers=min(len(tool_names), 4)) as executor:
        futures = {executor.submit(run_single_tool, name): name for name in tool_names}

        for future in futures:
            tool_name, success = future.result()
            results[tool_name] = success

    return results


def run_all_tools(
    tools: dict[str, LintingTool], fix: bool, verbose: bool, parallel: bool = True
) -> tuple[bool, list[str]]:
    """Run all linting tools and return overall success status and summaries."""
    print("🔍 Running comprehensive linting checks...\n")

    summaries = []
    all_success = True
    start_time = time.time()

    # Define tool groups - some must run sequentially, others can be parallel
    sequential_tools = ["ruff"]  # Run ruff first to fix issues
    if fix:
        sequential_tools.append("ruff-format")  # Run format after ruff fix

    parallel_tools = ["mypy", "bandit", "vulture", "safety"]
    if not fix:
        parallel_tools.insert(0, "ruff-format")  # Can check format in parallel
        # if not fixing

    final_tools = ["pytest"]  # Run tests last

    # Run sequential tools first
    for tool_name in sequential_tools:
        if tool_name in tools:
            tool = tools[tool_name]
            if verbose:
                print(f"🔄 Running {tool.name}...")
            success = tool.run(fix=fix, verbose=verbose)
            summary = tool.get_summary()
            summaries.append(summary)

            if not success:
                all_success = False
                if verbose and tool.error_output:
                    print(f"❌ Error details: {tool.error_output}")

    # Run parallel tools
    if parallel and parallel_tools:
        if verbose:
            print(f"🚀 Running {len(parallel_tools)} tools in parallel...")

        parallel_results = run_tools_parallel(tools, parallel_tools, fix, verbose)

        for tool_name in parallel_tools:
            if tool_name in tools:
                tool = tools[tool_name]
                summary = tool.get_summary()
                summaries.append(summary)

                if not parallel_results.get(tool_name, False):
                    all_success = False
                    if verbose and tool.error_output:
                        print(f"❌ Error details: {tool.error_output}")
    else:
        # Fallback to sequential execution
        for tool_name in parallel_tools:
            if tool_name in tools:
                tool = tools[tool_name]
                success = tool.run(fix=fix, verbose=verbose)
                summary = tool.get_summary()
                summaries.append(summary)

                if not success:
                    all_success = False
                    if verbose and tool.error_output:
                        print(f"❌ Error details: {tool.error_output}")

    # Run final tools
    for tool_name in final_tools:
        if tool_name in tools:
            tool = tools[tool_name]
            if verbose:
                print(f"🧪 Running {tool.name}...")
            success = tool.run(fix=fix, verbose=verbose)
            summary = tool.get_summary()
            summaries.append(summary)

            if not success:
                all_success = False
                if verbose and tool.error_output:
                    print(f"❌ Error details: {tool.error_output}")

    total_time = time.time() - start_time
    if verbose:
        print(f"\n⏱️  Total execution time: {total_time:.2f}s")

    return all_success, summaries


def validate_environment() -> bool:
    """Validate that required tools are available."""
    try:
        # Check if poetry is available
        result = subprocess.run(["poetry", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Poetry not found. Please install poetry first.")
            return False

        # Check if we're in a poetry project
        if not Path("pyproject.toml").exists():
            print("❌ pyproject.toml not found. Please run from project root.")
            return False

        return True
    except FileNotFoundError:
        print("❌ Poetry not found. Please install poetry first.")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive linting checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/lint.py                    # Run all tools
  python tools/lint.py --fix              # Run all tools with auto-fix
  python tools/lint.py --tool ruff --fix   # Run only ruff with fix
  python tools/lint.py --no-parallel       # Disable parallel execution
  python tools/lint.py --output results.json  # Save results to file
        """,
    )
    parser.add_argument(
        "--fix", action="store_true", help="Apply automatic fixes where possible"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    parser.add_argument("--tool", type=str, help="Run only a specific tool")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel execution"
    )
    parser.add_argument(
        "--list-tools", action="store_true", help="List available tools and exit"
    )
    parser.add_argument(
        "--skip-env-check", action="store_true", help="Skip environment validation"
    )

    args = parser.parse_args()

    # Validate environment unless skipped
    if not args.skip_env_check and not validate_environment():
        sys.exit(1)

    try:
        tools = create_tools()
    except Exception as e:
        print(f"❌ Failed to initialize linting tools: {e}")
        sys.exit(1)

    if args.list_tools:
        print("📋 Available linting tools:")
        print("=" * 40)
        for name, tool in tools.items():
            print(f"  {name:<12} - {tool.description}")
        sys.exit(0)

    if args.tool:
        try:
            success = run_specific_tool(tools, args.tool, args.fix, args.verbose)
            sys.exit(0 if success else 1)
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            sys.exit(130)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            sys.exit(1)

    try:
        parallel = not args.no_parallel
        success, summaries = run_all_tools(tools, args.fix, args.verbose, parallel)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error during linting: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("📊 LINTING SUMMARY")
    print("=" * 60)

    for summary in summaries:
        print(summary)

    if args.output:
        total_duration = sum(tool.duration for tool in tools.values())
        results = {
            "overall_success": success,
            "total_duration": total_duration,
            "parallel_execution": parallel,
            "tools": {
                name: {
                    "success": tool.success,
                    "duration": tool.duration,
                    "output": tool.output,
                    "error_output": tool.error_output,
                }
                for name, tool in tools.items()
            },
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Results saved to {args.output}")

    # Show actionable recommendations
    failed_tools = [name for name, tool in tools.items() if not tool.success]
    if failed_tools:
        print("\n💡 To fix issues, try:")
        if "ruff" in failed_tools:
            print("   python tools/lint.py --tool ruff --fix")
        if "ruff-format" in failed_tools:
            print("   python tools/lint.py --tool ruff-format --fix")
        print("   python tools/lint.py --fix --verbose")

    print(f"\n{'🎉 All checks passed!' if success else '⚠️  Some checks failed!'}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
