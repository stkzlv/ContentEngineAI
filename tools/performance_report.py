#!/usr/bin/env python3
"""Performance monitoring report generator.

This tool generates comprehensive reports from historical pipeline performance data.
It can create summary reports, trend analysis, and performance comparisons.

Usage:
    python tools/performance_report.py --report-type summary
    python tools/performance_report.py --report-type trends --product-id B0BTYCRJSS
    python tools/performance_report.py --report-type detailed --limit 10
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.performance import PerformanceHistoryManager, PipelineRunMetrics


class PerformanceReportGenerator:
    """Generates performance monitoring reports from historical data."""

    def __init__(self, history_manager: PerformanceHistoryManager):
        self.history_manager = history_manager

    def generate_summary_report(self, limit: int = 50) -> dict[str, Any]:
        """Generate overall summary report."""
        runs = self.history_manager.get_run_history(limit=limit)

        if not runs:
            return {"error": "No historical data available"}

        # Basic statistics
        total_runs = len(runs)
        successful_runs = sum(1 for run in runs if run.success)
        success_rate = (successful_runs / total_runs) * 100 if total_runs > 0 else 0

        # Duration statistics
        durations = [run.total_duration for run in runs]
        avg_duration = sum(durations) / len(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0

        # Memory statistics
        memory_deltas = [run.total_memory_delta for run in runs]
        peak_memories = [run.peak_memory for run in runs]
        avg_memory_delta = (
            sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0
        )
        avg_peak_memory = (
            sum(peak_memories) / len(peak_memories) if peak_memories else 0
        )

        # CPU statistics
        cpu_percentages = [run.total_cpu_percent for run in runs]
        avg_cpu = sum(cpu_percentages) / len(cpu_percentages) if cpu_percentages else 0

        # Product distribution
        product_counts: dict[str, int] = {}
        profile_counts: dict[str, int] = {}

        for run in runs:
            product_counts[run.product_id] = product_counts.get(run.product_id, 0) + 1
            profile_counts[run.profile_name] = (
                profile_counts.get(run.profile_name, 0) + 1
            )

        # Recent performance trends
        recent_runs = runs[: min(10, len(runs))]
        recent_avg_duration = (
            sum(run.total_duration for run in recent_runs) / len(recent_runs)
            if recent_runs
            else 0
        )

        # Step performance analysis
        step_stats = self._analyze_step_performance(runs)

        return {
            "report_type": "summary",
            "generated_at": datetime.now().isoformat(),
            "data_range": {
                "total_runs": total_runs,
                "oldest_run": runs[-1].start_timestamp if runs else None,
                "newest_run": runs[0].start_timestamp if runs else None,
            },
            "success_metrics": {
                "total_runs": total_runs,
                "successful_runs": successful_runs,
                "failed_runs": total_runs - successful_runs,
                "success_rate_percent": round(success_rate, 2),
            },
            "performance_metrics": {
                "duration": {
                    "average_seconds": round(avg_duration, 2),
                    "minimum_seconds": round(min_duration, 2),
                    "maximum_seconds": round(max_duration, 2),
                    "recent_average_seconds": round(recent_avg_duration, 2),
                },
                "memory": {
                    "average_delta_mb": round(avg_memory_delta, 2),
                    "average_peak_mb": round(avg_peak_memory, 2),
                },
                "cpu": {"average_percent": round(avg_cpu, 2)},
            },
            "distribution": {
                "products": dict(
                    sorted(product_counts.items(), key=lambda x: x[1], reverse=True)
                ),
                "profiles": dict(
                    sorted(profile_counts.items(), key=lambda x: x[1], reverse=True)
                ),
            },
            "step_analysis": step_stats,
        }

    def generate_trends_report(
        self, product_id: str | None = None, days: int = 30
    ) -> dict[str, Any]:
        """Generate performance trends report."""
        runs = self.history_manager.get_run_history()

        # Filter by product if specified
        if product_id:
            runs = [run for run in runs if run.product_id == product_id]

        # Filter by date range
        cutoff_date = datetime.now() - timedelta(days=days)
        runs = [
            run
            for run in runs
            if (
                datetime.fromisoformat(run.start_timestamp.replace("Z", "+00:00"))
                >= cutoff_date
            )
        ]

        if not runs:
            return {"error": "No data available for the specified criteria"}

        # Sort by timestamp for trend analysis
        runs.sort(key=lambda x: x.start_timestamp)

        # Calculate daily aggregates
        daily_stats: dict[str, dict[str, Any]] = {}
        for run in runs:
            date_key = run.start_timestamp[:10]  # YYYY-MM-DD

            if date_key not in daily_stats:
                daily_stats[date_key] = {
                    "runs": [],
                    "successful_runs": 0,
                    "total_duration": 0.0,
                    "avg_memory_delta": 0.0,
                    "avg_cpu": 0.0,
                }

            daily_stats[date_key]["runs"].append(run)
            if run.success:
                daily_stats[date_key]["successful_runs"] += 1
            daily_stats[date_key]["total_duration"] += run.total_duration
            daily_stats[date_key]["avg_memory_delta"] += run.total_memory_delta
            daily_stats[date_key]["avg_cpu"] += run.total_cpu_percent

        # Calculate averages for each day
        trend_data: list[dict[str, Any]] = []
        for date, stats in sorted(daily_stats.items()):
            runs_list = stats["runs"]
            run_count = len(runs_list)
            successful_runs_count = stats["successful_runs"]
            total_duration = stats["total_duration"]
            avg_memory_delta = stats["avg_memory_delta"]
            avg_cpu = stats["avg_cpu"]
            trend_data.append(
                {
                    "date": date,
                    "run_count": run_count,
                    "success_rate": (
                        (successful_runs_count / run_count) * 100
                        if run_count > 0
                        else 0
                    ),
                    "avg_duration": (
                        total_duration / run_count if run_count > 0 else 0
                    ),
                    "avg_memory_delta": (
                        avg_memory_delta / run_count if run_count > 0 else 0
                    ),
                    "avg_cpu": avg_cpu / run_count if run_count > 0 else 0,
                }
            )

        return {
            "report_type": "trends",
            "generated_at": datetime.now().isoformat(),
            "filters": {
                "product_id": product_id,
                "days": days,
                "total_runs": len(runs),
            },
            "trend_data": trend_data,
            "summary": {
                "date_range": (
                    f"{trend_data[0]['date']} to {trend_data[-1]['date']}"
                    if trend_data
                    else None
                ),
                "total_days": len(trend_data),
                "avg_daily_runs": (
                    sum(d["run_count"] for d in trend_data) / len(trend_data)
                    if trend_data
                    else 0
                ),
            },
        }

    def generate_detailed_report(self, limit: int = 20) -> dict[str, Any]:
        """Generate detailed report with individual run information."""
        runs = self.history_manager.get_run_history(limit=limit)

        if not runs:
            return {"error": "No historical data available"}

        detailed_runs = []
        for run in runs:
            # Analyze step performance for this run
            step_details = []
            for step_data in run.step_metrics:
                step_details.append(
                    {
                        "step_name": step_data["step_name"],
                        "duration": round(step_data["duration"], 3),
                        "memory_delta": round(
                            step_data["memory_end"] - step_data["memory_start"], 2
                        ),
                        "cpu_percent": round(step_data["cpu_percent"], 1),
                        "errors": step_data.get("errors", []),
                    }
                )

            detailed_runs.append(
                {
                    "run_id": run.run_id,
                    "product_id": run.product_id,
                    "profile_name": run.profile_name,
                    "timestamp": run.start_timestamp,
                    "success": run.success,
                    "error_message": run.error_message,
                    "metrics": {
                        "total_duration": round(run.total_duration, 2),
                        "memory_delta": round(run.total_memory_delta, 2),
                        "peak_memory": round(run.peak_memory, 2),
                        "cpu_percent": round(run.total_cpu_percent, 1),
                    },
                    "step_details": step_details,
                }
            )

        return {
            "report_type": "detailed",
            "generated_at": datetime.now().isoformat(),
            "limit": limit,
            "runs": detailed_runs,
        }

    def _analyze_step_performance(
        self, runs: list[PipelineRunMetrics]
    ) -> dict[str, Any]:
        """Analyze performance by pipeline step."""
        step_stats = {}

        for run in runs:
            for step_data in run.step_metrics:
                step_name = step_data["step_name"]

                if step_name not in step_stats:
                    step_stats[step_name] = {
                        "count": 0,
                        "total_duration": 0,
                        "total_memory_delta": 0,
                        "error_count": 0,
                    }

                step_stats[step_name]["count"] += 1
                step_stats[step_name]["total_duration"] += step_data["duration"]
                step_stats[step_name]["total_memory_delta"] += (
                    step_data["memory_end"] - step_data["memory_start"]
                )
                step_stats[step_name]["error_count"] += len(step_data.get("errors", []))

        # Calculate averages
        step_analysis = {}
        for step_name, stats in step_stats.items():
            count = stats["count"]
            step_analysis[step_name] = {
                "execution_count": count,
                "average_duration": (
                    round(stats["total_duration"] / count, 3) if count > 0 else 0
                ),
                "average_memory_delta": (
                    round(stats["total_memory_delta"] / count, 2) if count > 0 else 0
                ),
                "error_rate": (
                    round((stats["error_count"] / count) * 100, 2) if count > 0 else 0
                ),
            }

        # Sort by average duration (slowest first)
        return dict(
            sorted(
                step_analysis.items(),
                key=lambda x: x[1]["average_duration"],
                reverse=True,
            )
        )


def main():
    """Main entry point for the performance report tool."""
    parser = argparse.ArgumentParser(
        description="Generate performance monitoring reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate summary report
  python tools/performance_report.py --report-type summary

  # Generate trends for specific product
  python tools/performance_report.py --report-type trends --product-id B0BTYCRJSS

  # Generate detailed report with last 20 runs
  python tools/performance_report.py --report-type detailed --limit 20

  # Save report to file
  python tools/performance_report.py --report-type summary --output report.json
        """,
    )

    parser.add_argument(
        "--report-type",
        choices=["summary", "trends", "detailed"],
        default="summary",
        help="Type of report to generate",
    )

    parser.add_argument(
        "--product-id", help="Filter by specific product ID (for trends report)"
    )

    parser.add_argument(
        "--limit", type=int, default=50, help="Maximum number of runs to include"
    )

    parser.add_argument(
        "--days", type=int, default=30, help="Number of days for trends analysis"
    )

    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("outputs/performance_history"),
        help="Directory containing performance history data",
    )

    parser.add_argument(
        "--output", type=Path, help="Save report to JSON file instead of printing"
    )

    parser.add_argument(
        "--format", choices=["json", "pretty"], default="pretty", help="Output format"
    )

    args = parser.parse_args()

    # Initialize history manager
    history_manager = PerformanceHistoryManager(
        history_dir=args.history_dir,
        max_runs=1000,  # High limit for report generation
    )

    # Generate report
    generator = PerformanceReportGenerator(history_manager)

    if args.report_type == "summary":
        report = generator.generate_summary_report(limit=args.limit)
    elif args.report_type == "trends":
        report = generator.generate_trends_report(
            product_id=args.product_id, days=args.days
        )
    elif args.report_type == "detailed":
        report = generator.generate_detailed_report(limit=args.limit)
    else:
        parser.error(f"Unknown report type: {args.report_type}")

    # Output report
    if args.output:
        # Save to file
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")
    else:
        # Print to stdout
        if args.format == "json":
            print(json.dumps(report, indent=2))
        else:
            # Pretty print format
            print_pretty_report(report)


def print_pretty_report(report: dict[str, Any]) -> None:
    """Print report in human-readable format."""
    report_type = report.get("report_type", "unknown")

    print(f"\n{'='*60}")
    print(f"PERFORMANCE MONITORING REPORT - {report_type.upper()}")
    print(f"Generated: {report.get('generated_at', 'Unknown')}")
    print(f"{'='*60}")

    if "error" in report:
        print(f"\n❌ ERROR: {report['error']}")
        return

    if report_type == "summary":
        print_summary_report(report)
    elif report_type == "trends":
        print_trends_report(report)
    elif report_type == "detailed":
        print_detailed_report(report)


def print_summary_report(report: dict[str, Any]) -> None:
    """Print summary report in pretty format."""
    data_range = report.get("data_range", {})
    success = report.get("success_metrics", {})
    perf = report.get("performance_metrics", {})
    dist = report.get("distribution", {})
    steps = report.get("step_analysis", {})

    print("\n📊 DATA OVERVIEW")
    print(f"   Total Runs: {data_range.get('total_runs', 0)}")
    print(
        f"   Date Range: {data_range.get('oldest_run', 'N/A')} to "
        f"{data_range.get('newest_run', 'N/A')}"
    )

    print("\n✅ SUCCESS METRICS")
    print(
        f"   Success Rate: {success.get('success_rate_percent', 0)}% "
        f"({success.get('successful_runs', 0)}/{success.get('total_runs', 0)})"
    )
    print(f"   Failed Runs: {success.get('failed_runs', 0)}")

    print("\n⏱️  PERFORMANCE METRICS")
    duration = perf.get("duration", {})
    memory = perf.get("memory", {})
    cpu = perf.get("cpu", {})

    print("   Duration:")
    print(f"     Average: {duration.get('average_seconds', 0)}s")
    print(
        f"     Range: {duration.get('minimum_seconds', 0)}s - "
        f"{duration.get('maximum_seconds', 0)}s"
    )
    print(f"     Recent Avg: {duration.get('recent_average_seconds', 0)}s")

    print("   Memory:")
    print(f"     Avg Delta: {memory.get('average_delta_mb', 0)} MB")
    print(f"     Avg Peak: {memory.get('average_peak_mb', 0)} MB")

    print(f"   CPU: {cpu.get('average_percent', 0)}% average")

    print("\n📈 DISTRIBUTION")
    products = dist.get("products", {})
    profiles = dist.get("profiles", {})

    print("   Top Products:")
    for product, count in list(products.items())[:5]:
        print(f"     {product}: {count} runs")

    print("   Profiles:")
    for profile, count in profiles.items():
        print(f"     {profile}: {count} runs")

    print("\n🔧 STEP PERFORMANCE (Top 5 Slowest)")
    for i, (step_name, stats) in enumerate(list(steps.items())[:5]):
        print(f"   {i+1}. {step_name}:")
        print(f"      Avg Duration: {stats.get('average_duration', 0)}s")
        print(f"      Executions: {stats.get('execution_count', 0)}")
        print(f"      Error Rate: {stats.get('error_rate', 0)}%")


def print_trends_report(report: dict[str, Any]) -> None:
    """Print trends report in pretty format."""
    filters = report.get("filters", {})
    summary = report.get("summary", {})
    trends = report.get("trend_data", [])

    print("\n🔍 FILTERS")
    print(f"   Product ID: {filters.get('product_id', 'All')}")
    print(f"   Days: {filters.get('days', 0)}")
    print(f"   Total Runs: {filters.get('total_runs', 0)}")

    print("\n📈 TREND SUMMARY")
    print(f"   Date Range: {summary.get('date_range', 'N/A')}")
    print(f"   Total Days: {summary.get('total_days', 0)}")
    print(f"   Avg Daily Runs: {summary.get('avg_daily_runs', 0):.1f}")

    print("\n📊 DAILY TRENDS (Last 10 Days)")
    for trend in trends[-10:]:
        print(
            f"   {trend['date']}: {trend['run_count']} runs, "
            f"{trend['success_rate']:.1f}% success, "
            f"{trend['avg_duration']:.1f}s avg"
        )


def print_detailed_report(report: dict[str, Any]) -> None:
    """Print detailed report in pretty format."""
    runs = report.get("runs", [])
    limit = report.get("limit", 0)

    print(f"\n📋 DETAILED RUNS (Showing {len(runs)} of max {limit})")

    for i, run in enumerate(runs[:10]):  # Show first 10 runs
        status = "✅" if run["success"] else "❌"
        print(f"\n   {i+1}. {status} {run['run_id']}")
        print(f"      Product: {run['product_id']} | Profile: {run['profile_name']}")
        print(f"      Time: {run['timestamp']}")

        metrics = run["metrics"]
        print(
            f"      Duration: {metrics['total_duration']}s | "
            f"Memory: {metrics['memory_delta']}MB | "
            f"CPU: {metrics['cpu_percent']}%"
        )

        if not run["success"] and run["error_message"]:
            print(f"      Error: {run['error_message']}")

        # Show slowest steps
        steps = sorted(run["step_details"], key=lambda x: x["duration"], reverse=True)[
            :3
        ]
        print("      Slowest Steps:")
        for step in steps:
            print(f"        - {step['step_name']}: {step['duration']}s")

    if len(runs) > 10:
        print(f"\n   ... and {len(runs) - 10} more runs")


if __name__ == "__main__":
    main()
