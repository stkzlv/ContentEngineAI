#!/usr/bin/env python3
"""Performance monitoring report generator.

This tool generates comprehensive reports from historical pipeline performance data.
It can create summary reports, trend analysis, performance comparisons, and
regression detection.

Usage:
    python tools/performance_report.py --report-type summary
    python tools/performance_report.py --report-type trends --product-id B0BTYCRJSS
    python tools/performance_report.py --report-type detailed --limit 10
    python tools/performance_report.py --report-type comparison
    python tools/performance_report.py --report-type detailed --format csv --limit 5
"""

import argparse
import csv
import io
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.utils.performance import PerformanceHistoryManager, PipelineRunMetrics


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Calculate percentile from a pre-sorted list of values."""
    if not sorted_values:
        return 0.0
    idx = (pct / 100) * (len(sorted_values) - 1)
    lower = int(math.floor(idx))
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = idx - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


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

        # Duration statistics with percentiles
        durations = sorted(run.total_duration for run in runs)
        avg_duration = sum(durations) / len(durations) if durations else 0
        min_duration = durations[0] if durations else 0
        max_duration = durations[-1] if durations else 0

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
            "generated_at": datetime.now(tz=UTC).isoformat(),
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
                    "p50_seconds": round(_percentile(durations, 50), 2),
                    "p95_seconds": round(_percentile(durations, 95), 2),
                    "p99_seconds": round(_percentile(durations, 99), 2),
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
        """Generate performance trends report with step-level breakdowns."""
        runs = self.history_manager.get_run_history()

        # Filter by product if specified
        if product_id:
            runs = [run for run in runs if run.product_id == product_id]

        # Filter by date range
        cutoff_date = datetime.now(tz=UTC) - timedelta(days=days)
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

        # Calculate daily aggregates (pipeline-level)
        daily_stats: dict[str, dict[str, Any]] = {}
        # Step-level daily aggregates
        step_daily: dict[str, dict[str, list[float]]] = {}

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

            # Collect step-level durations per day
            for step_data in run.step_metrics:
                step_name = step_data["step_name"]
                if step_name not in step_daily:
                    step_daily[step_name] = {}
                if date_key not in step_daily[step_name]:
                    step_daily[step_name][date_key] = []
                step_daily[step_name][date_key].append(step_data["duration"])

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

        # Build step-level trend data
        step_trends: dict[str, list[dict[str, Any]]] = {}
        for step_name, dates in sorted(step_daily.items()):
            step_trends[step_name] = []
            for date in sorted(dates):
                vals = dates[date]
                step_trends[step_name].append(
                    {
                        "date": date,
                        "avg_duration": round(sum(vals) / len(vals), 3),
                        "count": len(vals),
                    }
                )

        return {
            "report_type": "trends",
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "filters": {
                "product_id": product_id,
                "days": days,
                "total_runs": len(runs),
            },
            "trend_data": trend_data,
            "step_trends": step_trends,
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
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "limit": limit,
            "runs": detailed_runs,
        }

    def generate_comparison_report(self, limit: int = 100) -> dict[str, Any]:
        """Generate profile-vs-profile performance comparison."""
        runs = self.history_manager.get_run_history(limit=limit)

        if not runs:
            return {"error": "No historical data available"}

        # Group runs by profile
        profiles: dict[str, list[PipelineRunMetrics]] = {}
        for run in runs:
            if run.profile_name not in profiles:
                profiles[run.profile_name] = []
            profiles[run.profile_name].append(run)

        if len(profiles) < 2:
            return {
                "report_type": "comparison",
                "generated_at": datetime.now(tz=UTC).isoformat(),
                "error": "Need at least 2 profiles to compare (found %d)"
                % len(profiles),
            }

        # Build per-profile stats
        profile_stats: dict[str, dict[str, Any]] = {}
        for profile_name, profile_runs in sorted(profiles.items()):
            durations = sorted(r.total_duration for r in profile_runs)
            peak_mems = [r.peak_memory for r in profile_runs]
            successes = sum(1 for r in profile_runs if r.success)

            profile_stats[profile_name] = {
                "run_count": len(profile_runs),
                "success_rate": round(successes / len(profile_runs) * 100, 1),
                "duration": {
                    "avg": round(sum(durations) / len(durations), 2),
                    "min": round(durations[0], 2),
                    "max": round(durations[-1], 2),
                    "p50": round(_percentile(durations, 50), 2),
                    "p95": round(_percentile(durations, 95), 2),
                },
                "peak_memory_avg_mb": round(sum(peak_mems) / len(peak_mems), 1),
            }

        return {
            "report_type": "comparison",
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "total_runs": len(runs),
            "profiles": profile_stats,
        }

    def detect_regressions(
        self, window: int = 10, threshold_factor: float = 2.0
    ) -> dict[str, Any]:
        """Detect performance regressions by comparing recent vs previous runs.

        Compares the last `window` runs against the previous `window` runs.
        Flags steps where recent average duration exceeds the previous
        average by more than `threshold_factor`.
        """
        runs = self.history_manager.get_run_history(limit=window * 2)

        if len(runs) < window * 2:
            return {
                "report_type": "regressions",
                "generated_at": datetime.now(tz=UTC).isoformat(),
                "error": "Not enough data: need %d runs, have %d"
                % (window * 2, len(runs)),
            }

        # runs are newest-first, so recent = [:window], previous = [window:]
        recent_runs = runs[:window]
        previous_runs = runs[window:]

        def _step_avg(run_list: list[PipelineRunMetrics]) -> dict[str, float]:
            step_totals: dict[str, list[float]] = {}
            for run in run_list:
                for step in run.step_metrics:
                    name = step["step_name"]
                    if name not in step_totals:
                        step_totals[name] = []
                    step_totals[name].append(step["duration"])
            return {name: sum(vals) / len(vals) for name, vals in step_totals.items()}

        recent_avgs = _step_avg(recent_runs)
        previous_avgs = _step_avg(previous_runs)

        regressions = []
        for step_name in recent_avgs:
            if step_name not in previous_avgs:
                continue
            prev = previous_avgs[step_name]
            curr = recent_avgs[step_name]
            if prev > 0 and curr / prev > threshold_factor:
                regressions.append(
                    {
                        "step": step_name,
                        "previous_avg": round(prev, 3),
                        "recent_avg": round(curr, 3),
                        "factor": round(curr / prev, 2),
                    }
                )

        # Also check pipeline-level duration
        recent_pipeline_avg = sum(r.total_duration for r in recent_runs) / len(
            recent_runs
        )
        previous_pipeline_avg = sum(r.total_duration for r in previous_runs) / len(
            previous_runs
        )

        pipeline_regression = None
        if (
            previous_pipeline_avg > 0
            and recent_pipeline_avg / previous_pipeline_avg > threshold_factor
        ):
            pipeline_regression = {
                "previous_avg": round(previous_pipeline_avg, 2),
                "recent_avg": round(recent_pipeline_avg, 2),
                "factor": round(recent_pipeline_avg / previous_pipeline_avg, 2),
            }

        return {
            "report_type": "regressions",
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "window": window,
            "threshold_factor": threshold_factor,
            "pipeline_regression": pipeline_regression,
            "step_regressions": regressions,
            "status": "regressions_found"
            if regressions or pipeline_regression
            else "ok",
        }

    def _analyze_step_performance(
        self, runs: list[PipelineRunMetrics]
    ) -> dict[str, Any]:
        """Analyze performance by pipeline step with percentiles."""
        step_stats: dict[str, dict[str, Any]] = {}

        for run in runs:
            for step_data in run.step_metrics:
                step_name = step_data["step_name"]

                if step_name not in step_stats:
                    step_stats[step_name] = {
                        "count": 0,
                        "durations": [],
                        "total_memory_delta": 0,
                        "error_count": 0,
                    }

                step_stats[step_name]["count"] += 1
                step_stats[step_name]["durations"].append(step_data["duration"])
                step_stats[step_name]["total_memory_delta"] += (
                    step_data["memory_end"] - step_data["memory_start"]
                )
                step_stats[step_name]["error_count"] += len(step_data.get("errors", []))

        # Calculate averages and percentiles
        step_analysis = {}
        for step_name, stats in step_stats.items():
            count = stats["count"]
            durations = sorted(stats["durations"])
            step_analysis[step_name] = {
                "execution_count": count,
                "average_duration": (
                    round(sum(durations) / count, 3) if count > 0 else 0
                ),
                "p50_duration": round(_percentile(durations, 50), 3),
                "p95_duration": round(_percentile(durations, 95), 3),
                "p99_duration": round(_percentile(durations, 99), 3),
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


def _report_to_csv(report: dict[str, Any]) -> str:
    """Convert a report dict to CSV string."""
    output = io.StringIO()
    report_type = report.get("report_type", "unknown")

    if report_type == "detailed":
        runs = report.get("runs", [])
        if not runs:
            return "No data\n"
        writer = csv.writer(output)
        writer.writerow(
            [
                "run_id",
                "product_id",
                "profile_name",
                "timestamp",
                "success",
                "total_duration",
                "memory_delta",
                "peak_memory",
                "cpu_percent",
                "error_message",
            ]
        )
        for run in runs:
            m = run["metrics"]
            writer.writerow(
                [
                    run["run_id"],
                    run["product_id"],
                    run["profile_name"],
                    run["timestamp"],
                    run["success"],
                    m["total_duration"],
                    m["memory_delta"],
                    m["peak_memory"],
                    m["cpu_percent"],
                    run.get("error_message", ""),
                ]
            )

    elif report_type == "trends":
        trend_data = report.get("trend_data", [])
        if not trend_data:
            return "No data\n"
        writer = csv.writer(output)
        writer.writerow(
            [
                "date",
                "run_count",
                "success_rate",
                "avg_duration",
                "avg_memory_delta",
                "avg_cpu",
            ]
        )
        for row in trend_data:
            writer.writerow(
                [
                    row["date"],
                    row["run_count"],
                    round(row["success_rate"], 1),
                    round(row["avg_duration"], 2),
                    round(row["avg_memory_delta"], 2),
                    round(row["avg_cpu"], 1),
                ]
            )

    else:
        # Fallback: dump as JSON
        return json.dumps(report, indent=2) + "\n"

    return output.getvalue()


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

  # Compare profiles
  python tools/performance_report.py --report-type comparison

  # Detect regressions
  python tools/performance_report.py --report-type regressions --window 10

  # Export detailed report as CSV
  python tools/performance_report.py --report-type detailed --format csv --limit 5

  # Save report to file
  python tools/performance_report.py --report-type summary --output report.json
        """,
    )

    parser.add_argument(
        "--report-type",
        choices=["summary", "trends", "detailed", "comparison", "regressions"],
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
        "--window",
        type=int,
        default=10,
        help="Window size for regression detection (compares last N vs previous N)",
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
        "--format",
        choices=["json", "pretty", "csv"],
        default="pretty",
        help="Output format",
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
    elif args.report_type == "comparison":
        report = generator.generate_comparison_report(limit=args.limit)
    elif args.report_type == "regressions":
        report = generator.detect_regressions(window=args.window)
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
        if args.format == "csv":
            print(_report_to_csv(report), end="")
        elif args.format == "json":
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
        print(f"\nERROR: {report['error']}")
        return

    if report_type == "summary":
        print_summary_report(report)
    elif report_type == "trends":
        print_trends_report(report)
    elif report_type == "detailed":
        print_detailed_report(report)
    elif report_type == "comparison":
        print_comparison_report(report)
    elif report_type == "regressions":
        print_regressions_report(report)


def print_summary_report(report: dict[str, Any]) -> None:
    """Print summary report in pretty format."""
    data_range = report.get("data_range", {})
    success = report.get("success_metrics", {})
    perf = report.get("performance_metrics", {})
    dist = report.get("distribution", {})
    steps = report.get("step_analysis", {})

    print("\nDATA OVERVIEW")
    print(f"   Total Runs: {data_range.get('total_runs', 0)}")
    print(
        f"   Date Range: {data_range.get('oldest_run', 'N/A')} to "
        f"{data_range.get('newest_run', 'N/A')}"
    )

    print("\nSUCCESS METRICS")
    print(
        f"   Success Rate: {success.get('success_rate_percent', 0)}% "
        f"({success.get('successful_runs', 0)}/{success.get('total_runs', 0)})"
    )
    print(f"   Failed Runs: {success.get('failed_runs', 0)}")

    print("\nPERFORMANCE METRICS")
    duration = perf.get("duration", {})
    memory = perf.get("memory", {})
    cpu = perf.get("cpu", {})

    print("   Duration:")
    print(f"     Average: {duration.get('average_seconds', 0)}s")
    print(
        f"     Range: {duration.get('minimum_seconds', 0)}s - "
        f"{duration.get('maximum_seconds', 0)}s"
    )
    print(
        f"     Percentiles: p50={duration.get('p50_seconds', 0)}s "
        f"p95={duration.get('p95_seconds', 0)}s "
        f"p99={duration.get('p99_seconds', 0)}s"
    )
    print(f"     Recent Avg: {duration.get('recent_average_seconds', 0)}s")

    print("   Memory:")
    print(f"     Avg Delta: {memory.get('average_delta_mb', 0)} MB")
    print(f"     Avg Peak: {memory.get('average_peak_mb', 0)} MB")

    print(f"   CPU: {cpu.get('average_percent', 0)}% average")

    print("\nDISTRIBUTION")
    products = dist.get("products", {})
    profiles = dist.get("profiles", {})

    print("   Top Products:")
    for product, count in list(products.items())[:5]:
        print(f"     {product}: {count} runs")

    print("   Profiles:")
    for profile, count in profiles.items():
        print(f"     {profile}: {count} runs")

    print("\nSTEP PERFORMANCE (Top 5 Slowest)")
    for i, (step_name, stats) in enumerate(list(steps.items())[:5]):
        print(f"   {i+1}. {step_name}:")
        print(f"      Avg Duration: {stats.get('average_duration', 0)}s")
        print(
            f"      p50={stats.get('p50_duration', 0)}s "
            f"p95={stats.get('p95_duration', 0)}s"
        )
        print(f"      Executions: {stats.get('execution_count', 0)}")
        print(f"      Error Rate: {stats.get('error_rate', 0)}%")


def print_trends_report(report: dict[str, Any]) -> None:
    """Print trends report in pretty format."""
    filters = report.get("filters", {})
    summary = report.get("summary", {})
    trends = report.get("trend_data", [])
    step_trends = report.get("step_trends", {})

    print("\nFILTERS")
    print(f"   Product ID: {filters.get('product_id', 'All')}")
    print(f"   Days: {filters.get('days', 0)}")
    print(f"   Total Runs: {filters.get('total_runs', 0)}")

    print("\nTREND SUMMARY")
    print(f"   Date Range: {summary.get('date_range', 'N/A')}")
    print(f"   Total Days: {summary.get('total_days', 0)}")
    print(f"   Avg Daily Runs: {summary.get('avg_daily_runs', 0):.1f}")

    print("\nDAILY TRENDS (Last 10 Days)")
    for trend in trends[-10:]:
        print(
            f"   {trend['date']}: {trend['run_count']} runs, "
            f"{trend['success_rate']:.1f}% success, "
            f"{trend['avg_duration']:.1f}s avg"
        )

    if step_trends:
        print("\nSTEP TRENDS (Last 5 Days per Step)")
        for step_name, entries in list(step_trends.items())[:5]:
            print(f"   {step_name}:")
            for entry in entries[-5:]:
                print(
                    f"     {entry['date']}: {entry['avg_duration']}s avg "
                    f"({entry['count']} runs)"
                )


def print_detailed_report(report: dict[str, Any]) -> None:
    """Print detailed report in pretty format."""
    runs = report.get("runs", [])
    limit = report.get("limit", 0)

    print(f"\nDETAILED RUNS (Showing {len(runs)} of max {limit})")

    for i, run in enumerate(runs[:10]):  # Show first 10 runs
        status = "OK" if run["success"] else "FAIL"
        print(f"\n   {i+1}. [{status}] {run['run_id']}")
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


def print_comparison_report(report: dict[str, Any]) -> None:
    """Print profile comparison report."""
    profiles = report.get("profiles", {})

    if not profiles:
        return

    print("\nPROFILE COMPARISON")
    hdr = (
        f"   {'Profile':<25} {'Runs':>5} {'Success':>8}"
        f" {'Avg(s)':>8} {'p50(s)':>8} {'p95(s)':>8}"
        f" {'Mem(MB)':>8}"
    )
    print(hdr)
    sep = f"   {'-'*25} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}"
    print(sep)

    for name, stats in profiles.items():
        d = stats["duration"]
        print(
            f"   {name:<25} {stats['run_count']:>5} "
            f"{stats['success_rate']:>7.1f}% "
            f"{d['avg']:>8.2f} "
            f"{d['p50']:>8.2f} "
            f"{d['p95']:>8.2f} "
            f"{stats['peak_memory_avg_mb']:>8.1f}"
        )


def print_regressions_report(report: dict[str, Any]) -> None:
    """Print regression detection report."""
    status = report.get("status", "unknown")
    window = report.get("window", 0)
    threshold = report.get("threshold_factor", 0)

    print(f"\nREGRESSION DETECTION (window={window}, threshold={threshold}x)")
    print(f"   Status: {status}")

    pipeline = report.get("pipeline_regression")
    if pipeline:
        print(
            f"\n   PIPELINE REGRESSION: "
            f"{pipeline['previous_avg']}s -> {pipeline['recent_avg']}s "
            f"({pipeline['factor']}x slower)"
        )

    regressions = report.get("step_regressions", [])
    if regressions:
        print("\n   Step Regressions:")
        for reg in regressions:
            print(
                f"     {reg['step']}: "
                f"{reg['previous_avg']}s -> {reg['recent_avg']}s "
                f"({reg['factor']}x slower)"
            )
    elif not pipeline:
        print("\n   No regressions detected.")


if __name__ == "__main__":
    main()
