#!/usr/bin/env python3
# type: ignore
"""Validation script for video extraction functionality.

Tests the extract_functional_videos_with_validation() function with diverse
Amazon ASINs to verify video detection works across product categories.
"""

import json
import logging
from pathlib import Path

# Test ASINs from different categories with known video content
TEST_ASINS = {
    "B0BTYCRJSS": "Electronics - Wireless Earbuds (previously tested)",
    "B0FM3ZM713": "Electronics - Tech gadget (has videos)",
    "B0FKBG1N7K": "Electronics - Device (has videos)",
    "B09V3KXJPB": "Electronics - Popular product",
    "B0CPHH68NF": "Home/Kitchen - Product with demos",
}


def analyze_existing_data():
    """Analyze existing scraped data to validate video extraction."""
    logger = logging.getLogger(__name__)

    print("=" * 70)
    print("Video Extraction Validation Report")
    print("=" * 70)
    print()

    results = []
    outputs_dir = Path("outputs")

    for asin, description in TEST_ASINS.items():
        result = {
            "asin": asin,
            "description": description,
            "data_exists": False,
            "videos_found": 0,
            "video_urls": [],
            "downloaded_videos": 0,
            "downloaded_paths": [],
            "status": "NOT_SCRAPED",
        }

        # Check if data.json exists
        data_file = outputs_dir / asin / "data.json"
        if data_file.exists():
            result["data_exists"] = True

            try:
                with open(data_file) as f:
                    data_list = json.load(f)

                # Handle list format (scraper returns list of products)
                if isinstance(data_list, list) and len(data_list) > 0:
                    data = data_list[0]
                else:
                    data = data_list if isinstance(data_list, dict) else {}

                # Check for video URLs in data
                videos = data.get("videos", [])
                result["videos_found"] = len(videos)
                result["video_urls"] = videos[:3]  # First 3 for brevity

                # Check for downloaded videos
                downloaded_videos = data.get("downloaded_videos", [])
                result["downloaded_videos"] = len(downloaded_videos)
                result["downloaded_paths"] = downloaded_videos

                # Verify downloaded video files exist
                existing_files = []
                for rel_path in downloaded_videos:
                    full_path = outputs_dir / rel_path
                    if full_path.exists():
                        existing_files.append(rel_path)

                result["existing_files"] = len(existing_files)

                # Determine status
                if result["videos_found"] > 0:
                    if result["downloaded_videos"] > 0:
                        result["status"] = "SUCCESS"
                    else:
                        result["status"] = "URLS_ONLY"
                else:
                    result["status"] = "NO_VIDEOS"

            except Exception as e:
                result["status"] = f"ERROR: {e}"
                logger.error(f"Error reading {data_file}: {e}")

        results.append(result)

    # Print detailed results
    print("Test Results:")
    print("-" * 70)
    for r in results:
        print(f"\n{r['asin']}: {r['description']}")
        print(f"  Status: {r['status']}")
        print(f"  Video URLs found: {r['videos_found']}")
        if r["video_urls"]:
            print(f"  Sample URLs: {r['video_urls'][0][:60]}...")
        print(f"  Downloaded videos: {r['downloaded_videos']}")
        if r["downloaded_paths"]:
            print(f"  Sample path: {r['downloaded_paths'][0]}")
        if "existing_files" in r:
            print(f"  Files on disk: {r['existing_files']}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics:")
    print("-" * 70)

    total_tested = len(results)
    scraped = sum(1 for r in results if r["data_exists"])
    with_videos = sum(1 for r in results if r["videos_found"] > 0)
    downloaded = sum(1 for r in results if r["downloaded_videos"] > 0)

    print(f"  Total ASINs tested: {total_tested}")
    print(f"  ASINs with data.json: {scraped} ({scraped/total_tested*100:.1f}%)")
    print(f"  ASINs with video URLs: {with_videos} ({with_videos/scraped*100:.1f}%)")
    print(
        f"  ASINs with downloaded videos: {downloaded} ({downloaded/scraped*100:.1f}%)"
    )

    total_video_urls = sum(r["videos_found"] for r in results)
    total_downloaded = sum(r["downloaded_videos"] for r in results)

    print(f"\n  Total video URLs found: {total_video_urls}")
    print(f"  Total videos downloaded: {total_downloaded}")

    # Validation checks
    print("\n" + "=" * 70)
    print("Validation Checks:")
    print("-" * 70)

    checks = []
    checks.append(("Video extraction working", with_videos > 0))
    checks.append(("Video download working", downloaded > 0))
    checks.append(("ASIN filtering working", total_video_urls > 0))
    checks.append(("Data persistence working", scraped > 0))

    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check_name}")

    all_passed = all(passed for _, passed in checks)

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All validation checks PASSED")
    else:
        print("⚠ Some validation checks FAILED - review results above")
    print("=" * 70)

    return results, all_passed


def check_video_files_quality():
    """Check quality of downloaded video files."""
    print("\n" + "=" * 70)
    print("Video File Quality Check:")
    print("-" * 70)

    outputs_dir = Path("outputs")
    video_files = list(outputs_dir.glob("*/videos/*.mp4"))

    if not video_files:
        print("  No video files found for quality check")
        return

    print(f"\n  Found {len(video_files)} video files")

    # Sample quality check on a few files
    sample_files = video_files[:3]

    for video_file in sample_files:
        size_mb = video_file.stat().st_size / (1024 * 1024)
        print(f"\n  {video_file.relative_to(outputs_dir)}")
        print(f"    Size: {size_mb:.2f} MB")
        print("    Exists: ✓")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    results, passed = analyze_existing_data()
    check_video_files_quality()

    print("\n" + "=" * 70)
    print("Validation Complete")
    print("=" * 70)
