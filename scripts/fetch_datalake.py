#!/usr/bin/env python3
"""
Data Lake에서 Polymarket 데이터 다운로드
- 15분 마켓 (updown/15m) 우선
- rtds_crypto_prices (참조 가격)
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Configuration
BUCKET = "data-lake-prod-338401368290"
REGION = "eu-west-1"
BASE_PREFIX = "data/polymarket"
LOCAL_BASE = Path(__file__).parent.parent / "data" / "datalake"

# Data categories to download
CATEGORIES = {
    "15m": f"{BASE_PREFIX}/timebased/crypto/updown/15m/",
    "1h": f"{BASE_PREFIX}/timebased/crypto/updown/1h/",
    "4h": f"{BASE_PREFIX}/timebased/crypto/updown/4h/",
    "daily": f"{BASE_PREFIX}/timebased/crypto/updown/daily/",
    "above": f"{BASE_PREFIX}/timebased/crypto/above/",
    "price-on": f"{BASE_PREFIX}/timebased/crypto/price-on/",
    "crypto_prices": f"{BASE_PREFIX}/global/rtds_crypto_prices/",
    "market_info": f"{BASE_PREFIX}/global/market_info/",
    "markets": f"{BASE_PREFIX}/markets/",
}


def run_aws_cmd(cmd: list[str]) -> tuple[int, str, str]:
    """Run AWS CLI command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def list_s3_files(prefix: str) -> list[dict]:
    """List all files under S3 prefix."""
    cmd = [
        "aws", "s3", "ls", f"s3://{BUCKET}/{prefix}",
        "--recursive", "--region", REGION
    ]
    rc, stdout, stderr = run_aws_cmd(cmd)
    if rc != 0:
        print(f"Error listing {prefix}: {stderr}")
        return []

    files = []
    for line in stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 4:
            size = int(parts[2])
            key = parts[3]
            files.append({"key": key, "size": size})
    return files


def download_file(key: str) -> tuple[str, bool, str]:
    """Download single file from S3. Returns (key, success, message)."""
    # Convert S3 key to local path
    relative = key.replace(f"{BASE_PREFIX}/", "")
    local_path = LOCAL_BASE / relative

    # Skip if exists
    if local_path.exists():
        return key, True, "exists"

    # Create parent directory
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Download
    cmd = [
        "aws", "s3", "cp",
        f"s3://{BUCKET}/{key}",
        str(local_path),
        "--region", REGION
    ]
    rc, _, stderr = run_aws_cmd(cmd)

    if rc == 0:
        return key, True, "downloaded"
    else:
        return key, False, stderr.strip()


def sync_category(category: str, prefix: str, max_workers: int = 8) -> None:
    """Sync a category from S3 to local."""
    print(f"\n{'='*60}")
    print(f"Syncing: {category}")
    print(f"S3: s3://{BUCKET}/{prefix}")
    print(f"{'='*60}")

    # List files
    print("Listing files...")
    files = list_s3_files(prefix)

    if not files:
        print("No files found.")
        return

    total_size_mb = sum(f["size"] for f in files) / (1024 * 1024)
    print(f"Found {len(files)} files ({total_size_mb:.1f} MB)")

    # Download with progress
    downloaded = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, f["key"]): f for f in files}

        for i, future in enumerate(as_completed(futures), 1):
            key, success, msg = future.result()

            if success:
                if msg == "exists":
                    skipped += 1
                else:
                    downloaded += 1
            else:
                failed += 1
                print(f"  FAILED: {key}: {msg}")

            # Progress every 100 files or at end
            if i % 100 == 0 or i == len(files):
                print(f"  Progress: {i}/{len(files)} "
                      f"(downloaded: {downloaded}, skipped: {skipped}, failed: {failed})")

    print(f"\nComplete: {downloaded} downloaded, {skipped} skipped, {failed} failed")


def main():
    parser = argparse.ArgumentParser(description="Fetch data from Data Lake")
    parser.add_argument(
        "categories",
        nargs="*",
        default=["15m", "crypto_prices"],
        help=f"Categories to sync. Available: {', '.join(CATEGORIES.keys())}"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Sync all categories"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available categories and exit"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="Number of parallel downloads (default: 8)"
    )
    args = parser.parse_args()

    if args.list:
        print("Available categories:")
        for name, prefix in CATEGORIES.items():
            print(f"  {name}: {prefix}")
        return

    # Create base directory
    LOCAL_BASE.mkdir(parents=True, exist_ok=True)
    print(f"Local directory: {LOCAL_BASE}")

    # Select categories
    if args.all:
        categories = list(CATEGORIES.keys())
    else:
        categories = args.categories
        # Validate
        for cat in categories:
            if cat not in CATEGORIES:
                print(f"Unknown category: {cat}")
                print(f"Available: {', '.join(CATEGORIES.keys())}")
                sys.exit(1)

    print(f"Categories to sync: {', '.join(categories)}")

    # Sync each category
    for category in categories:
        sync_category(category, CATEGORIES[category], args.workers)

    print("\n" + "="*60)
    print("All done!")
    print(f"Data stored in: {LOCAL_BASE}")
    print("="*60)


if __name__ == "__main__":
    main()
