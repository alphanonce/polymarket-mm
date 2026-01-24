#!/usr/bin/env python3
"""
Fetch Polymarket data from S3 Data Lake.

Usage:
    python fetch_datalake.py 15m crypto_prices     # Sync 15m markets + reference prices
    python fetch_datalake.py --all                 # Sync all categories
    python fetch_datalake.py --list                # List available categories

S3 Bucket: data-lake-prod-338401368290 (eu-west-1)
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import boto3
from botocore.config import Config

BUCKET = "data-lake-prod-338401368290"
REGION = "eu-west-1"
S3_PREFIX = "data/polymarket"
LOCAL_BASE = Path(__file__).parent.parent / "data" / "datalake"

# Category definitions: name -> S3 path (relative to S3_PREFIX)
CATEGORIES = {
    "15m": "timebased/crypto/updown/15m",
    "1h": "timebased/crypto/updown/1h",
    "4h": "timebased/crypto/updown/4h",
    "daily": "timebased/crypto/updown/daily",
    "above": "timebased/crypto/above",
    "price-on": "timebased/crypto/price-on",
    "crypto_prices": "global/rtds_crypto_prices",
    "market_info": "global/market_info",
    "binance_klines": "global/binance_klines",
    "markets": "markets",
}


@dataclass
class SyncStats:
    """Sync operation statistics."""
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    total_bytes: int = 0


def get_s3_client():
    """Create S3 client for eu-west-1."""
    config = Config(
        region_name=REGION,
        retries={"max_attempts": 3, "mode": "adaptive"},
    )
    return boto3.client("s3", config=config)


def list_s3_files(s3, prefix: str) -> List[dict]:
    """List all files under an S3 prefix."""
    files = []
    paginator = s3.get_paginator("list_objects_v2")

    full_prefix = f"{S3_PREFIX}/{prefix}"
    print(f"  Listing: s3://{BUCKET}/{full_prefix}/")

    for page in paginator.paginate(Bucket=BUCKET, Prefix=full_prefix):
        for obj in page.get("Contents", []):
            files.append({
                "key": obj["Key"],
                "size": obj["Size"],
                "local_path": LOCAL_BASE / obj["Key"].replace(f"{S3_PREFIX}/", ""),
            })

    return files


def download_file(s3, file_info: dict) -> tuple:
    """Download a single file. Returns (success, skipped, bytes)."""
    key = file_info["key"]
    local_path = file_info["local_path"]
    size = file_info["size"]

    # Skip if exists and same size
    if local_path.exists():
        if local_path.stat().st_size == size:
            return (False, True, 0)

    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, key, str(local_path))
        return (True, False, size)
    except Exception as e:
        print(f"    ERROR: {key}: {e}")
        return (False, False, 0)


def sync_category(s3, category: str, max_workers: int = 8) -> SyncStats:
    """Sync all files in a category."""
    if category not in CATEGORIES:
        print(f"  Unknown category: {category}")
        return SyncStats()

    prefix = CATEGORIES[category]
    files = list_s3_files(s3, prefix)

    if not files:
        print(f"  No files found for {category}")
        return SyncStats()

    print(f"  Found {len(files)} files ({sum(f['size'] for f in files) / 1024 / 1024:.1f} MB)")

    stats = SyncStats()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, s3, f): f for f in files}

        for future in as_completed(futures):
            success, skipped, size = future.result()
            if success:
                stats.downloaded += 1
                stats.total_bytes += size
            elif skipped:
                stats.skipped += 1
            else:
                stats.failed += 1

    return stats


def list_categories():
    """Print available categories."""
    print("\nAvailable categories:")
    print("-" * 60)
    for name, path in CATEGORIES.items():
        print(f"  {name:<15} -> {path}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Fetch Polymarket data from S3 Data Lake")
    parser.add_argument("categories", nargs="*", help="Categories to sync")
    parser.add_argument("--all", action="store_true", help="Sync all categories")
    parser.add_argument("--list", action="store_true", help="List available categories")
    parser.add_argument("--workers", type=int, default=8, help="Parallel download workers")

    args = parser.parse_args()

    if args.list:
        list_categories()
        return

    if args.all:
        categories = list(CATEGORIES.keys())
    elif args.categories:
        categories = args.categories
    else:
        print("Usage: python fetch_datalake.py <categories...>")
        print("       python fetch_datalake.py --all")
        print("       python fetch_datalake.py --list")
        sys.exit(1)

    print("=" * 70)
    print("Polymarket Data Lake Sync")
    print(f"Bucket: s3://{BUCKET}/{S3_PREFIX}/")
    print(f"Local: {LOCAL_BASE}")
    print("=" * 70)

    s3 = get_s3_client()

    total_stats = SyncStats()

    for category in categories:
        print(f"\n[{category}]")
        stats = sync_category(s3, category, args.workers)

        total_stats.downloaded += stats.downloaded
        total_stats.skipped += stats.skipped
        total_stats.failed += stats.failed
        total_stats.total_bytes += stats.total_bytes

        print(f"  Downloaded: {stats.downloaded}, Skipped: {stats.skipped}, Failed: {stats.failed}")

    print("\n" + "=" * 70)
    print("SYNC COMPLETE")
    print(f"  Downloaded: {total_stats.downloaded} files ({total_stats.total_bytes / 1024 / 1024:.1f} MB)")
    print(f"  Skipped:    {total_stats.skipped} files (already exist)")
    print(f"  Failed:     {total_stats.failed} files")
    print("=" * 70)


if __name__ == "__main__":
    main()
