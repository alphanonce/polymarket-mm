#!/usr/bin/env python3
"""
Validate datalake data for 15-minute updown markets.

Performs:
1. Schema validation (columns, types)
2. Data integrity checks (nulls, timestamps)
3. Cross-reference validation (orderbook <-> market_info)
4. Summary statistics
"""

import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np

# Data paths
BASE_PATH = Path(__file__).parent.parent / "data" / "datalake"
TIMEBASED_15M = BASE_PATH / "timebased" / "crypto" / "updown" / "15m"
MARKET_INFO_15M = BASE_PATH / "global" / "market_info" / "crypto" / "updown" / "15m"
BINANCE_KLINES = BASE_PATH / "global" / "binance_klines"

ASSETS = ["btc", "eth", "sol", "xrp"]

# Expected schemas
EXPECTED_SCHEMAS = {
    "book": {
        "required": ["ts", "slug", "bids", "asks"],
        "optional": ["local_ts", "condition_id", "token_id", "hash"],
    },
    "last_trade_price": {
        "required": ["ts", "slug"],
        "optional": ["local_ts", "price", "token_id", "condition_id", "hash"],
    },
    "price_change": {
        "required": ["ts"],
        "optional": ["slug", "price", "rtds_price", "local_ts"],
    },
    "market_info": {
        "required": ["slug", "status"],
        "optional": ["start_ts", "end_ts", "outcome", "close_price", "condition_id"],
    },
}


def load_parquet_file(filepath: Path) -> pd.DataFrame:
    """Load a single parquet file, handling gzip compression."""
    import gzip
    from io import BytesIO
    import pyarrow.parquet as pq

    if str(filepath).endswith('.parquet.gz'):
        # Decompress and read
        with gzip.open(filepath, 'rb') as f:
            data = f.read()
        buffer = BytesIO(data)
        table = pq.read_table(buffer)
        return table.to_pandas()
    else:
        return pd.read_parquet(filepath)


def load_parquet_files(directory: Path, pattern: str = "*.parquet*") -> pd.DataFrame:
    """Load and concatenate all parquet files matching pattern."""
    files = sorted(directory.rglob(pattern))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = load_parquet_file(f)
            dfs.append(df)
        except Exception as e:
            print(f"    Warning: Could not load {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def validate_schema(df: pd.DataFrame, file_type: str) -> List[str]:
    """Validate dataframe schema against expected columns."""
    issues = []

    if df.empty:
        return ["Empty dataframe"]

    schema = EXPECTED_SCHEMAS.get(file_type, {})
    required = set(schema.get("required", []))
    optional = set(schema.get("optional", []))
    expected = required | optional

    actual = set(df.columns)

    # Check missing required columns
    missing_required = required - actual
    if missing_required:
        issues.append(f"Missing required columns: {missing_required}")

    # Report extra columns (informational)
    extra = actual - expected
    if extra:
        issues.append(f"Extra columns found: {extra}")

    return issues


def validate_data_integrity(df: pd.DataFrame, file_type: str) -> List[str]:
    """Check for data integrity issues."""
    issues = []

    if df.empty:
        return issues

    # Check for null values in critical columns
    critical_cols = ["ts", "slug"] if "slug" in df.columns else ["ts"]
    for col in critical_cols:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                issues.append(f"Column '{col}' has {null_count} null values")

    # Check timestamp ordering
    if "ts" in df.columns:
        # Check for negative or zero timestamps
        invalid_ts = (df["ts"] <= 0).sum()
        if invalid_ts > 0:
            issues.append(f"Found {invalid_ts} invalid timestamps (<=0)")

        # Check for reasonable timestamp range (2024-2027)
        min_ts = df["ts"].min()
        max_ts = df["ts"].max()
        min_year = datetime.utcfromtimestamp(min_ts / 1000).year if min_ts > 0 else 0
        max_year = datetime.utcfromtimestamp(max_ts / 1000).year if max_ts > 0 else 0

        if min_year < 2024 or max_year > 2030:
            issues.append(f"Timestamp range suspect: {min_year}-{max_year}")

    # Check bids/asks are non-empty for orderbook
    if file_type == "book":
        import json
        def check_empty_orderbook(x):
            """Check if orderbook side is empty."""
            if isinstance(x, str):
                try:
                    parsed = json.loads(x)
                    return len(parsed) == 0
                except:
                    return True
            elif isinstance(x, list):
                return len(x) == 0
            return True

        if "bids" in df.columns:
            sample = df["bids"].head(100)
            empty_bids = sample.apply(check_empty_orderbook).sum()
            if empty_bids > len(sample) * 0.1:  # More than 10% empty
                issues.append(f"High empty bids ratio: {empty_bids}/{len(sample)} (sampled)")

        if "asks" in df.columns:
            sample = df["asks"].head(100)
            empty_asks = sample.apply(check_empty_orderbook).sum()
            if empty_asks > len(sample) * 0.1:
                issues.append(f"High empty asks ratio: {empty_asks}/{len(sample)} (sampled)")

    return issues


def validate_market_info(df: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
    """Validate market info data and return stats."""
    issues = []
    stats = {}

    if df.empty:
        return ["Empty market info"], stats

    # Check status distribution
    if "status" in df.columns:
        status_counts = df["status"].value_counts().to_dict()
        stats["status_distribution"] = status_counts

        # Each market should have exactly 2 records (active + resolved)
        if "slug" in df.columns:
            records_per_market = df.groupby("slug").size()
            single_record = (records_per_market == 1).sum()
            double_record = (records_per_market == 2).sum()
            more_records = (records_per_market > 2).sum()

            stats["single_record_markets"] = single_record
            stats["double_record_markets"] = double_record
            stats["extra_record_markets"] = more_records

            if more_records > 0:
                issues.append(f"{more_records} markets have >2 records")

    # Check outcome only present for resolved
    if "outcome" in df.columns and "status" in df.columns:
        active_with_outcome = ((df["status"] == "active") & (df["outcome"].notna())).sum()
        if active_with_outcome > 0:
            issues.append(f"{active_with_outcome} active records have outcome set")

        resolved_without_outcome = ((df["status"] == "resolved") & (df["outcome"].isna())).sum()
        if resolved_without_outcome > 0:
            issues.append(f"{resolved_without_outcome} resolved records missing outcome")

    # Check timestamps
    if "start_ts" in df.columns and "end_ts" in df.columns:
        invalid_duration = (df["end_ts"] <= df["start_ts"]).sum()
        if invalid_duration > 0:
            issues.append(f"{invalid_duration} markets have end_ts <= start_ts")

        # Check duration is ~15 minutes (900,000 ms)
        expected_duration_ms = 15 * 60 * 1000
        df_resolved = df[df["status"] == "resolved"].copy()
        if not df_resolved.empty:
            df_resolved["duration"] = df_resolved["end_ts"] - df_resolved["start_ts"]
            wrong_duration = (abs(df_resolved["duration"] - expected_duration_ms) > 10000).sum()  # 10s tolerance
            if wrong_duration > 0:
                issues.append(f"{wrong_duration} markets have unexpected duration")

    # Outcome distribution
    if "outcome" in df.columns:
        resolved = df[df["status"] == "resolved"]
        outcome_dist = resolved["outcome"].value_counts().to_dict()
        stats["outcome_distribution"] = outcome_dist

    return issues, stats


def get_unique_slugs(df: pd.DataFrame) -> Set[str]:
    """Extract unique slugs from dataframe."""
    if "slug" not in df.columns:
        return set()
    return set(df["slug"].dropna().unique())


def validate_cross_references(
    orderbook_slugs: Set[str],
    market_info_slugs: Set[str],
) -> List[str]:
    """Check that orderbook data has matching market info."""
    issues = []

    # Orderbook data should have market info
    missing_info = orderbook_slugs - market_info_slugs
    if missing_info:
        issues.append(f"{len(missing_info)} orderbook markets missing from market_info")

    # Market info should have orderbook data
    missing_orderbook = market_info_slugs - orderbook_slugs
    if missing_orderbook:
        issues.append(f"{len(missing_orderbook)} market_info entries missing orderbook data")

    return issues


def compute_statistics(df: pd.DataFrame, file_type: str) -> Dict[str, Any]:
    """Compute summary statistics."""
    stats = {}

    if df.empty:
        return stats

    stats["total_rows"] = len(df)

    if "slug" in df.columns:
        stats["unique_markets"] = df["slug"].nunique()

    if "ts" in df.columns:
        min_ts = df["ts"].min()
        max_ts = df["ts"].max()
        stats["date_range"] = {
            "min": datetime.utcfromtimestamp(min_ts / 1000).isoformat() if min_ts > 0 else None,
            "max": datetime.utcfromtimestamp(max_ts / 1000).isoformat() if max_ts > 0 else None,
        }

    if file_type == "book":
        # Average orderbook depth
        if "bids" in df.columns:
            depths = df["bids"].apply(lambda x: len(x) if isinstance(x, list) else 0)
            stats["avg_bid_depth"] = depths.mean()
        if "asks" in df.columns:
            depths = df["asks"].apply(lambda x: len(x) if isinstance(x, list) else 0)
            stats["avg_ask_depth"] = depths.mean()

    return stats


def validate_asset(asset: str) -> Dict[str, Any]:
    """Validate all data for a single asset."""
    print(f"\n{'='*60}")
    print(f"Validating {asset.upper()}")
    print("=" * 60)

    results = {
        "asset": asset,
        "issues": [],
        "stats": {},
    }

    # 1. Orderbook data
    print("\n  [1] Orderbook data (book_*.parquet.gz)...")
    book_path = TIMEBASED_15M / asset
    book_df = load_parquet_files(book_path, "book_*.parquet.gz")

    if not book_df.empty:
        issues = validate_schema(book_df, "book")
        issues.extend(validate_data_integrity(book_df, "book"))
        results["issues"].extend([f"book: {i}" for i in issues])
        results["stats"]["book"] = compute_statistics(book_df, "book")
        print(f"      Rows: {len(book_df):,}, Markets: {book_df['slug'].nunique() if 'slug' in book_df.columns else 'N/A'}")
        if issues:
            for i in issues:
                print(f"      Issue: {i}")
    else:
        print("      No data found")
        results["issues"].append("book: No data found")

    # 2. Last trade price data
    print("\n  [2] Last trade price (last_trade_price_*.parquet.gz)...")
    ltp_df = load_parquet_files(book_path, "last_trade_price_*.parquet.gz")

    if not ltp_df.empty:
        issues = validate_schema(ltp_df, "last_trade_price")
        issues.extend(validate_data_integrity(ltp_df, "last_trade_price"))
        results["issues"].extend([f"last_trade_price: {i}" for i in issues])
        results["stats"]["last_trade_price"] = compute_statistics(ltp_df, "last_trade_price")
        print(f"      Rows: {len(ltp_df):,}, Markets: {ltp_df['slug'].nunique() if 'slug' in ltp_df.columns else 'N/A'}")
        if issues:
            for i in issues:
                print(f"      Issue: {i}")
    else:
        print("      No data found")

    # 3. Price change (Chainlink RTDS) data
    print("\n  [3] Price change / RTDS (price_change_*.parquet.gz)...")
    price_df = load_parquet_files(book_path, "price_change_*.parquet.gz")

    if not price_df.empty:
        issues = validate_schema(price_df, "price_change")
        issues.extend(validate_data_integrity(price_df, "price_change"))
        results["issues"].extend([f"price_change: {i}" for i in issues])
        results["stats"]["price_change"] = compute_statistics(price_df, "price_change")
        print(f"      Rows: {len(price_df):,}")
        if issues:
            for i in issues:
                print(f"      Issue: {i}")
    else:
        print("      No data found")

    # 4. Market info
    print("\n  [4] Market info...")
    market_info_path = MARKET_INFO_15M / asset
    mi_df = load_parquet_files(market_info_path, "market_info_*.parquet.gz")

    if not mi_df.empty:
        issues, mi_stats = validate_market_info(mi_df)
        results["issues"].extend([f"market_info: {i}" for i in issues])
        results["stats"]["market_info"] = {**compute_statistics(mi_df, "market_info"), **mi_stats}
        print(f"      Rows: {len(mi_df):,}, Markets: {mi_df['slug'].nunique() if 'slug' in mi_df.columns else 'N/A'}")
        if "outcome_distribution" in mi_stats:
            print(f"      Outcomes: {mi_stats['outcome_distribution']}")
        if issues:
            for i in issues:
                print(f"      Issue: {i}")
    else:
        print("      No data found")
        results["issues"].append("market_info: No data found")

    # 5. Cross-reference check
    print("\n  [5] Cross-reference validation...")
    book_slugs = get_unique_slugs(book_df)
    mi_slugs = get_unique_slugs(mi_df)

    xref_issues = validate_cross_references(book_slugs, mi_slugs)
    results["issues"].extend([f"cross_ref: {i}" for i in xref_issues])
    if xref_issues:
        for i in xref_issues:
            print(f"      Issue: {i}")
    else:
        print("      OK - slugs match between orderbook and market_info")

    return results


def main():
    print("=" * 70)
    print("Polymarket 15m Datalake Validation")
    print("=" * 70)
    print(f"Base path: {BASE_PATH}")
    print(f"Timebased: {TIMEBASED_15M}")
    print(f"Market info: {MARKET_INFO_15M}")

    all_results = []

    for asset in ASSETS:
        results = validate_asset(asset)
        all_results.append(results)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    total_issues = 0
    for r in all_results:
        asset = r["asset"]
        issues = r["issues"]
        stats = r["stats"]

        print(f"\n{asset.upper()}:")
        if "book" in stats:
            print(f"  Orderbook: {stats['book'].get('total_rows', 0):,} rows, {stats['book'].get('unique_markets', 0):,} markets")
        if "market_info" in stats:
            print(f"  Market Info: {stats['market_info'].get('total_rows', 0):,} rows")
            if "outcome_distribution" in stats["market_info"]:
                od = stats["market_info"]["outcome_distribution"]
                print(f"  Outcomes: up={od.get('up', 0)}, down={od.get('down', 0)}")

        if issues:
            print(f"  Issues: {len(issues)}")
            total_issues += len(issues)
        else:
            print("  Issues: None")

    print("\n" + "=" * 70)
    if total_issues == 0:
        print("VALIDATION PASSED - No critical issues found")
    else:
        print(f"VALIDATION COMPLETED - {total_issues} issues found")
    print("=" * 70)


if __name__ == "__main__":
    main()
