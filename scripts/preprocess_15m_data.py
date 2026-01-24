#!/usr/bin/env python3
"""
Preprocess 15-minute updown market data.

Takes raw 30-minute overlapping parquet files from the datalake
and splits them into individual market segments.

Output structure:
    data/datalake/processed/15m/{symbol}/{date}/{slug}/{type}.parquet.gz

Where:
    - symbol: btc, eth, sol, xrp
    - date: YYYY-MM-DD
    - slug: e.g., btc-updown-15m-1768665600
    - type: book, last_trade_price, price_change
"""

import argparse
import gzip
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO

RAW_BASE = Path(__file__).parent.parent / "data" / "datalake" / "timebased" / "crypto" / "updown" / "15m"


def read_parquet_gz(filepath: Path) -> pd.DataFrame:
    """Read a gzip-compressed parquet file."""
    if str(filepath).endswith('.parquet.gz'):
        with gzip.open(filepath, 'rb') as f:
            data = f.read()
        buffer = BytesIO(data)
        table = pq.read_table(buffer)
        return table.to_pandas()
    else:
        return pd.read_parquet(filepath)
OUTPUT_BASE = Path(__file__).parent.parent / "data" / "datalake" / "processed" / "15m"
SYMBOLS = ["btc", "eth", "sol", "xrp"]

# Market duration in milliseconds
MARKET_DURATION_MS = 15 * 60 * 1000  # 15 minutes


def load_raw_files(symbol: str, file_type: str) -> pd.DataFrame:
    """Load all raw parquet files for a symbol and type."""
    symbol_dir = RAW_BASE / symbol
    if not symbol_dir.exists():
        print(f"    Warning: {symbol_dir} does not exist")
        return pd.DataFrame()

    # Use recursive glob to find files in nested directories
    pattern = f"**/{file_type}_*.parquet.gz"
    files = sorted(symbol_dir.glob(pattern))

    if not files:
        print(f"    No {file_type} files found for {symbol}")
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = read_parquet_gz(f)
            dfs.append(df)
        except Exception as e:
            print(f"    Error loading {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Remove duplicates based on hash or timestamp
    if "hash" in combined.columns:
        combined = combined.drop_duplicates(subset=["hash"])
    elif "ts" in combined.columns:
        combined = combined.drop_duplicates(subset=["ts", "slug"])

    # Sort by timestamp
    if "ts" in combined.columns:
        combined = combined.sort_values("ts").reset_index(drop=True)

    return combined


def extract_market_timestamps(slug: str) -> Tuple[int, int]:
    """Extract start/end timestamps from slug.

    Slug format: btc-updown-15m-1768665600
    The last part is the Unix timestamp (seconds) of market start.
    """
    parts = slug.split("-")
    start_ts_sec = int(parts[-1])
    start_ts_ms = start_ts_sec * 1000
    end_ts_ms = start_ts_ms + MARKET_DURATION_MS
    return start_ts_ms, end_ts_ms


def filter_to_market_window(df: pd.DataFrame, slug: str) -> pd.DataFrame:
    """Filter dataframe to exact market window based on slug."""
    if df.empty or "ts" not in df.columns:
        return df

    start_ts, end_ts = extract_market_timestamps(slug)

    # Filter to market window
    mask = (df["ts"] >= start_ts) & (df["ts"] <= end_ts)
    filtered = df[mask].copy()

    return filtered


def get_date_from_slug(slug: str) -> str:
    """Extract date string from slug."""
    parts = slug.split("-")
    ts_sec = int(parts[-1])
    dt = datetime.utcfromtimestamp(ts_sec)
    return dt.strftime("%Y-%m-%d")


def process_symbol(symbol: str, verbose: bool = False) -> Dict[str, int]:
    """Process all files for a symbol."""
    stats = defaultdict(int)

    print(f"\n  Processing {symbol.upper()}...")

    # Load all raw data for each type
    file_types = ["book", "last_trade_price", "price_change"]
    raw_data = {}

    for ft in file_types:
        df = load_raw_files(symbol, ft)
        if not df.empty:
            raw_data[ft] = df
            print(f"    Loaded {ft}: {len(df)} rows, {df['slug'].nunique() if 'slug' in df.columns else 'N/A'} unique slugs")

    if not raw_data:
        print(f"    No data found for {symbol}")
        return dict(stats)

    # Get all unique slugs
    all_slugs = set()
    for ft, df in raw_data.items():
        if "slug" in df.columns:
            all_slugs.update(df["slug"].unique())

    print(f"    Found {len(all_slugs)} unique markets")

    # Process each market
    for slug in sorted(all_slugs):
        date_str = get_date_from_slug(slug)
        output_dir = OUTPUT_BASE / symbol / date_str / slug
        output_dir.mkdir(parents=True, exist_ok=True)

        market_has_data = False

        for ft, df in raw_data.items():
            if "slug" not in df.columns:
                continue

            # Filter to this slug
            slug_df = df[df["slug"] == slug].copy()
            if slug_df.empty:
                continue

            # Filter to exact market window
            market_df = filter_to_market_window(slug_df, slug)
            if market_df.empty:
                continue

            market_has_data = True

            # Check coverage (should have ~15 minutes of data)
            if "ts" in market_df.columns and len(market_df) > 1:
                ts_range_ms = market_df["ts"].max() - market_df["ts"].min()
                coverage_pct = ts_range_ms / MARKET_DURATION_MS * 100

                if coverage_pct < 85 and verbose:
                    print(f"      Warning: {slug}/{ft} has only {coverage_pct:.1f}% coverage")

            # Save processed file
            output_path = output_dir / f"{ft}.parquet.gz"
            market_df.to_parquet(output_path, compression="gzip", index=False)

            stats[f"{ft}_files"] += 1

        if market_has_data:
            stats["markets"] += 1

    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description="Preprocess 15-minute market data")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS, help="Symbols to process")

    args = parser.parse_args()

    print("=" * 70)
    print("15-Minute Market Data Preprocessing")
    print(f"Input: {RAW_BASE}")
    print(f"Output: {OUTPUT_BASE}")
    print("=" * 70)

    total_stats = defaultdict(int)

    for symbol in args.symbols:
        stats = process_symbol(symbol, args.verbose)
        for k, v in stats.items():
            total_stats[k] += v

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print(f"  Total markets: {total_stats['markets']}")
    print(f"  Book files: {total_stats['book_files']}")
    print(f"  Last trade price files: {total_stats['last_trade_price_files']}")
    print(f"  Price change files: {total_stats['price_change_files']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
