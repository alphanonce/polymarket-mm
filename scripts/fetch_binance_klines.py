#!/usr/bin/env python3
"""
Fetch historical klines from Binance API.

Fetches 1-minute klines for crypto assets to support distribution model evaluation.
"""

import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "datalake" / "global" / "binance_klines"
BASE_URL = "https://api.binance.com/api/v3/klines"

ASSETS = {
    "btc": "BTCUSDT",
    "eth": "ETHUSDT",
    "sol": "SOLUSDT",
    "xrp": "XRPUSDT",
}

COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def fetch_klines(symbol: str, start_time: int, end_time: int, interval: str = "1m") -> pd.DataFrame:
    """Fetch klines from Binance API."""
    all_data = []
    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000,  # Max allowed
        }

        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_data.extend(data)

            # Move start to after last kline
            last_close_time = data[-1][6]
            current_start = last_close_time + 1

            # Rate limiting
            time.sleep(0.2)

        except Exception as e:
            print(f"    Error fetching {symbol}: {e}")
            break

    if not all_data:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame(all_data, columns=COLUMNS)

    # Convert types
    df["open_time"] = pd.to_numeric(df["open_time"])
    df["close_time"] = pd.to_numeric(df["close_time"])
    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_base_volume", "taker_buy_quote_volume"]:
        df[col] = pd.to_numeric(df[col])
    df["trades"] = pd.to_numeric(df["trades"])

    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch Binance klines data")
    parser.add_argument("--days", type=int, default=7, help="Number of days of data to fetch")
    parser.add_argument("--assets", nargs="+", default=list(ASSETS.keys()), help="Assets to fetch")

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=args.days)).timestamp() * 1000)

    print("=" * 70)
    print("Binance Klines Fetcher")
    print(f"Period: {args.days} days")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    for asset in args.assets:
        symbol = ASSETS.get(asset)
        if not symbol:
            print(f"Unknown asset: {asset}")
            continue

        print(f"\n[{asset.upper()}] Fetching {symbol}...")

        df = fetch_klines(symbol, start_time, end_time)

        if df.empty:
            print(f"  No data fetched")
            continue

        # Save to parquet
        output_file = OUTPUT_DIR / f"{symbol.lower()}_{args.days}d.parquet"
        df.to_parquet(output_file, index=False)

        print(f"  Saved {len(df):,} klines to {output_file}")
        print(f"  Date range: {datetime.fromtimestamp(df['open_time'].min()/1000)} - {datetime.fromtimestamp(df['open_time'].max()/1000)}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
