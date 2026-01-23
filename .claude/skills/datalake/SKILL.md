---
name: datalake
description: Use when user asks to "sync data", "fetch datalake", "download S3 data", "데이터 동기화", "datalake 받아줘", or mentions syncing Polymarket data from the data lake.
version: 0.2.0
---

# Data Lake Sync Skill

Synchronize Polymarket data from S3 Data Lake to local storage, then preprocess into 15-minute market segments.

## S3 Data Lake Info

- **Bucket**: `data-lake-prod-338401368290`
- **Region**: `eu-west-1`
- **Base Prefix**: `data/polymarket`
- **Local Storage**: `data/datalake/`
- **Processed Output**: `data/datalake/processed/15m/`

## Available Categories

| Category | Path | Description |
|----------|------|-------------|
| `15m` | timebased/crypto/updown/15m/ | 15-minute crypto up/down markets |
| `1h` | timebased/crypto/updown/1h/ | 1-hour crypto up/down markets |
| `4h` | timebased/crypto/updown/4h/ | 4-hour crypto up/down markets |
| `daily` | timebased/crypto/updown/daily/ | Daily crypto up/down markets |
| `above` | timebased/crypto/above/ | Crypto above/below markets |
| `price-on` | timebased/crypto/price-on/ | Price-on markets |
| `crypto_prices` | global/rtds_crypto_prices/ | Real-time crypto reference prices |
| `market_info` | global/market_info/ | Market metadata |
| `markets` | markets/ | Market listings |

## Instructions

### Step 1: Ask User for Categories

Use `AskUserQuestion` to ask which data to sync:

```
question: "Which data categories do you want to sync from the Data Lake?"
header: "Categories"
multiSelect: false
options:
  - label: "15m + crypto_prices (Recommended)"
    description: "15-minute markets and reference prices - most common for backtesting"
  - label: "All timebased"
    description: "All time-based markets (15m, 1h, 4h, daily, above, price-on) + crypto_prices"
  - label: "All categories"
    description: "Download everything including market_info and markets metadata"
```

### Step 2: Run Sync Script

Based on user selection, run the appropriate command:

**15m + crypto_prices (default):**
```bash
python3 scripts/fetch_datalake.py 15m crypto_prices
```

**All timebased:**
```bash
python3 scripts/fetch_datalake.py 15m 1h 4h daily above price-on crypto_prices
```

**All categories:**
```bash
python3 scripts/fetch_datalake.py --all
```

**Custom selection (if user chooses Other):**
```bash
python3 scripts/fetch_datalake.py <space-separated-categories>
```

### Step 3: Preprocess 15m Data

After syncing completes, run the preprocessing script to divide raw parquet files into 15-minute market segments:

```bash
python3 scripts/preprocess_15m_data.py -v
```

This script:
- Reads raw 30-minute overlapping files from `data/datalake/timebased/crypto/updown/15m/`
- Splits data by slug (market identifier)
- Filters to exact 15-minute market windows
- Removes duplicates and sorts by timestamp
- Outputs to `data/datalake/processed/15m/{symbol}/{date}/{slug}/{type}.parquet.gz`

### Step 4: Summarize Results

After both scripts complete, summarize:

**Sync Results:**
- Number of files **downloaded** (new files)
- Number of files **skipped** (already existed locally)
- Number of files **failed** (if any)

**Preprocessing Results:**
- Number of markets processed per symbol (BTC, ETH, SOL, XRP)
- Total markets processed
- Any coverage alerts (markets with less than 13 minutes of data)

**Data Locations:**
- Raw data: `data/datalake/`
- Processed data: `data/datalake/processed/15m/`

## Notes

- Requires AWS CLI configured with appropriate credentials
- Downloads are parallelized (8 workers by default)
- Existing files are automatically skipped (incremental sync)
- Use `--list` flag to see available categories: `python3 scripts/fetch_datalake.py --list`
- Preprocessing handles BTC, ETH, SOL, XRP by default
