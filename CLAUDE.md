# Project Context for Claude

## Datalake Data Structure

The datalake contains two main data types:

### Timebased Data (Orderbook)
- **Path**: `data/datalake/timebased/crypto/updown/{timeframe}/{asset}/`
- **Files**: `book_*.parquet.gz`, `last_trade_price_*.parquet.gz`
- **Columns**: `ts`, `local_ts`, `slug`, `condition_id`, `token_id`, `bids`, `asks`, `hash`
- **Schedule**: 30-minute batch uploads

### Market Info (Resolution Data)
- **Path**: `data/datalake/global/market_info/crypto/updown/{timeframe}/{asset}/`
- **Files**: `market_info_*.parquet.gz`
- **Key columns**:
  - `slug`: Original market slug (JOIN key)
  - `status`: `active` (subscription start) / `resolved` (market end)
  - `outcome`: `up` / `down` (only when `status='resolved'`)
  - `close_price`: End price (only when `status='resolved'`)
  - `start_ts`, `end_ts`: Market timestamps (ms)

**Important**: Each market has 2 records:
1. `active`: When market subscription starts
2. `resolved`: When market ends + outcome confirmed

### Asset Naming Convention
| Timeframe | Assets |
|-----------|--------|
| 5m, 15m, 4h | `btc`, `eth`, `sol`, `xrp` |
| 1h, daily, weekly | `bitcoin`, `ethereum`, `solana`, `xrp` |

### Slug Format
- Original: `btc-updown-15m-1768665600` (includes epoch timestamp)
- Normalized (S3 path): `btc-updown-15m`

### Data Sync Command
```bash
cd scripts && python fetch_datalake.py 15m market_info
```

## Key Analysis Modules

### `python/analysis/`
- `bs_validator.py`: Black-Scholes binary option pricing validation
- `bs_eval_orderbook.py`: BS evaluation using CSV orderbook data
- `datalake_loader.py`: Load orderbook data from datalake
- `binance_cache.py`: Cached Binance klines for volatility

## CRITICAL: Price Source for 15-Minute Markets

**The 15-minute updown markets use Chainlink (RTDS) as the reference price source.**

This means:
- **Strike price**: MUST use Chainlink price at market start (NOT Binance)
- **Outcome determination**: Based on Chainlink price at market end
- **BS model evaluation**: Strike price must be Chainlink to match actual market mechanics

**Why this matters:**
- Binance and Chainlink prices can differ by several basis points
- Using Binance strike price in BS model will produce incorrect fair value estimates
- The market resolves based on Chainlink, so all pricing models must use Chainlink as reference

**Chainlink data location:**
- Path: `data/datalake/timebased/crypto/updown/{timeframe}/{asset}/price_change_*.parquet.gz`
- Also available via: `DatalakeOrderbookLoader.get_rtds_prices_for_period()`

## Common Patterns

### Time-to-Expiry Calculation
```python
time_to_expiry_years = seconds_remaining / (365 * 24 * 3600)
```

### Volatility Windows
Default windows in periods: `[0.2, 0.5, 1, 2, 3, 5, 10]`
For 15m market: 0.2 periods = 3 minutes window
