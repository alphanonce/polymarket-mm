# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

polymarket-mm is a high-frequency trading market-making system for Polymarket. It's a hybrid Go/Python system:
- **Go (go/)**: Data collection, order execution, market aggregation
- **Python (python/)**: Strategy logic, backtesting, analysis, paper trading

## Build & Run Commands

### Go Services
```bash
cd go/
go build ./cmd/executor      # Build executor
go build ./cmd/collector     # Build collector
go build ./cmd/replay        # Build replay tool

# Run with config
go run ./cmd/executor -config ../data/config/executor.yaml
go run ./cmd/collector -config ../data/config/collector.yaml
```

### Python Environment
```bash
cd python/
uv sync                      # Install dependencies (using uv)
uv lock                      # Update lock file
```

### Python Quality Tools
```bash
cd python/
mypy .                       # Type checking (strict mode)
ruff check .                 # Linting
ruff format .                # Formatting
pytest                       # Run tests
```

### Running the Strategy
```bash
cd python/
python -m strategy.orchestrator
```

### Backtesting
```bash
cd python/
python backtest/run_enhanced.py --config ../data/config/backtest.yaml
python -m streamlit run backtest/dashboard/app.py
```

### Paper Trading
```bash
cd python/
python -m paper.executor --config ../data/config/paper.yaml
```

## Architecture

### Data Flow
```
Polymarket WS  ──┐
Binance WS     ──┤──> [Go Aggregator] ──> [Shared Memory] ──> [Python Strategy]
Chainlink RPC  ──┘                                                    │
                     [Go Executor]  <─────────────────────────────────┘
```

### Inter-Process Communication
- Shared memory segment: `/polymarket_mm_shm`
- Go writes: market state, external prices, positions, open orders
- Python writes: order signals
- Layout defined in `/shared/shm_layout.h` (single source of truth)

### Key Components

**Go Services (go/cmd/)**:
- `executor/`: Order execution, risk management, Polymarket API integration
- `collector/`: Market data snapshots to disk
- `replay/`: Historical data replay

**Go Internal Packages (go/internal/)**:
- `aggregator/`: Data aggregation
- `binance/`: Binance spot/options integration
- `chainlink/`: Oracle price feeds (uses strings.NewReader for ABI parsing)
- `polymarket/`: API and WebSocket client
- `shm/`: Shared memory read/write
- `executor/`: Order execution and risk

**Python Strategy (python/strategy/)**:
- `orchestrator.py`: Main strategy loop (10ms ticks)
- `models/`: Quote and size models (base.py defines interfaces)
- `shm/`: Shared memory interface
- `utils/`: Polymarket tick sizes, Black-Scholes pricing, volatility, distributions

**Python Backtest (python/backtest/)**:
- `engine.py` / `engine_fast.py`: Backtest engines
- `simulator.py`: Market simulation
- `data_loader.py`: S3 and local data loading
- `dashboard/`: Streamlit visualization

**Python Analysis (python/analysis/)**:
- `distribution_models.py`: Distribution models for BS pricing evaluation
- `evaluate_distributions.py`: Evaluation framework for comparing models

**Python Paper Trading (python/paper/)**:
- `executor.py`: Paper trade execution engine
- `order_simulator.py`: Market order simulation with fills
- `position_tracker.py`: Position and P&L tracking

**Python Tests (python/tests/)**:
- `test_distributions.py`: Distribution model tests
- `test_black_scholes.py`: Black-Scholes pricing tests
- `test_polymarket_utils.py`: Polymarket utility tests

## Data Scripts

### Datalake Scripts (scripts/)
- `fetch_datalake.py`: S3 data sync for orderbook and market info
- `fetch_binance_klines.py`: Binance kline data fetcher
- `preprocess_15m_data.py`: 15-minute market data preprocessing
- `validate_datalake.py`: Data validation utilities

### Data Sync Command
```bash
cd scripts && python fetch_datalake.py 15m market_info
cd scripts && python fetch_datalake.py 15m timebased
```

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

## 15-Minute Updown Market Structure

### Underlying Assets
- BTC (Bitcoin), ETH (Ethereum), SOL (Solana), XRP (Ripple)

### Market Slug Format
`{asset}-updown-{timeframe}-{timestamp}`
- Example: `btc-updown-15m-1768665600`
- Timestamp: Unix epoch seconds of market START time

### UP/DOWN Outcome Determination
| Condition | Result |
|-----------|--------|
| Chainlink End Price >= Chainlink Start Price | **UP wins** (UP token pays $1) |
| Chainlink End Price < Chainlink Start Price | **DOWN wins** (DOWN token pays $1) |

### Token Structure
Each market has exactly 2 tokens:
- **UP token**: Pays $1 if underlying price goes UP
- **DOWN token**: Pays $1 if underlying price goes DOWN
- Token price range: $0.01 - $0.99

## Configuration

All config files in `data/config/`:
- `strategy.yaml`: Quote/size model parameters, tick interval, inventory limits
- `executor.yaml`: API credentials (via env vars), markets, risk limits
- `collector.yaml`: WebSocket endpoints, symbols, snapshot intervals
- `backtest.yaml`: Data sources, fees, simulation parameters
- `paper.yaml`: Paper trading configuration

## Environment Variables

Required for live trading:
- `POLYMARKET_API_KEY`
- `POLYMARKET_API_SECRET`
- `POLYMARKET_PASSPHRASE`
- `POLYGON_RPC_URL` (for Chainlink feeds)

## Code Style

- Python: line-length=100, Python 3.11+, strict mypy
- Go: standard formatting (gofmt)
- Both use structured logging (Python: structlog, Go: zap)

## Common Patterns

### Time-to-Expiry Calculation
```python
time_to_expiry_years = seconds_remaining / (365 * 24 * 3600)
```

### Volatility Windows
Default windows in periods: `[0.2, 0.5, 1, 2, 3, 5, 10]`
For 15m market: 0.2 periods = 3 minutes window
