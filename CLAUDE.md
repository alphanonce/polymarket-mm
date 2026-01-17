# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Workflow (MUST FOLLOW)

코드 수정 요청 시 반드시 아래 워크플로우를 따를 것:

1. **브랜치 생성** - 작업 시작 전 feature 브랜치 생성
   ```bash
   git checkout -b feature/<작업명>
   ```
2. **수정 작업** - 요청한 변경사항 구현
3. **확인 요청** - 변경 내용을 사용자에게 보여주고 컨펌 받기
4. **커밋 & PR** - 컨펌되면 `/commit` 스킬로 커밋, 푸시, PR 생성

**브랜치 네이밍:**
- `feature/<기능명>` - 새 기능
- `fix/<버그명>` - 버그 수정
- `refactor/<대상>` - 리팩토링

## Project Overview

Polymarket-mm is a high-frequency trading (HFT) market-making system for Polymarket prediction markets. The system uses a hybrid architecture with a Go execution layer for low-latency order management and a Python strategy layer for quote/size computation.

**Key Features:**
- Target latency: 1-10ms tick-to-trade
- Event-driven architecture with shared memory IPC
- Multiple data sources: Polymarket, Binance, Chainlink
- Modular strategy models (quote/size)
- Full backtesting support

## Architecture

```
┌──────────────────────── Go Layer ────────────────────────┐
│  Polymarket WS  │  Binance WS  │  Chainlink Oracle       │
│        ↓               ↓              ↓                  │
│              ┌─────────────────────┐                     │
│              │    Aggregator       │                     │
│              └──────────┬──────────┘                     │
│                         ↓                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │          Shared Memory (mmap)                     │   │
│  │  - Market books  - External prices  - Positions   │   │
│  │  - Order signals - Strategy state                 │   │
│  └──────────────────────────────────────────────────┘   │
│                         ↓                                │
│              ┌─────────────────────┐                     │
│              │     Executor        │ → Polymarket API    │
│              └─────────────────────┘                     │
└──────────────────────────────────────────────────────────┘
                          ↕ (mmap)
┌──────────────────── Python Layer ────────────────────────┐
│              ┌─────────────────────┐                     │
│              │   SHM Reader        │                     │
│              └──────────┬──────────┘                     │
│                         ↓                                │
│              ┌─────────────────────┐                     │
│              │   Orchestrator      │                     │
│              └──────────┬──────────┘                     │
│                    ↙         ↘                           │
│           Quote Model    Size Model                      │
└──────────────────────────────────────────────────────────┘
```

## Project Structure

```
polymarket-mm/
├── go/
│   ├── cmd/
│   │   ├── executor/     # Main trading executable
│   │   ├── collector/    # Data collection tool
│   │   └── replay/       # Data replay for backtesting
│   ├── internal/
│   │   ├── polymarket/   # Polymarket CLOB client
│   │   ├── binance/      # Binance WebSocket client
│   │   ├── chainlink/    # Chainlink oracle reader
│   │   ├── shm/          # Shared memory management
│   │   ├── aggregator/   # Data aggregation
│   │   ├── executor/     # Order execution
│   │   └── collector/    # Snapshot collection
│   └── go.mod
├── python/
│   ├── strategy/
│   │   ├── shm/          # Shared memory reader/writer
│   │   ├── models/       # Quote and size models
│   │   └── orchestrator.py
│   ├── backtest/
│   │   ├── engine.py     # Backtest engine
│   │   ├── simulator.py  # Market simulator
│   │   └── data_loader.py
│   └── pyproject.toml
├── shared/
│   └── shm_layout.h      # Shared memory layout (source of truth)
├── data/
│   └── config/           # Configuration files
└── Makefile
```

## Development Setup

### Go Setup
```bash
cd go
go mod download
go mod tidy
```

### Python Setup
```bash
cd python
poetry install
```

## Commands

```bash
# Build
make build              # Build all components
make build-go           # Build Go executables only
make deps               # Install all dependencies

# Test
make test               # Run all tests
make test-go            # Run Go tests
make test-python        # Run Python tests

# Lint
make lint               # Lint all code
make format             # Format all code

# Run
make run-executor       # Run the Go executor
make run-collector      # Run data collector
make run-strategy       # Run Python strategy
make run-backtest       # Run backtester
```

## Key Files

- `shared/shm_layout.h` - Single source of truth for shared memory layout
- `go/internal/shm/types.go` - Go implementation of SHM types
- `python/strategy/shm/types.py` - Python implementation of SHM types
- `python/strategy/models/base.py` - Strategy model interfaces
- `data/config/executor.yaml` - Executor configuration
- `data/config/strategy.yaml` - Strategy configuration

## Environment Variables

```bash
POLYMARKET_API_KEY      # Polymarket L2 API key
POLYMARKET_API_SECRET   # Polymarket L2 API secret
POLYMARKET_PASSPHRASE   # Polymarket L2 passphrase
POLYGON_RPC_URL         # Polygon RPC for Chainlink
```

## Order Logic (IMPORTANT)

모든 호가는 **UP token 기준**으로 표시됨.

```
┌─────────────────────────────────────────────────────────────────┐
│  BID (UP token 매수 방향)                                        │
│  ─────────────────────────                                       │
│  if position_down < order_size:                                  │
│      → BUY UP token                                              │
│  else:                                                           │
│      → SELL DOWN token (= 실질적으로 UP 매수와 동일)              │
├─────────────────────────────────────────────────────────────────┤
│  ASK (UP token 매도 방향)                                        │
│  ─────────────────────────                                       │
│  if position_up < order_size:                                    │
│      → BUY DOWN token (= 실질적으로 UP 매도와 동일)               │
│  else:                                                           │
│      → SELL UP token                                             │
└─────────────────────────────────────────────────────────────────┘
```

**핵심 원리:**
- UP + DOWN = 1 (항상)
- SELL DOWN = BUY UP (경제적 등가)
- SELL UP = BUY DOWN (경제적 등가)
- 기존 포지션이 있으면 sell, 없으면 buy

## Strategy Models

Quote models compute bid/ask prices:
- `SpreadQuoteModel` - Fixed spread around mid price
- `InventoryAdjustedQuoteModel` - Skews prices based on inventory

Size models compute order sizes:
- `FixedSizeModel` - Fixed order sizes
- `InventoryBasedSizeModel` - Adjusts size based on position

## Adding New Features

1. **New Data Source**: Add to `go/internal/`, integrate in `aggregator.go`
2. **New Quote Model**: Implement `QuoteModel` interface in `python/strategy/models/`
3. **New Size Model**: Implement `SizeModel` interface in `python/strategy/models/`
4. **Modify SHM Layout**: Update `shared/shm_layout.h`, then update Go and Python implementations
