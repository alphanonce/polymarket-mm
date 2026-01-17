# Polymarket Market Making System Makefile

.PHONY: all build build-go build-python clean test test-go test-python lint lint-go lint-python run-executor run-collector run-strategy run-backtest deps deps-go deps-python

# Default target
all: deps build test

# ============================================================================
# Dependencies
# ============================================================================

deps: deps-go deps-python

deps-go:
	@echo "Installing Go dependencies..."
	cd go && go mod download
	cd go && go mod tidy

deps-python:
	@echo "Installing Python dependencies..."
	cd python && poetry install

# ============================================================================
# Build
# ============================================================================

build: build-go build-python

build-go:
	@echo "Building Go executables..."
	cd go && go build -o ../bin/executor ./cmd/executor
	cd go && go build -o ../bin/collector ./cmd/collector

build-python:
	@echo "Python package ready (interpreted language)"

# Build with optimizations
build-release:
	@echo "Building Go executables with optimizations..."
	cd go && CGO_ENABLED=0 go build -ldflags="-s -w" -o ../bin/executor ./cmd/executor
	cd go && CGO_ENABLED=0 go build -ldflags="-s -w" -o ../bin/collector ./cmd/collector

# ============================================================================
# Test
# ============================================================================

test: test-go test-python

test-go:
	@echo "Running Go tests..."
	cd go && go test -v ./...

test-python:
	@echo "Running Python tests..."
	cd python && poetry run pytest -v

test-coverage:
	@echo "Running tests with coverage..."
	cd go && go test -coverprofile=coverage.out ./...
	cd go && go tool cover -html=coverage.out -o coverage.html
	cd python && poetry run pytest --cov=strategy --cov=backtest --cov-report=html

# ============================================================================
# Lint
# ============================================================================

lint: lint-go lint-python

lint-go:
	@echo "Linting Go code..."
	cd go && go vet ./...
	@command -v golangci-lint >/dev/null 2>&1 && cd go && golangci-lint run || echo "golangci-lint not installed, skipping"

lint-python:
	@echo "Linting Python code..."
	cd python && poetry run ruff check .
	cd python && poetry run ruff format --check .
	cd python && poetry run mypy strategy backtest

format:
	@echo "Formatting code..."
	cd go && go fmt ./...
	cd python && poetry run ruff format .
	cd python && poetry run ruff check --fix .

# ============================================================================
# Run
# ============================================================================

run-executor:
	@echo "Starting executor..."
	./bin/executor -config data/config/executor.yaml

run-collector:
	@echo "Starting data collector..."
	./bin/collector -config data/config/collector.yaml

run-strategy:
	@echo "Starting Python strategy..."
	cd python && PYTHONPATH=. poetry run python -m strategy.orchestrator

run-backtest:
	@echo "Running backtest..."
	cd python && PYTHONPATH=. poetry run python -m backtest.engine

run-trades-backtest:
	@echo "Running trades-based backtest..."
	cd python && PYTHONPATH=. poetry run python -c "from backtest.engine import run_trades_backtest; run_trades_backtest()"

run-fast-backtest:
	@echo "Running fast backtest with visualization..."
	cd python && PYTHONPATH=. python3 -c "\
		import os; \
		[os.environ.__setitem__(k, v) for k, v in (l.strip().split('=', 1) for l in open('../.env') if l.strip() and not l.startswith('#') and '=' in l)]; \
		import sys; sys.path.insert(0, '.'); \
		from backtest import FastBacktestConfig, FastBacktestEngine, generate_report; \
		config = FastBacktestConfig.from_yaml('../data/config/backtest.yaml'); \
		engine = FastBacktestEngine(config); \
		report = engine.run(); \
		generate_report(report)"

run-backtest-enhanced:
	@echo "Running enhanced backtest with timeseries collection..."
	cd python && PYTHONPATH=. poetry run python -m backtest.run_enhanced --config ../data/config/backtest.yaml

run-dashboard:
	@echo "Starting backtest dashboard..."
	cd python && PYTHONPATH=. poetry run streamlit run backtest/dashboard/app.py

fetch-s3-data:
	@echo "Fetching S3 data..."
	cd python && PYTHONPATH=. poetry run python ../scripts/fetch_s3_data.py

# Run all components (in separate terminals recommended)
run-all:
	@echo "Starting all components..."
	@echo "Run these in separate terminals:"
	@echo "  make run-executor"
	@echo "  make run-strategy"

# ============================================================================
# Development
# ============================================================================

# Create shared memory for testing
create-shm:
	@echo "Creating shared memory file..."
	cd go && go run -exec "sudo" ./cmd/executor -create-shm

# Watch for changes and rebuild
watch-go:
	@command -v air >/dev/null 2>&1 && cd go && air || echo "air not installed: go install github.com/cosmtrek/air@latest"

# ============================================================================
# Docker
# ============================================================================

docker-build:
	docker build -t polymarket-mm:latest .

docker-run:
	docker run -it --rm polymarket-mm:latest

# ============================================================================
# Clean
# ============================================================================

clean:
	@echo "Cleaning build artifacts..."
	rm -rf bin/
	rm -rf go/coverage.out go/coverage.html
	rm -rf python/.pytest_cache python/.mypy_cache python/.ruff_cache
	rm -rf python/**/__pycache__
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -delete

clean-data:
	@echo "Cleaning data files..."
	rm -rf data/snapshots/*.json

# ============================================================================
# Help
# ============================================================================

help:
	@echo "Polymarket Market Making System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  deps               Install all dependencies"
	@echo "  build              Build all components"
	@echo "  test               Run all tests"
	@echo "  lint               Lint all code"
	@echo "  format             Format all code"
	@echo ""
	@echo "  run-executor       Run the Go executor"
	@echo "  run-collector      Run the data collector"
	@echo "  run-strategy       Run the Python strategy"
	@echo "  run-backtest       Run backtester (synthetic data)"
	@echo "  run-trades-backtest Run trades-based backtest (S3 data)"
	@echo "  run-backtest-enhanced Run enhanced backtest with timeseries"
	@echo "  run-dashboard      Start the Streamlit backtest dashboard"
	@echo "  fetch-s3-data      Fetch and inspect S3 data"
	@echo ""
	@echo "  clean              Clean build artifacts"
	@echo "  help               Show this help"
