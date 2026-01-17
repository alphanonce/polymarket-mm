// Package collector provides orderbook data collection functionality
package collector

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/alphanonce/polymarket-mm/internal/binance"
	"github.com/alphanonce/polymarket-mm/internal/polymarket"
	"go.uber.org/zap"
)

// Snapshot represents a point-in-time snapshot of market data
type Snapshot struct {
	Timestamp     time.Time                      `json:"timestamp"`
	TimestampNs   int64                          `json:"timestamp_ns"`
	PolyBooks     map[string]*PolyBookSnapshot   `json:"poly_books"`
	BinancePrices map[string]*BinancePriceSnapshot `json:"binance_prices"`
}

// PolyBookSnapshot is a snapshot of a Polymarket orderbook
type PolyBookSnapshot struct {
	AssetID   string           `json:"asset_id"`
	Bids      []PriceLevelSnap `json:"bids"`
	Asks      []PriceLevelSnap `json:"asks"`
	Timestamp int64            `json:"timestamp"`
}

// BinancePriceSnapshot is a snapshot of Binance prices
type BinancePriceSnapshot struct {
	Symbol    string    `json:"symbol"`
	Price     float64   `json:"price"`
	BidPrice  float64   `json:"bid_price"`
	AskPrice  float64   `json:"ask_price"`
	Timestamp time.Time `json:"timestamp"`
}

// PriceLevelSnap is a price level snapshot
type PriceLevelSnap struct {
	Price string `json:"price"`
	Size  string `json:"size"`
}

// Collector collects orderbook snapshots
type Collector struct {
	logger *zap.Logger

	// Data sources
	polyWS    *polymarket.WSClient
	binanceWS *binance.SpotClient

	// Markets to collect
	polyMarkets   []string
	binanceSymbols []string

	// Output
	outputDir string

	// State
	snapshots []Snapshot
	snapshotsMu sync.RWMutex

	// Control
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	// Config
	snapshotInterval time.Duration
	flushInterval    time.Duration
	maxSnapshots     int
}

// CollectorConfig configures the collector
type CollectorConfig struct {
	Logger           *zap.Logger
	PolyWS           *polymarket.WSClient
	BinanceWS        *binance.SpotClient
	PolyMarkets      []string
	BinanceSymbols   []string
	OutputDir        string
	SnapshotInterval time.Duration
	FlushInterval    time.Duration
	MaxSnapshots     int
}

// NewCollector creates a new collector
func NewCollector(cfg CollectorConfig) *Collector {
	if cfg.Logger == nil {
		cfg.Logger, _ = zap.NewProduction()
	}
	if cfg.SnapshotInterval == 0 {
		cfg.SnapshotInterval = 100 * time.Millisecond
	}
	if cfg.FlushInterval == 0 {
		cfg.FlushInterval = 1 * time.Minute
	}
	if cfg.MaxSnapshots == 0 {
		cfg.MaxSnapshots = 10000
	}
	if cfg.OutputDir == "" {
		cfg.OutputDir = "data/snapshots"
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &Collector{
		logger:           cfg.Logger,
		polyWS:           cfg.PolyWS,
		binanceWS:        cfg.BinanceWS,
		polyMarkets:      cfg.PolyMarkets,
		binanceSymbols:   cfg.BinanceSymbols,
		outputDir:        cfg.OutputDir,
		snapshots:        make([]Snapshot, 0, cfg.MaxSnapshots),
		ctx:              ctx,
		cancel:           cancel,
		snapshotInterval: cfg.SnapshotInterval,
		flushInterval:    cfg.FlushInterval,
		maxSnapshots:     cfg.MaxSnapshots,
	}
}

// Start starts the collector
func (c *Collector) Start() error {
	c.logger.Info("Starting collector",
		zap.Strings("polyMarkets", c.polyMarkets),
		zap.Strings("binanceSymbols", c.binanceSymbols),
		zap.String("outputDir", c.outputDir))

	// Ensure output directory exists
	if err := os.MkdirAll(c.outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output dir: %w", err)
	}

	// Start snapshot loop
	c.wg.Add(1)
	go c.snapshotLoop()

	// Start flush loop
	c.wg.Add(1)
	go c.flushLoop()

	return nil
}

// Stop stops the collector
func (c *Collector) Stop() {
	c.logger.Info("Stopping collector")
	c.cancel()
	c.wg.Wait()

	// Final flush
	c.flush()
}

func (c *Collector) snapshotLoop() {
	defer c.wg.Done()

	ticker := time.NewTicker(c.snapshotInterval)
	defer ticker.Stop()

	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			c.takeSnapshot()
		}
	}
}

func (c *Collector) flushLoop() {
	defer c.wg.Done()

	ticker := time.NewTicker(c.flushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			c.flush()
		}
	}
}

func (c *Collector) takeSnapshot() {
	now := time.Now()

	snap := Snapshot{
		Timestamp:     now,
		TimestampNs:   now.UnixNano(),
		PolyBooks:     make(map[string]*PolyBookSnapshot),
		BinancePrices: make(map[string]*BinancePriceSnapshot),
	}

	// Collect Polymarket orderbooks
	if c.polyWS != nil {
		for _, assetID := range c.polyMarkets {
			ob, ok := c.polyWS.GetOrderbook(assetID)
			if !ok {
				continue
			}

			bookSnap := &PolyBookSnapshot{
				AssetID:   assetID,
				Bids:      make([]PriceLevelSnap, 0, len(ob.Bids)),
				Asks:      make([]PriceLevelSnap, 0, len(ob.Asks)),
				Timestamp: ob.LastTimestamp,
			}

			for price, size := range ob.Bids {
				bookSnap.Bids = append(bookSnap.Bids, PriceLevelSnap{
					Price: price,
					Size:  size,
				})
			}
			for price, size := range ob.Asks {
				bookSnap.Asks = append(bookSnap.Asks, PriceLevelSnap{
					Price: price,
					Size:  size,
				})
			}

			snap.PolyBooks[assetID] = bookSnap
		}
	}

	// Collect Binance prices
	if c.binanceWS != nil {
		for _, symbol := range c.binanceSymbols {
			price, ok := c.binanceWS.GetPrice(symbol)
			if !ok {
				continue
			}

			snap.BinancePrices[symbol] = &BinancePriceSnapshot{
				Symbol:    symbol,
				Price:     price.Price,
				BidPrice:  price.BidPrice,
				AskPrice:  price.AskPrice,
				Timestamp: price.Timestamp,
			}
		}
	}

	c.snapshotsMu.Lock()
	c.snapshots = append(c.snapshots, snap)

	// Check if we need to flush
	if len(c.snapshots) >= c.maxSnapshots {
		go c.flush()
	}
	c.snapshotsMu.Unlock()
}

func (c *Collector) flush() {
	c.snapshotsMu.Lock()
	if len(c.snapshots) == 0 {
		c.snapshotsMu.Unlock()
		return
	}

	snaps := c.snapshots
	c.snapshots = make([]Snapshot, 0, c.maxSnapshots)
	c.snapshotsMu.Unlock()

	// Generate filename with timestamp
	ts := snaps[0].Timestamp
	filename := fmt.Sprintf("snapshots_%s.json", ts.Format("20060102_150405"))
	path := filepath.Join(c.outputDir, filename)

	// Write to file
	data, err := json.Marshal(snaps)
	if err != nil {
		c.logger.Error("Failed to marshal snapshots", zap.Error(err))
		return
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		c.logger.Error("Failed to write snapshots", zap.String("path", path), zap.Error(err))
		return
	}

	c.logger.Info("Flushed snapshots",
		zap.String("path", path),
		zap.Int("count", len(snaps)))
}

// GetSnapshotCount returns the current number of buffered snapshots
func (c *Collector) GetSnapshotCount() int {
	c.snapshotsMu.RLock()
	defer c.snapshotsMu.RUnlock()
	return len(c.snapshots)
}
