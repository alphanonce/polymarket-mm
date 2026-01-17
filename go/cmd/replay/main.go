// Package main is the entry point for the data replay tool
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/alphanonce/polymarket-mm/internal/shm"
	"go.uber.org/zap"
)

// Snapshot represents a recorded data snapshot
type Snapshot struct {
	Timestamp   time.Time                    `json:"timestamp"`
	TimestampNs int64                        `json:"timestamp_ns"`
	PolyBooks   map[string]*PolyBookSnapshot `json:"poly_books"`
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

func main() {
	dataDir := flag.String("data", "data/snapshots", "Directory containing snapshot files")
	speed := flag.Float64("speed", 1.0, "Replay speed multiplier")
	flag.Parse()

	// Initialize logger
	logger, _ := zap.NewProduction()
	defer logger.Sync()

	// Find snapshot files
	files, err := filepath.Glob(filepath.Join(*dataDir, "snapshots_*.json"))
	if err != nil {
		logger.Fatal("Failed to find snapshot files", zap.Error(err))
	}
	if len(files) == 0 {
		logger.Fatal("No snapshot files found", zap.String("dir", *dataDir))
	}
	sort.Strings(files)

	logger.Info("Found snapshot files", zap.Int("count", len(files)))

	// Initialize shared memory writer
	shmWriter, err := shm.NewWriter()
	if err != nil {
		logger.Fatal("Failed to create SHM writer", zap.Error(err))
	}
	defer shmWriter.Close()

	// Replay each file
	var prevTimestamp int64
	totalSnapshots := 0

	for _, file := range files {
		snapshots, err := loadSnapshots(file)
		if err != nil {
			logger.Error("Failed to load snapshots", zap.String("file", file), zap.Error(err))
			continue
		}

		logger.Info("Replaying file", zap.String("file", file), zap.Int("snapshots", len(snapshots)))

		for _, snap := range snapshots {
			// Calculate delay
			if prevTimestamp > 0 {
				delay := time.Duration(float64(snap.TimestampNs-prevTimestamp) / *speed)
				if delay > 0 && delay < 10*time.Second {
					time.Sleep(delay)
				}
			}
			prevTimestamp = snap.TimestampNs

			// Write to shared memory
			writeSnapshot(shmWriter, &snap)
			totalSnapshots++
		}
	}

	logger.Info("Replay complete", zap.Int("total_snapshots", totalSnapshots))
}

func loadSnapshots(path string) ([]Snapshot, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var snapshots []Snapshot
	if err := json.Unmarshal(data, &snapshots); err != nil {
		return nil, err
	}

	return snapshots, nil
}

func writeSnapshot(w *shm.Writer, snap *Snapshot) {
	// Write Polymarket orderbooks
	for assetID, book := range snap.PolyBooks {
		mb := shm.MarketBook{
			AssetID:     shm.StringToAssetID(assetID),
			TimestampNs: uint64(snap.TimestampNs),
		}

		// Convert bids
		for i, level := range book.Bids {
			if i >= shm.MaxOrderbookLevels {
				break
			}
			var price, size float64
			fmt.Sscanf(level.Price, "%f", &price)
			fmt.Sscanf(level.Size, "%f", &size)
			mb.Bids[i] = shm.PriceLevel{Price: price, Size: size}
			mb.BidLevels++
		}

		// Convert asks
		for i, level := range book.Asks {
			if i >= shm.MaxOrderbookLevels {
				break
			}
			var price, size float64
			fmt.Sscanf(level.Price, "%f", &price)
			fmt.Sscanf(level.Size, "%f", &size)
			mb.Asks[i] = shm.PriceLevel{Price: price, Size: size}
			mb.AskLevels++
		}

		// Calculate mid and spread
		if mb.BidLevels > 0 && mb.AskLevels > 0 {
			mb.MidPrice = (mb.Bids[0].Price + mb.Asks[0].Price) / 2
			mb.Spread = mb.Asks[0].Price - mb.Bids[0].Price
		}

		w.UpdateMarket(mb)
	}

	// Write Binance prices
	for symbol, price := range snap.BinancePrices {
		ep := shm.ExternalPrice{
			Symbol:      shm.StringToSymbol(symbol),
			Price:       price.Price,
			Bid:         price.BidPrice,
			Ask:         price.AskPrice,
			TimestampNs: uint64(snap.TimestampNs),
		}
		w.UpdateExternalPrice(ep)
	}
}
