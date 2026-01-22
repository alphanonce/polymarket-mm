// Package aggregator combines data from multiple sources and writes to shared memory
package aggregator

import (
	"context"
	"strconv"
	"sync"
	"time"

	"github.com/alphanonce/polymarket-mm/internal/binance"
	"github.com/alphanonce/polymarket-mm/internal/chainlink"
	"github.com/alphanonce/polymarket-mm/internal/polymarket"
	"github.com/alphanonce/polymarket-mm/internal/shm"
	"go.uber.org/zap"
)

// Aggregator combines data from all sources and writes to shared memory
type Aggregator struct {
	logger *zap.Logger
	shm    *shm.Writer

	// Data sources
	polyWS    *polymarket.WSClient
	binanceWS *binance.SpotClient
	chainlink *chainlink.Reader

	// Market mappings (Polymarket asset ID -> external symbol)
	marketMappings map[string]string
	mappingsMu     sync.RWMutex

	// Control
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// Config configures the aggregator
type Config struct {
	Logger         *zap.Logger
	SHMWriter      *shm.Writer
	PolyWS         *polymarket.WSClient
	BinanceWS      *binance.SpotClient
	ChainlinkReader *chainlink.Reader
}

// New creates a new aggregator
func New(cfg Config) *Aggregator {
	if cfg.Logger == nil {
		cfg.Logger, _ = zap.NewProduction()
	}

	ctx, cancel := context.WithCancel(context.Background())

	agg := &Aggregator{
		logger:         cfg.Logger,
		shm:            cfg.SHMWriter,
		polyWS:         cfg.PolyWS,
		binanceWS:      cfg.BinanceWS,
		chainlink:      cfg.ChainlinkReader,
		marketMappings: make(map[string]string),
		ctx:            ctx,
		cancel:         cancel,
	}

	// Set up callbacks
	if cfg.PolyWS != nil {
		cfg.PolyWS.OnBookUpdate(agg.onPolyBookUpdate)
		cfg.PolyWS.OnTrade(agg.onPolyTrade)
	}

	if cfg.BinanceWS != nil {
		cfg.BinanceWS.OnPriceUpdate(agg.onBinancePriceUpdate)
	}

	if cfg.ChainlinkReader != nil {
		cfg.ChainlinkReader.OnPriceUpdate(agg.onChainlinkPriceUpdate)
	}

	return agg
}

// AddMarketMapping maps a Polymarket asset ID to an external symbol
func (a *Aggregator) AddMarketMapping(assetID, externalSymbol string) {
	a.mappingsMu.Lock()
	a.marketMappings[assetID] = externalSymbol
	a.mappingsMu.Unlock()
}

// Start starts the aggregator
func (a *Aggregator) Start() error {
	a.logger.Info("Starting aggregator")

	// Start periodic state sync
	a.wg.Add(1)
	go a.syncLoop()

	return nil
}

// Stop stops the aggregator
func (a *Aggregator) Stop() {
	a.logger.Info("Stopping aggregator")
	a.cancel()
	a.wg.Wait()
}

func (a *Aggregator) syncLoop() {
	defer a.wg.Done()

	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-a.ctx.Done():
			return
		case <-ticker.C:
			// Periodic sync if needed
		}
	}
}

func (a *Aggregator) onPolyBookUpdate(assetID string, book *polymarket.OrderbookState) {
	if a.shm == nil {
		return
	}

	marketBook := a.convertPolyBook(assetID, book)
	if err := a.shm.UpdateMarket(marketBook); err != nil {
		a.logger.Error("Failed to update market in SHM",
			zap.String("assetID", assetID),
			zap.Error(err))
	}
}

func (a *Aggregator) onPolyTrade(trade polymarket.Trade) {
	// Update last trade info in the market book
	if a.shm == nil {
		return
	}

	// Get current market and update trade info
	layout := a.shm.Layout()
	for i := 0; i < int(layout.NumMarkets); i++ {
		if shm.AssetIDToString(layout.Markets[i].AssetID) == trade.AssetID {
			price, _ := strconv.ParseFloat(trade.Price, 64)
			size, _ := strconv.ParseFloat(trade.Size, 64)

			layout.Markets[i].LastTradePrice = price
			layout.Markets[i].LastTradeSize = size
			if trade.Side == "BUY" {
				layout.Markets[i].LastTradeSide = shm.SideBuy
			} else {
				layout.Markets[i].LastTradeSide = shm.SideSell
			}
			break
		}
	}
}

func (a *Aggregator) onBinancePriceUpdate(symbol string, price *binance.TickerPrice) {
	if a.shm == nil {
		return
	}

	extPrice := shm.ExternalPrice{
		Symbol:      shm.StringToSymbol(symbol),
		Price:       price.Price,
		Bid:         price.BidPrice,
		Ask:         price.AskPrice,
		TimestampNs: uint64(price.Timestamp.UnixNano()),
	}

	if err := a.shm.UpdateExternalPrice(extPrice); err != nil {
		a.logger.Error("Failed to update external price in SHM",
			zap.String("symbol", symbol),
			zap.Error(err))
	}
}

func (a *Aggregator) onChainlinkPriceUpdate(symbol string, data *chainlink.PriceData) {
	if a.shm == nil {
		return
	}

	// Chainlink prices are typically formatted as "BTC/USD"
	// Convert to simple format for SHM
	extPrice := shm.ExternalPrice{
		Symbol:      shm.StringToSymbol(symbol),
		Price:       data.Price,
		Bid:         data.Price, // Chainlink doesn't have bid/ask
		Ask:         data.Price,
		TimestampNs: uint64(data.FetchedAt.UnixNano()),
	}

	if err := a.shm.UpdateExternalPrice(extPrice); err != nil {
		a.logger.Error("Failed to update Chainlink price in SHM",
			zap.String("symbol", symbol),
			zap.Error(err))
	}
}

func (a *Aggregator) convertPolyBook(assetID string, book *polymarket.OrderbookState) shm.MarketBook {
	mb := shm.MarketBook{
		AssetID:     shm.StringToAssetID(assetID),
		TimestampNs: uint64(time.Now().UnixNano()),
	}

	// Convert bids
	bidIdx := 0
	for price, size := range book.Bids {
		if bidIdx >= shm.MaxOrderbookLevels {
			break
		}
		p, _ := strconv.ParseFloat(price, 64)
		s, _ := strconv.ParseFloat(size, 64)
		mb.Bids[bidIdx] = shm.PriceLevel{Price: p, Size: s}
		bidIdx++
	}
	mb.BidLevels = uint32(bidIdx)

	// Convert asks
	askIdx := 0
	for price, size := range book.Asks {
		if askIdx >= shm.MaxOrderbookLevels {
			break
		}
		p, _ := strconv.ParseFloat(price, 64)
		s, _ := strconv.ParseFloat(size, 64)
		mb.Asks[askIdx] = shm.PriceLevel{Price: p, Size: s}
		askIdx++
	}
	mb.AskLevels = uint32(askIdx)

	// Calculate mid price and spread
	if mb.BidLevels > 0 && mb.AskLevels > 0 {
		bestBid := mb.Bids[0].Price
		bestAsk := mb.Asks[0].Price
		mb.MidPrice = (bestBid + bestAsk) / 2
		mb.Spread = bestAsk - bestBid
	}

	return mb
}
