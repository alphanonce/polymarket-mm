// Package executor handles order execution and management
package executor

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/alphanonce/polymarket-mm/internal/polymarket"
	"github.com/alphanonce/polymarket-mm/internal/shm"
	"go.uber.org/zap"
)

// Executor reads signals from shared memory and executes orders
type Executor struct {
	logger    *zap.Logger
	shmWriter *shm.Writer
	client    *polymarket.Client
	risk      *RiskManager

	// Order tracking
	pendingOrders map[uint64]string // signal_id -> order_id
	ordersMu      sync.RWMutex

	// Position tracking
	positions   map[string]float64 // asset_id -> position
	positionsMu sync.RWMutex

	// Control
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	// Config
	pollInterval time.Duration
}

// Config configures the executor
type Config struct {
	Logger       *zap.Logger
	SHMWriter    *shm.Writer
	Client       *polymarket.Client
	RiskManager  *RiskManager
	PollInterval time.Duration
}

// New creates a new executor
func New(cfg Config) *Executor {
	if cfg.Logger == nil {
		cfg.Logger, _ = zap.NewProduction()
	}
	if cfg.PollInterval == 0 {
		cfg.PollInterval = 1 * time.Millisecond
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &Executor{
		logger:        cfg.Logger,
		shmWriter:     cfg.SHMWriter,
		client:        cfg.Client,
		risk:          cfg.RiskManager,
		pendingOrders: make(map[uint64]string),
		positions:     make(map[string]float64),
		ctx:           ctx,
		cancel:        cancel,
		pollInterval:  cfg.PollInterval,
	}
}

// Start starts the executor loop
func (e *Executor) Start() error {
	e.logger.Info("Starting executor")

	// Start signal processing loop
	e.wg.Add(1)
	go e.processSignalsLoop()

	// Start position sync loop
	e.wg.Add(1)
	go e.syncPositionsLoop()

	return nil
}

// Stop stops the executor
func (e *Executor) Stop() {
	e.logger.Info("Stopping executor")
	e.cancel()
	e.wg.Wait()
}

func (e *Executor) processSignalsLoop() {
	defer e.wg.Done()

	ticker := time.NewTicker(e.pollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-e.ctx.Done():
			return
		case <-ticker.C:
			e.processSignals()
		}
	}
}

func (e *Executor) processSignals() {
	signals := e.shmWriter.ReadSignals()
	if len(signals) == 0 {
		return
	}

	for _, sig := range signals {
		if err := e.processSignal(sig); err != nil {
			e.logger.Error("Failed to process signal",
				zap.Uint64("signalID", sig.SignalID),
				zap.Error(err))
		}
	}

	e.shmWriter.MarkSignalsProcessed()
}

func (e *Executor) processSignal(sig shm.OrderSignal) error {
	assetID := shm.AssetIDToString(sig.AssetID)

	switch sig.Action {
	case shm.ActionPlace:
		return e.placeOrder(sig, assetID)
	case shm.ActionCancel:
		return e.cancelOrder(sig, assetID)
	case shm.ActionModify:
		return e.modifyOrder(sig, assetID)
	default:
		e.logger.Warn("Unknown action", zap.Uint8("action", sig.Action))
		return nil
	}
}

func (e *Executor) placeOrder(sig shm.OrderSignal, assetID string) error {
	// Risk check
	if e.risk != nil {
		if err := e.risk.CheckOrder(assetID, sig.Side, sig.Size); err != nil {
			e.logger.Warn("Order rejected by risk manager",
				zap.String("assetID", assetID),
				zap.Error(err))
			return nil
		}
	}

	// Convert side
	side := "BUY"
	if sig.Side == shm.SideSell {
		side = "SELL"
	}

	// Place order
	order := polymarket.Order{
		TokenID:   assetID,
		Price:     sig.Price,
		Size:      sig.Size,
		Side:      side,
		OrderType: "GTC",
	}

	resp, err := e.client.PlaceOrder(e.ctx, order)
	if err != nil {
		return err
	}

	// Track order
	e.ordersMu.Lock()
	e.pendingOrders[sig.SignalID] = resp.OrderID
	e.ordersMu.Unlock()

	// Update SHM with open order
	openOrder := shm.OpenOrder{
		OrderID:     shm.StringToOrderID(resp.OrderID),
		AssetID:     sig.AssetID,
		Side:        sig.Side,
		Price:       sig.Price,
		Size:        sig.Size,
		FilledSize:  0,
		Status:      shm.OrderStatusOpen,
	}
	e.shmWriter.AddOpenOrder(openOrder)

	e.logger.Info("Order placed",
		zap.String("orderID", resp.OrderID),
		zap.String("assetID", assetID),
		zap.String("side", side),
		zap.Float64("price", sig.Price),
		zap.Float64("size", sig.Size))

	return nil
}

func (e *Executor) cancelOrder(sig shm.OrderSignal, assetID string) error {
	orderID := shm.OrderIDToString(sig.CancelOrderID)
	if orderID == "" {
		e.logger.Warn("No order ID for cancel")
		return nil
	}

	_, err := e.client.CancelOrder(e.ctx, orderID)
	if err != nil {
		return err
	}

	// Update SHM
	e.shmWriter.UpdateOpenOrder(orderID, shm.OrderStatusCancelled, 0)
	e.shmWriter.RemoveOpenOrder(orderID)

	e.logger.Info("Order cancelled",
		zap.String("orderID", orderID),
		zap.String("assetID", assetID))

	return nil
}

func (e *Executor) modifyOrder(sig shm.OrderSignal, assetID string) error {
	// Modify = cancel + place
	orderID := shm.OrderIDToString(sig.CancelOrderID)

	// Cancel existing
	if orderID != "" {
		if _, err := e.client.CancelOrder(e.ctx, orderID); err != nil {
			e.logger.Warn("Failed to cancel for modify", zap.Error(err))
		}
		e.shmWriter.RemoveOpenOrder(orderID)
	}

	// Place new
	return e.placeOrder(sig, assetID)
}

func (e *Executor) syncPositionsLoop() {
	defer e.wg.Done()

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-e.ctx.Done():
			return
		case <-ticker.C:
			e.syncPositions()
		}
	}
}

func (e *Executor) syncPositions() {
	// Fetch positions from API
	positions, err := e.client.GetPositions(e.ctx)
	if err != nil {
		e.logger.Error("Failed to fetch positions", zap.Error(err))
		return
	}

	// Update SHM
	for _, pos := range positions {
		var posSize float64
		if pos.Size != "" {
			_, _ = fmt.Sscanf(pos.Size, "%f", &posSize)
		}

		shmPos := shm.Position{
			AssetID:  shm.StringToAssetID(pos.AssetID),
			Position: posSize,
		}
		e.shmWriter.UpdatePosition(shmPos)

		e.positionsMu.Lock()
		e.positions[pos.AssetID] = posSize
		e.positionsMu.Unlock()
	}

	// Fetch and update balance
	balance, err := e.client.GetBalance(e.ctx)
	if err != nil {
		e.logger.Error("Failed to fetch balance", zap.Error(err))
		return
	}

	var equity float64
	if balance.Collateral != "" {
		_, _ = fmt.Sscanf(balance.Collateral, "%f", &equity)
	}
	e.shmWriter.SetEquity(equity, equity)
}

// GetPosition returns the current position for an asset
func (e *Executor) GetPosition(assetID string) float64 {
	e.positionsMu.RLock()
	defer e.positionsMu.RUnlock()
	return e.positions[assetID]
}
