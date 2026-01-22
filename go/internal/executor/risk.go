package executor

import (
	"fmt"
	"sync"
)

// RiskManager enforces risk limits on orders
type RiskManager struct {
	mu sync.RWMutex

	// Position limits per asset
	maxPositionPerAsset float64

	// Total exposure limit
	maxTotalExposure float64

	// Order limits
	maxOrderSize float64
	minOrderSize float64

	// Current state
	positions      map[string]float64
	totalExposure  float64
	tradingEnabled bool
}

// RiskConfig configures the risk manager
type RiskConfig struct {
	MaxPositionPerAsset float64
	MaxTotalExposure    float64
	MaxOrderSize        float64
	MinOrderSize        float64
}

// NewRiskManager creates a new risk manager
func NewRiskManager(cfg RiskConfig) *RiskManager {
	if cfg.MaxPositionPerAsset == 0 {
		cfg.MaxPositionPerAsset = 500
	}
	if cfg.MaxTotalExposure == 0 {
		cfg.MaxTotalExposure = 5000
	}
	if cfg.MaxOrderSize == 0 {
		cfg.MaxOrderSize = 100
	}
	if cfg.MinOrderSize == 0 {
		cfg.MinOrderSize = 1
	}

	return &RiskManager{
		maxPositionPerAsset: cfg.MaxPositionPerAsset,
		maxTotalExposure:    cfg.MaxTotalExposure,
		maxOrderSize:        cfg.MaxOrderSize,
		minOrderSize:        cfg.MinOrderSize,
		positions:           make(map[string]float64),
		tradingEnabled:      true,
	}
}

// CheckOrder validates an order against risk limits
func (r *RiskManager) CheckOrder(assetID string, side int8, size float64) error {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.tradingEnabled {
		return fmt.Errorf("trading disabled")
	}

	// Check order size
	if size < r.minOrderSize {
		return fmt.Errorf("order size %f below minimum %f", size, r.minOrderSize)
	}
	if size > r.maxOrderSize {
		return fmt.Errorf("order size %f exceeds maximum %f", size, r.maxOrderSize)
	}

	// Check position limit
	currentPos := r.positions[assetID]
	var newPos float64
	if side == 1 { // Buy
		newPos = currentPos + size
	} else { // Sell
		newPos = currentPos - size
	}

	if abs(newPos) > r.maxPositionPerAsset {
		return fmt.Errorf("position %f would exceed limit %f", newPos, r.maxPositionPerAsset)
	}

	// Check total exposure
	newExposure := r.totalExposure
	// Simplified: add absolute position change
	newExposure += abs(newPos) - abs(currentPos)

	if newExposure > r.maxTotalExposure {
		return fmt.Errorf("total exposure %f would exceed limit %f", newExposure, r.maxTotalExposure)
	}

	return nil
}

// UpdatePosition updates the position for an asset
func (r *RiskManager) UpdatePosition(assetID string, position float64) {
	r.mu.Lock()
	defer r.mu.Unlock()

	old := r.positions[assetID]
	r.positions[assetID] = position

	// Update total exposure
	r.totalExposure += abs(position) - abs(old)
}

// UpdatePositions updates all positions
func (r *RiskManager) UpdatePositions(positions map[string]float64) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.totalExposure = 0
	r.positions = make(map[string]float64, len(positions))

	for assetID, pos := range positions {
		r.positions[assetID] = pos
		r.totalExposure += abs(pos)
	}
}

// SetTradingEnabled enables or disables trading
func (r *RiskManager) SetTradingEnabled(enabled bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tradingEnabled = enabled
}

// IsTradingEnabled returns whether trading is enabled
func (r *RiskManager) IsTradingEnabled() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.tradingEnabled
}

// GetTotalExposure returns the current total exposure
func (r *RiskManager) GetTotalExposure() float64 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.totalExposure
}

// GetPosition returns the position for an asset
func (r *RiskManager) GetPosition(assetID string) float64 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.positions[assetID]
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
