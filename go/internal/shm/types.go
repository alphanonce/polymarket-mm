// Package shm provides shared memory types and utilities for inter-process communication
// between the Go executor and Python strategy layer.
//
// This package implements the memory layout defined in shared/shm_layout.h
package shm

import (
	"unsafe"
)

// Constants matching shared/shm_layout.h
const (
	SHMMagic   uint32 = 0x504D4D4D // "PMMM"
	SHMVersion uint32 = 1
	SHMName    string = "/polymarket_mm_shm"

	MaxMarkets         = 64
	MaxOrderbookLevels = 20
	MaxExternalPrices  = 32
	MaxPositions       = 64
	MaxSignals         = 16
	MaxOpenOrders      = 128

	AssetIDLen  = 66
	SymbolLen   = 16
	OrderIDLen  = 64
)

// Side constants
const (
	SideBuy  int8 = 1
	SideSell int8 = -1
)

// Order type constants
const (
	OrderTypeLimit  uint8 = 0
	OrderTypeMarket uint8 = 1
)

// Signal action constants
const (
	ActionPlace  uint8 = 0
	ActionCancel uint8 = 1
	ActionModify uint8 = 2
)

// Order status constants
const (
	OrderStatusPending   uint8 = 0
	OrderStatusOpen      uint8 = 1
	OrderStatusFilled    uint8 = 2
	OrderStatusCancelled uint8 = 3
	OrderStatusRejected  uint8 = 4
)

// PriceLevel represents a single price level in the orderbook
type PriceLevel struct {
	Price float64
	Size  float64
}

// MarketBook represents the orderbook state for a single market
type MarketBook struct {
	AssetID        [AssetIDLen]byte
	TimestampNs    uint64
	MidPrice       float64
	Spread         float64
	Bids           [MaxOrderbookLevels]PriceLevel
	Asks           [MaxOrderbookLevels]PriceLevel
	BidLevels      uint32
	AskLevels      uint32
	LastTradePrice float64
	LastTradeSize  float64
	LastTradeSide  int8
	_padding       [7]byte
}

// ExternalPrice represents an external price feed
type ExternalPrice struct {
	Symbol      [SymbolLen]byte
	Price       float64
	Bid         float64
	Ask         float64
	TimestampNs uint64
}

// Position represents a position in a market
type Position struct {
	AssetID       [AssetIDLen]byte
	Position      float64
	AvgEntryPrice float64
	UnrealizedPnL float64
	RealizedPnL   float64
}

// OpenOrder represents an open order
type OpenOrder struct {
	OrderID     [OrderIDLen]byte
	AssetID     [AssetIDLen]byte
	Side        int8
	Price       float64
	Size        float64
	FilledSize  float64
	Status      uint8
	CreatedAtNs uint64
	UpdatedAtNs uint64
	_padding    [6]byte
}

// OrderSignal represents an order signal from strategy to executor
type OrderSignal struct {
	SignalID      uint64
	AssetID       [AssetIDLen]byte
	Side          int8
	Price         float64
	Size          float64
	OrderType     uint8
	Action        uint8
	CancelOrderID [OrderIDLen]byte
	_padding      [5]byte
}

// SharedMemoryLayout is the main shared memory structure
type SharedMemoryLayout struct {
	// Header
	Magic   uint32
	Version uint32

	// Synchronization
	StateSequence  uint32
	SignalSequence uint32

	// Timestamps
	StateTimestampNs  uint64
	SignalTimestampNs uint64

	// Market state
	NumMarkets uint32
	_padding1  uint32
	Markets    [MaxMarkets]MarketBook

	// External prices
	NumExternalPrices uint32
	_padding2         uint32
	ExternalPrices    [MaxExternalPrices]ExternalPrice

	// Positions
	NumPositions uint32
	_padding3    uint32
	Positions    [MaxPositions]Position

	// Open orders
	NumOpenOrders uint32
	_padding4     uint32
	OpenOrders    [MaxOpenOrders]OpenOrder

	// Strategy state
	TotalEquity     float64
	AvailableMargin float64
	TradingEnabled  uint8
	_padding5       [7]byte

	// Order signals
	NumSignals       uint32
	SignalsProcessed uint32
	Signals          [MaxSignals]OrderSignal
}

// SHMSize returns the size of the shared memory layout
func SHMSize() int {
	return int(unsafe.Sizeof(SharedMemoryLayout{}))
}

// Helper functions for string conversion

// AssetIDToString converts a fixed-size asset ID to string
func AssetIDToString(id [AssetIDLen]byte) string {
	for i, b := range id {
		if b == 0 {
			return string(id[:i])
		}
	}
	return string(id[:])
}

// StringToAssetID converts a string to fixed-size asset ID
func StringToAssetID(s string) [AssetIDLen]byte {
	var id [AssetIDLen]byte
	copy(id[:], s)
	return id
}

// SymbolToString converts a fixed-size symbol to string
func SymbolToString(sym [SymbolLen]byte) string {
	for i, b := range sym {
		if b == 0 {
			return string(sym[:i])
		}
	}
	return string(sym[:])
}

// StringToSymbol converts a string to fixed-size symbol
func StringToSymbol(s string) [SymbolLen]byte {
	var sym [SymbolLen]byte
	copy(sym[:], s)
	return sym
}

// OrderIDToString converts a fixed-size order ID to string
func OrderIDToString(id [OrderIDLen]byte) string {
	for i, b := range id {
		if b == 0 {
			return string(id[:i])
		}
	}
	return string(id[:])
}

// StringToOrderID converts a string to fixed-size order ID
func StringToOrderID(s string) [OrderIDLen]byte {
	var id [OrderIDLen]byte
	copy(id[:], s)
	return id
}
