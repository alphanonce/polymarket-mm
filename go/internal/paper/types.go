// Package paper provides paper trading data types and Supabase integration.
package paper

import (
	"time"
)

// SlugLen is the maximum length of a market slug
const SlugLen = 64

// Position represents a paper trading position
type Position struct {
	ID            string    `json:"id,omitempty"`
	AssetID       string    `json:"asset_id"`
	Slug          string    `json:"slug,omitempty"`
	Side          string    `json:"side,omitempty"` // "up" or "down"
	Size          float64   `json:"size"`
	AvgEntryPrice float64   `json:"avg_entry_price"`
	UnrealizedPnL float64   `json:"unrealized_pnl"`
	RealizedPnL   float64   `json:"realized_pnl"`
	UpdatedAt     time.Time `json:"updated_at,omitempty"`
}

// Trade represents a paper trading trade
type Trade struct {
	ID        string    `json:"id,omitempty"`
	AssetID   string    `json:"asset_id"`
	Slug      string    `json:"slug"`
	Side      int       `json:"side"` // 1=buy, -1=sell
	Price     float64   `json:"price"`
	Size      float64   `json:"size"`
	PnL       float64   `json:"pnl"`
	Timestamp time.Time `json:"timestamp"`
	CreatedAt time.Time `json:"created_at,omitempty"`
}

// EquitySnapshot represents a point-in-time equity snapshot
type EquitySnapshot struct {
	ID            string    `json:"id,omitempty"`
	Equity        float64   `json:"equity"`
	Cash          float64   `json:"cash"`
	PositionValue float64   `json:"position_value"`
	Timestamp     time.Time `json:"timestamp"`
	CreatedAt     time.Time `json:"created_at,omitempty"`
}

// Metrics represents aggregated trading metrics
type Metrics struct {
	ID            int       `json:"id"`
	TotalPnL      float64   `json:"total_pnl"`
	RealizedPnL   float64   `json:"realized_pnl"`
	UnrealizedPnL float64   `json:"unrealized_pnl"`
	TotalTrades   int       `json:"total_trades"`
	WinRate       float64   `json:"win_rate"`
	SharpeRatio   float64   `json:"sharpe_ratio"`
	MaxDrawdown   float64   `json:"max_drawdown"`
	UpdatedAt     time.Time `json:"updated_at,omitempty"`
}

// Market represents a trading period/market
type Market struct {
	ID        string    `json:"id,omitempty"`
	Slug      string    `json:"slug"`
	Asset     string    `json:"asset"`
	Timeframe string    `json:"timeframe"`
	Status    string    `json:"status"` // "active", "resolved"
	Outcome   string    `json:"outcome,omitempty"`
	StartTs   int64     `json:"start_ts,omitempty"`
	EndTs     int64     `json:"end_ts,omitempty"`
	PeriodPnL float64   `json:"period_pnl"`
	CreatedAt time.Time `json:"created_at,omitempty"`
}

// MarketSnapshot represents a point-in-time market state snapshot
type MarketSnapshot struct {
	ID             string    `json:"id,omitempty"`
	Slug           string    `json:"slug"`
	Timestamp      time.Time `json:"timestamp"`
	BestBid        float64   `json:"best_bid,omitempty"`
	BestAsk        float64   `json:"best_ask,omitempty"`
	MidPrice       float64   `json:"mid_price,omitempty"`
	Spread         float64   `json:"spread,omitempty"`
	OurBid         float64   `json:"our_bid,omitempty"`
	OurAsk         float64   `json:"our_ask,omitempty"`
	Inventory      float64   `json:"inventory"`
	InventoryValue float64   `json:"inventory_value"`
	PeriodPnL      float64   `json:"period_pnl"`
	CreatedAt      time.Time `json:"created_at,omitempty"`
}

// PaperPosition is the SHM representation of a paper position
type PaperPosition struct {
	AssetID       [66]byte // AssetIDLen
	Slug          [SlugLen]byte
	Side          [8]byte // "up" or "down"
	Size          float64
	AvgEntryPrice float64
	UnrealizedPnL float64
	RealizedPnL   float64
	UpdatedAtNs   uint64
	_padding      [6]byte
}

// PaperQuote is the SHM representation of our current quote for a market
type PaperQuote struct {
	Slug      [SlugLen]byte
	OurBid    float64
	OurAsk    float64
	BestBid   float64
	BestAsk   float64
	MidPrice  float64
	Spread    float64
	Inventory float64
	UpdatedNs uint64
}

// PaperTrade is the SHM representation of a trade to be persisted
type PaperTrade struct {
	AssetID     [66]byte
	Slug        [SlugLen]byte
	Side        int8
	Price       float64
	Size        float64
	PnL         float64
	TimestampNs uint64
	Persisted   uint8 // 0=pending, 1=persisted
	_padding    [6]byte
}

// MaxPaperPositions is the maximum number of paper positions in SHM
const MaxPaperPositions = 64

// MaxPaperQuotes is the maximum number of paper quotes in SHM
const MaxPaperQuotes = 64

// MaxPaperTrades is the maximum number of pending trades in SHM
const MaxPaperTrades = 128

// PaperTradingState is the SHM layout for paper trading state
// This extends the main SharedMemoryLayout
type PaperTradingState struct {
	// Magic number for validation
	Magic   uint32
	Version uint32

	// Sequence number for change detection
	StateSequence uint32
	_padding0     uint32

	// Positions (written by Python, read by Go)
	NumPositions uint32
	_padding1    uint32
	Positions    [MaxPaperPositions]PaperPosition

	// Current quotes (for market snapshots)
	NumQuotes uint32
	_padding2 uint32
	Quotes    [MaxPaperQuotes]PaperQuote

	// Pending trades (written by Python, consumed by Go)
	NumTrades    uint32
	TradesHead   uint32 // Ring buffer head (next to consume)
	TradesTail   uint32 // Ring buffer tail (next to write)
	_padding3    uint32
	Trades       [MaxPaperTrades]PaperTrade

	// Equity state (written by Python)
	TotalEquity   float64
	Cash          float64
	PositionValue float64

	// Metrics (written by Python)
	TotalPnL      float64
	RealizedPnL   float64
	UnrealizedPnL float64
	TotalTrades   uint32
	WinCount      uint32
	SharpeRatio   float64
	MaxDrawdown   float64

	// Timestamps
	LastUpdateNs      uint64
	LastSnapshotNs    uint64
	LastMetricsNs     uint64
}

// PaperSHMMagic is the magic number for paper trading SHM
const PaperSHMMagic uint32 = 0x50415052 // "PAPR"

// PaperSHMVersion is the current version of paper trading SHM layout
const PaperSHMVersion uint32 = 1

// PaperSHMName is the name of the paper trading shared memory file
const PaperSHMName = "/polymarket_paper_shm"
